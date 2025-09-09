"""
Query Processor Agent - Processes user queries and extracts requirements.
"""
import json
import re
from typing import Dict, Any, List, Optional
import asyncio

from ..models.agentic_models import (
    AgentTask, AgentResponse, AgentRole, UserQuery, ProcessedQuery
)
from .base_agent import BaseAgent
from ..prompts.prompt_manager import prompt_manager


class QueryProcessorAgent(BaseAgent):
    """Agent responsible for processing user queries and extracting requirements."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = AgentRole.QUERY_PROCESSOR
        
        # Common patterns for requirement extraction
        self.age_patterns = [
            r"(?:age|aged)\s*(?:between\s*)?(\d+)(?:\s*(?:to|and|-)\s*(\d+))?",
            r"(\d+)\s*(?:to|-)\s*(\d+)\s*years?\s*old",
            r"around\s*(\d+)",
            r"(\d+)\s*[+-]\s*(?:years?|yrs?)"
        ]
        
        self.profession_keywords = [
            "engineer", "developer", "programmer", "software", "tech", "IT",
            "doctor", "physician", "nurse", "medical", "healthcare",
            "teacher", "professor", "educator", "academic",
            "lawyer", "attorney", "legal",
            "manager", "executive", "director", "CEO", "CTO",
            "designer", "artist", "creative", "marketing",
            "consultant", "analyst", "researcher",
            "entrepreneur", "startup", "business owner"
        ]
        
        self.interest_keywords = [
            "hiking", "travel", "photography", "music", "reading", "books",
            "sports", "fitness", "gym", "running", "cycling", "swimming",
            "cooking", "food", "wine", "coffee",
            "movies", "theater", "art", "museums",
            "gaming", "technology", "programming",
            "yoga", "meditation", "mindfulness",
            "dancing", "singing", "instrument",
            "outdoors", "nature", "camping", "adventure"
        ]
        
        self.location_patterns = [
            r"(?:in|from|near|around)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+area",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+based"
        ]
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for query processing."""
        return prompt_manager.get_agent_prompt("query_processor")
    
    async def process_task(self, task: AgentTask) -> AgentResponse:
        """Process query analysis task."""
        try:
            # Extract input data
            query_data = task.input_data.get("query")
            if not query_data:
                raise ValueError("No query data provided")
            
            # Convert to UserQuery if it's a dict
            if isinstance(query_data, dict):
                user_query = UserQuery(**query_data)
            else:
                user_query = query_data
            
            # Process the query
            processed_query = await self._process_user_query(user_query)
            
            return AgentResponse(
                agent_name=self.name,
                agent_role=self.role,
                task_id=task.task_id,
                success=True,
                response_data={
                    "processed_query": processed_query.dict(),
                    "original_query": user_query.query_text
                },
                confidence_score=processed_query.confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            raise
    
    async def _process_user_query(self, user_query: UserQuery) -> ProcessedQuery:
        """Process user query and extract structured requirements."""
        query_text = user_query.query_text.lower()
        
        # Extract requirements using rule-based approach
        rule_based_requirements = self._extract_requirements_rule_based(query_text)
        
        # Use LLM for more nuanced understanding
        llm_requirements = await self._extract_requirements_llm(user_query.query_text)
        
        # Combine and validate results
        structured_requirements = self._combine_requirements(
            rule_based_requirements, 
            llm_requirements
        )
        
        # Generate search keywords
        search_keywords = self._generate_search_keywords(query_text, structured_requirements)
        
        # Identify priority factors
        priority_factors = self._identify_priority_factors(query_text, structured_requirements)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            structured_requirements, 
            search_keywords,
            user_query.query_text
        )
        
        return ProcessedQuery(
            query_id=user_query.query_id,
            original_query=user_query.query_text,
            structured_requirements=structured_requirements,
            search_keywords=search_keywords,
            priority_factors=priority_factors,
            confidence_score=confidence_score,
            processing_agent=self.name,
            processing_time_ms=0.0  # Will be updated by base class
        )
    
    def _extract_requirements_rule_based(self, query_text: str) -> Dict[str, Any]:
        """Extract requirements using rule-based patterns."""
        requirements = {}
        
        # Extract age preferences
        age_info = self._extract_age_preferences(query_text)
        if age_info:
            requirements["age_range"] = age_info
        
        # Extract location preferences
        location_info = self._extract_location_preferences(query_text)
        if location_info:
            requirements["location"] = location_info
        
        # Extract profession preferences
        profession_info = self._extract_profession_preferences(query_text)
        if profession_info:
            requirements["profession"] = profession_info
        
        # Extract interest preferences
        interest_info = self._extract_interest_preferences(query_text)
        if interest_info:
            requirements["interests"] = interest_info
        
        # Extract education preferences
        education_info = self._extract_education_preferences(query_text)
        if education_info:
            requirements["education"] = education_info
        
        return requirements
    
    def _extract_age_preferences(self, query_text: str) -> Optional[Dict[str, int]]:
        """Extract age preferences from query text."""
        for pattern in self.age_patterns:
            match = re.search(pattern, query_text, re.IGNORECASE)
            if match:
                if match.group(2):  # Range
                    return {
                        "min": int(match.group(1)),
                        "max": int(match.group(2))
                    }
                else:  # Single age
                    age = int(match.group(1))
                    return {
                        "min": max(18, age - 3),
                        "max": age + 3
                    }
        return None
    
    def _extract_location_preferences(self, query_text: str) -> Optional[List[str]]:
        """Extract location preferences from query text."""
        locations = []
        for pattern in self.location_patterns:
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            locations.extend(matches)
        
        return list(set(locations)) if locations else None
    
    def _extract_profession_preferences(self, query_text: str) -> Optional[List[str]]:
        """Extract profession preferences from query text."""
        found_professions = []
        for keyword in self.profession_keywords:
            if keyword.lower() in query_text:
                found_professions.append(keyword)
        
        return found_professions if found_professions else None
    
    def _extract_interest_preferences(self, query_text: str) -> Optional[List[str]]:
        """Extract interest preferences from query text."""
        found_interests = []
        for keyword in self.interest_keywords:
            if keyword.lower() in query_text:
                found_interests.append(keyword)
        
        return found_interests if found_interests else None
    
    def _extract_education_preferences(self, query_text: str) -> Optional[Dict[str, Any]]:
        """Extract education preferences from query text."""
        education_keywords = {
            "bachelor": ["bachelor", "ba", "bs", "undergraduate"],
            "master": ["master", "ma", "ms", "mba", "graduate"],
            "phd": ["phd", "doctorate", "doctoral"],
            "college": ["college", "university", "degree"]
        }
        
        found_education = {}
        for level, keywords in education_keywords.items():
            for keyword in keywords:
                if keyword.lower() in query_text:
                    found_education[level] = True
                    break
        
        return found_education if found_education else None
    
    async def _extract_requirements_llm(self, query_text: str) -> Dict[str, Any]:
        """Use LLM to extract more nuanced requirements."""
        prompt = f"""
Analyze the following user query for bio data matching and extract structured requirements:

Query: "{query_text}"

Please identify and extract:
1. Age preferences (specific ages or ranges)
2. Location preferences (cities, regions, countries)
3. Professional preferences (jobs, industries, career levels)
4. Educational preferences (degree levels, institutions, fields)
5. Interest and hobby preferences
6. Lifestyle preferences (diet, exercise, smoking, drinking)
7. Relationship preferences (casual, serious, marriage, etc.)
8. Physical preferences (if mentioned)
9. Personality traits desired

Respond ONLY with a valid JSON object containing the extracted requirements. Use null for missing information.

Example format:
{{
    "age_range": {{"min": 25, "max": 35}},
    "location": ["New York", "California"],
    "profession": ["software engineer", "technology"],
    "education": {{"minimum_level": "bachelor", "preferred_fields": ["computer science"]}},
    "interests": ["hiking", "photography", "travel"],
    "lifestyle": {{"diet": "vegetarian", "exercise": "regular"}},
    "relationship_type": "serious",
    "personality": ["outgoing", "adventurous"]
}}
"""
        
        try:
            response = await self.generate_response(prompt, max_tokens=800, temperature=0.3)
            
            # Try to parse JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                self.logger.warning("Failed to extract JSON from LLM response")
                return {}
                
        except Exception as e:
            self.logger.error(f"LLM requirement extraction failed: {e}")
            return {}
    
    def _combine_requirements(
        self, 
        rule_based: Dict[str, Any], 
        llm_based: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine rule-based and LLM-based requirements."""
        combined = {}
        
        # Start with rule-based (more reliable for structured data)
        combined.update(rule_based)
        
        # Add LLM insights where rule-based didn't find anything
        for key, value in llm_based.items():
            if value and (key not in combined or not combined[key]):
                combined[key] = value
        
        # Special handling for age range
        if "age_range" in llm_based and "age_range" in rule_based:
            # Use rule-based if available, otherwise LLM
            combined["age_range"] = rule_based["age_range"]
        
        return combined
    
    def _generate_search_keywords(
        self, 
        query_text: str, 
        requirements: Dict[str, Any]
    ) -> List[str]:
        """Generate search keywords for vector search."""
        keywords = []
        
        # Add words from original query (filtered)
        stop_words = {
            "looking", "for", "someone", "who", "is", "the", "a", "an", "and", 
            "or", "but", "in", "on", "at", "to", "from", "with", "by"
        }
        
        query_words = [
            word.strip(".,!?") 
            for word in query_text.lower().split() 
            if word not in stop_words and len(word) > 2
        ]
        keywords.extend(query_words)
        
        # Add keywords from extracted requirements
        for key, value in requirements.items():
            if isinstance(value, list):
                keywords.extend(value)
            elif isinstance(value, str):
                keywords.append(value)
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (str, list)):
                        if isinstance(subvalue, list):
                            keywords.extend(subvalue)
                        else:
                            keywords.append(subvalue)
        
        # Remove duplicates and return
        return list(set(keywords))
    
    def _identify_priority_factors(
        self, 
        query_text: str, 
        requirements: Dict[str, Any]
    ) -> List[str]:
        """Identify priority factors based on query emphasis."""
        priority_factors = []
        
        # Keywords that indicate importance
        importance_indicators = {
            "must": 3,
            "need": 3,
            "require": 3,
            "important": 2,
            "prefer": 2,
            "would like": 1,
            "interested in": 1
        }
        
        # Analyze query for emphasis
        query_lower = query_text.lower()
        
        for factor, weight in importance_indicators.items():
            if factor in query_lower:
                # Find what comes after the importance indicator
                parts = query_lower.split(factor)
                if len(parts) > 1:
                    following_text = parts[1][:50]  # Next 50 characters
                    
                    # Check which requirement categories are mentioned
                    for req_key in requirements.keys():
                        if req_key.replace("_", " ") in following_text:
                            priority_factors.append(f"{req_key}:{weight}")
        
        # If no explicit priorities found, infer from query structure
        if not priority_factors:
            # First mentioned items often have higher priority
            for req_key in list(requirements.keys())[:3]:
                priority_factors.append(f"{req_key}:2")
        
        return priority_factors
    
    def _calculate_confidence_score(
        self, 
        requirements: Dict[str, Any], 
        keywords: List[str],
        original_query: str
    ) -> float:
        """Calculate confidence score for the processed query."""
        score = 0.0
        
        # Base score for having any requirements
        if requirements:
            score += 0.3
        
        # Score for specific requirement types
        structured_requirements = ["age_range", "location", "profession", "education"]
        for req in structured_requirements:
            if req in requirements and requirements[req]:
                score += 0.1
        
        # Score for keyword richness
        if len(keywords) > 5:
            score += 0.2
        elif len(keywords) > 2:
            score += 0.1
        
        # Score for query clarity (length and structure)
        word_count = len(original_query.split())
        if 10 <= word_count <= 50:
            score += 0.2
        elif 5 <= word_count <= 100:
            score += 0.1
        
        # Penalty for very short or very long queries
        if word_count < 3:
            score -= 0.2
        elif word_count > 100:
            score -= 0.1
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
