"""
Bio Matcher Agent - Finds and ranks bio data matches from vector store.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ..models.agentic_models import (
    AgentTask, AgentResponse, AgentRole, ProcessedQuery, BioMatch
)
from .base_agent import BaseAgent
from ..prompts.prompt_manager import prompt_manager

# Import vector store functionality
try:
    from data_vector_store.setup_vector_store import VectorStoreSetup
    from data_vector_store.models.vector_models import SearchQuery
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False


class BioMatcherAgent(BaseAgent):
    """Agent responsible for finding bio data matches using vector store."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = AgentRole.BIO_MATCHER
        
        # Initialize vector store if available
        if VECTOR_STORE_AVAILABLE:
            try:
                self.vector_store = VectorStoreSetup()
                self.logger.info("Vector store initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize vector store: {e}")
                self.vector_store = None
        else:
            self.vector_store = None
            self.logger.warning("Vector store not available")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for bio matching."""
        return prompt_manager.get_agent_prompt("bio_matcher")
    
    async def process_task(self, task: AgentTask) -> AgentResponse:
        """Process bio matching task."""
        try:
            # Extract input data
            processed_query = task.input_data.get("processed_query")
            user_bio_id = task.input_data.get("user_bio_id")
            max_results = task.input_data.get("max_results", 10)
            
            if not processed_query:
                raise ValueError("No processed query provided")
            
            # Convert to ProcessedQuery if it's a dict
            if isinstance(processed_query, dict):
                processed_query = ProcessedQuery(**processed_query)
            
            # Find matches
            matches = await self._find_bio_matches(processed_query, user_bio_id, max_results)
            
            return AgentResponse(
                agent_name=self.name,
                agent_role=self.role,
                task_id=task.task_id,
                success=True,
                response_data={
                    "matches": [match.dict() for match in matches],
                    "total_found": len(matches),
                    "search_method": "vector_similarity" if self.vector_store else "mock"
                },
                confidence_score=self._calculate_matching_confidence(matches)
            )
            
        except Exception as e:
            self.logger.error(f"Bio matching failed: {e}")
            raise
    
    async def _find_bio_matches(
        self, 
        processed_query: ProcessedQuery, 
        user_bio_id: str,
        max_results: int
    ) -> List[BioMatch]:
        """Find bio data matches using vector store."""
        
        if not self.vector_store:
            # Return mock matches for testing
            return self._generate_mock_matches(processed_query, max_results)
        
        try:
            # Build search query
            search_query = self._build_search_query(processed_query, max_results)
            
            # Perform vector search
            search_results = self.vector_store.search_similar_bio_data(
                query_text=search_query.query_text,
                target_bio_type="ppl_biodata",
                k=max_results * 2,  # Get more results for filtering
                similarity_threshold=0.5  # Lower threshold, we'll filter later
            )
            
            # Convert results to BioMatch objects
            bio_matches = []
            for result in search_results.get("results", []):
                bio_match = self._convert_to_bio_match(result, processed_query)
                if bio_match:
                    bio_matches.append(bio_match)
            
            # Apply additional filtering
            filtered_matches = self._apply_filters(bio_matches, processed_query)
            
            # Re-rank matches using LLM reasoning
            ranked_matches = await self._rerank_matches(filtered_matches, processed_query)
            
            # Return top matches
            return ranked_matches[:max_results]
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return self._generate_mock_matches(processed_query, max_results)
    
    def _build_search_query(self, processed_query: ProcessedQuery, max_results: int) -> SearchQuery:
        """Build search query for vector store."""
        # Create search text from requirements
        search_components = []
        
        # Add keywords
        search_components.extend(processed_query.search_keywords)
        
        # Add specific requirements as text
        requirements = processed_query.structured_requirements
        
        if "profession" in requirements:
            prof_list = requirements["profession"]
            if isinstance(prof_list, list):
                search_components.extend(prof_list)
            else:
                search_components.append(str(prof_list))
        
        if "interests" in requirements:
            interest_list = requirements["interests"]
            if isinstance(interest_list, list):
                search_components.extend(interest_list)
            else:
                search_components.append(str(interest_list))
        
        if "education" in requirements:
            edu_info = requirements["education"]
            if isinstance(edu_info, dict):
                for key, value in edu_info.items():
                    if value:
                        search_components.append(f"{key} {value}")
            else:
                search_components.append(str(edu_info))
        
        # Join components into search text
        search_text = " ".join(search_components)
        
        return SearchQuery(
            query_text=search_text,
            target_bio_type="ppl_biodata",
            k=max_results,
            similarity_threshold=0.5
        )
    
    def _convert_to_bio_match(self, search_result: Dict[str, Any], processed_query: ProcessedQuery) -> Optional[BioMatch]:
        """Convert search result to BioMatch object."""
        try:
            # Extract match reasons
            match_reasons = self._generate_match_reasons(search_result, processed_query)
            
            return BioMatch(
                bio_data_id=search_result["bio_data_id"],
                similarity_score=search_result["similarity_score"],
                personal_info=search_result.get("personal_info", {}),
                education=search_result.get("education"),
                professional=search_result.get("professional"),
                interests=search_result.get("interests"),
                lifestyle=search_result.get("lifestyle"),
                relationship=search_result.get("relationship"),
                source_file=search_result.get("source_file", ""),
                match_reasons=match_reasons
            )
        except Exception as e:
            self.logger.error(f"Failed to convert search result: {e}")
            return None
    
    def _generate_match_reasons(
        self, 
        search_result: Dict[str, Any], 
        processed_query: ProcessedQuery
    ) -> List[str]:
        """Generate reasons why this is a good match."""
        reasons = []
        requirements = processed_query.structured_requirements
        
        # Check professional alignment
        if "profession" in requirements and search_result.get("professional"):
            user_profs = requirements["profession"]
            candidate_job = search_result["professional"].get("current_job", "")
            candidate_skills = search_result["professional"].get("skills", [])
            
            if isinstance(user_profs, list):
                for prof in user_profs:
                    if prof.lower() in candidate_job.lower():
                        reasons.append(f"Professional match: {prof} aligns with {candidate_job}")
                    for skill in candidate_skills:
                        if prof.lower() in skill.lower():
                            reasons.append(f"Skill match: {prof} found in skills")
        
        # Check interest alignment
        if "interests" in requirements and search_result.get("interests"):
            user_interests = requirements["interests"]
            candidate_interests = search_result["interests"]
            
            if isinstance(user_interests, list):
                for interest in user_interests:
                    for category, items in candidate_interests.items():
                        if items and isinstance(items, list):
                            for item in items:
                                if interest.lower() in item.lower():
                                    reasons.append(f"Shared interest: {interest}")
        
        # Check education compatibility
        if "education" in requirements and search_result.get("education"):
            candidate_edu = search_result["education"]
            if candidate_edu.get("degree"):
                reasons.append(f"Education: {candidate_edu['degree']}")
        
        # Check age compatibility
        if "age_range" in requirements and search_result.get("personal_info", {}).get("age"):
            age_range = requirements["age_range"]
            candidate_age = search_result["personal_info"]["age"]
            
            if (isinstance(age_range, dict) and 
                age_range.get("min", 0) <= candidate_age <= age_range.get("max", 100)):
                reasons.append(f"Age compatibility: {candidate_age} years old")
        
        # Check location compatibility
        if "location" in requirements and search_result.get("personal_info", {}).get("location"):
            user_locations = requirements["location"]
            candidate_location = search_result["personal_info"]["location"]
            
            if isinstance(user_locations, list):
                for location in user_locations:
                    if location.lower() in candidate_location.lower():
                        reasons.append(f"Location match: {location}")
        
        # Default reason if no specific matches found
        if not reasons and search_result.get("similarity_score", 0) > 0.7:
            reasons.append("High semantic similarity in bio data")
        
        return reasons
    
    def _apply_filters(self, matches: List[BioMatch], processed_query: ProcessedQuery) -> List[BioMatch]:
        """Apply additional filters to matches."""
        filtered_matches = []
        requirements = processed_query.structured_requirements
        
        for match in matches:
            # Age filter
            if "age_range" in requirements and match.personal_info.get("age"):
                age_range = requirements["age_range"]
                candidate_age = match.personal_info["age"]
                
                if isinstance(age_range, dict):
                    min_age = age_range.get("min", 18)
                    max_age = age_range.get("max", 100)
                    
                    if not (min_age <= candidate_age <= max_age):
                        continue  # Skip this match
            
            # Add other filters as needed
            filtered_matches.append(match)
        
        return filtered_matches
    
    async def _rerank_matches(
        self, 
        matches: List[BioMatch], 
        processed_query: ProcessedQuery
    ) -> List[BioMatch]:
        """Use LLM to rerank matches based on query requirements."""
        if not matches:
            return matches
        
        # Prepare match data for LLM analysis
        match_summaries = []
        for i, match in enumerate(matches):
            summary = {
                "index": i,
                "bio_data_id": match.bio_data_id,
                "similarity_score": match.similarity_score,
                "personal_info": match.personal_info,
                "professional": match.professional,
                "interests": match.interests,
                "match_reasons": match.match_reasons
            }
            match_summaries.append(summary)
        
        prompt = f"""
You are analyzing bio data matches for reranking. Given the user's requirements and a list of potential matches, please rerank them by overall compatibility.

User Requirements:
{json.dumps(processed_query.structured_requirements, indent=2)}

Priority Factors:
{processed_query.priority_factors}

Potential Matches:
{json.dumps(match_summaries, indent=2)}

Please analyze each match and provide a reranked list with compatibility scores (0-1). Consider:
1. How well the match fulfills the user's requirements
2. Priority factors mentioned in the query
3. Overall compatibility beyond just similarity scores
4. Quality and completeness of the bio data

Respond with ONLY a JSON array of objects with "index" and "compatibility_score" fields, ordered by compatibility (highest first).

Example:
[
    {{"index": 2, "compatibility_score": 0.95}},
    {{"index": 0, "compatibility_score": 0.87}},
    {{"index": 1, "compatibility_score": 0.72}}
]
"""
        
        try:
            response = await self.generate_response(prompt, max_tokens=1000, temperature=0.3)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                rankings = json.loads(json_match.group(0))
                
                # Reorder matches based on LLM ranking
                reranked_matches = []
                for ranking in rankings:
                    index = ranking["index"]
                    compatibility_score = ranking["compatibility_score"]
                    
                    if 0 <= index < len(matches):
                        match = matches[index]
                        # Update similarity score with compatibility score
                        match.similarity_score = max(match.similarity_score, compatibility_score)
                        reranked_matches.append(match)
                
                return reranked_matches
            
        except Exception as e:
            self.logger.error(f"LLM reranking failed: {e}")
        
        # Fallback: sort by original similarity score
        return sorted(matches, key=lambda x: x.similarity_score, reverse=True)
    
    def _generate_mock_matches(self, processed_query: ProcessedQuery, max_results: int) -> List[BioMatch]:
        """Generate mock matches for testing."""
        mock_matches = []
        
        for i in range(min(max_results, 5)):
            mock_match = BioMatch(
                bio_data_id=f"mock_bio_{i+1}",
                similarity_score=0.8 - (i * 0.1),
                personal_info={
                    "name": f"Mock Person {i+1}",
                    "age": 28 + i,
                    "gender": "unknown",
                    "location": "Mock City"
                },
                professional={
                    "current_job": f"Mock Job {i+1}",
                    "company": f"Mock Company {i+1}",
                    "skills": ["skill1", "skill2"]
                },
                interests={
                    "hobbies": ["reading", "traveling"],
                    "sports": ["hiking", "swimming"]
                },
                source_file=f"mock_file_{i+1}.pdf",
                match_reasons=[
                    f"Mock reason {i+1}",
                    "High compatibility in interests"
                ]
            )
            mock_matches.append(mock_match)
        
        return mock_matches
    
    def _calculate_matching_confidence(self, matches: List[BioMatch]) -> float:
        """Calculate confidence score for the matching results."""
        if not matches:
            return 0.0
        
        # Base confidence on number and quality of matches
        score = 0.0
        
        # Score for having matches
        if matches:
            score += 0.3
        
        # Score for match quality
        avg_similarity = sum(match.similarity_score for match in matches) / len(matches)
        score += avg_similarity * 0.5
        
        # Score for match diversity (different similarity scores indicate good ranking)
        if len(matches) > 1:
            similarity_scores = [match.similarity_score for match in matches]
            score_variance = sum((s - avg_similarity) ** 2 for s in similarity_scores) / len(similarity_scores)
            if score_variance > 0.01:  # Some variance is good
                score += 0.2
        
        return min(1.0, score)
