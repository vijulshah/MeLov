"""
Compatibility Scorer Agent - Calculates compatibility scores between users and matches.
"""
import json
import math
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..models.agentic_models import (
    AgentTask, AgentResponse, AgentRole, BioData, SocialAnalysis, 
    CompatibilityScore, CompatibilityFactor
)
from .base_agent import BaseAgent
from ..prompts.prompt_manager import prompt_manager


class CompatibilityScorerAgent(BaseAgent):
    """Agent responsible for calculating comprehensive compatibility scores."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = AgentRole.COMPATIBILITY_SCORER
        
        # Scoring weights for different factors
        self.scoring_weights = {
            "basic_demographics": 0.15,  # Age, location, basic info
            "professional_alignment": 0.25,  # Career, education, goals
            "interest_compatibility": 0.20,  # Hobbies, lifestyle
            "personality_match": 0.25,  # Personality traits
            "values_alignment": 0.10,  # Core values and priorities
            "social_compatibility": 0.05   # Social media behavior
        }
        
        # Age compatibility curve parameters
        self.age_compatibility_params = {
            "optimal_range": 5,  # ±5 years is optimal
            "acceptable_range": 10,  # ±10 years is acceptable
            "penalty_factor": 0.1   # Penalty per year beyond acceptable
        }
        
        # Professional compatibility mappings
        self.professional_compatibility = {
            "same_field": 0.9,
            "related_field": 0.7,
            "complementary_field": 0.8,
            "different_field": 0.5
        }
        
        # Personality trait compatibility matrix
        self.personality_compatibility = {
            ("outgoing", "outgoing"): 0.8,
            ("outgoing", "introverted"): 0.6,
            ("adventurous", "adventurous"): 0.9,
            ("adventurous", "cautious"): 0.4,
            ("creative", "creative"): 0.8,
            ("creative", "analytical"): 0.7,
            ("ambitious", "ambitious"): 0.8,
            ("ambitious", "laid-back"): 0.5,
        }
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for compatibility scoring."""
        return prompt_manager.get_agent_prompt("compatibility_scorer")
    
    async def process_task(self, task: AgentTask) -> AgentResponse:
        """Process compatibility scoring task."""
        try:
            # Extract input data
            user_bio_data = task.input_data.get("user_bio_data")
            user_analysis = task.input_data.get("user_analysis")
            match_analyses = task.input_data.get("match_analyses", [])
            
            if not user_bio_data or not match_analyses:
                raise ValueError("User bio data and match analyses required")
            
            # Calculate compatibility scores for each match
            compatibility_scores = []
            for match_analysis in match_analyses:
                score = await self._calculate_compatibility_score(
                    user_bio_data, user_analysis, match_analysis
                )
                compatibility_scores.append(score.dict())
            
            # Sort by overall score
            compatibility_scores.sort(key=lambda x: x["overall_score"], reverse=True)
            
            # Calculate distribution statistics
            scores = [cs["overall_score"] for cs in compatibility_scores]
            stats = self._calculate_score_statistics(scores)
            
            return AgentResponse(
                agent_name=self.name,
                agent_role=self.role,
                task_id=task.task_id,
                success=True,
                response_data={
                    "compatibility_scores": compatibility_scores,
                    "total_matches": len(compatibility_scores),
                    "score_statistics": stats
                },
                confidence_score=self._calculate_scoring_confidence(compatibility_scores)
            )
            
        except Exception as e:
            self.logger.error(f"Compatibility scoring failed: {e}")
            raise
    
    async def _calculate_compatibility_score(
        self, 
        user_bio_data: Dict[str, Any], 
        user_analysis: Optional[Dict[str, Any]], 
        match_analysis: Dict[str, Any]
    ) -> CompatibilityScore:
        """Calculate comprehensive compatibility score for a single match."""
        
        match_bio_data_id = match_analysis["bio_data_id"]
        
        # Get match bio data (in production, would fetch from database)
        match_bio_data = await self._get_match_bio_data(match_bio_data_id)
        
        # Calculate individual factor scores
        factors = {}
        
        # 1. Basic Demographics Compatibility
        factors["basic_demographics"] = await self._score_basic_demographics(
            user_bio_data, match_bio_data
        )
        
        # 2. Professional Alignment
        factors["professional_alignment"] = await self._score_professional_alignment(
            user_bio_data, user_analysis, match_bio_data, match_analysis
        )
        
        # 3. Interest Compatibility
        factors["interest_compatibility"] = await self._score_interest_compatibility(
            user_analysis, match_analysis
        )
        
        # 4. Personality Match
        factors["personality_match"] = await self._score_personality_match(
            user_analysis, match_analysis
        )
        
        # 5. Values Alignment
        factors["values_alignment"] = await self._score_values_alignment(
            user_analysis, match_analysis
        )
        
        # 6. Social Compatibility
        factors["social_compatibility"] = await self._score_social_compatibility(
            user_analysis, match_analysis
        )
        
        # Calculate weighted overall score
        overall_score = sum(
            score * self.scoring_weights[factor] 
            for factor, score in factors.items()
        )
        
        # Generate compatibility explanation
        explanation = await self._generate_compatibility_explanation(
            user_bio_data, match_bio_data, factors, overall_score
        )
        
        # Identify key strengths and challenges
        strengths, challenges = self._identify_strengths_and_challenges(factors)
        
        return CompatibilityScore(
            bio_data_id=match_bio_data_id,
            overall_score=round(overall_score, 3),
            factor_scores={
                factor: CompatibilityFactor(
                    score=round(score, 3),
                    weight=self.scoring_weights[factor],
                    explanation=f"Compatibility in {factor.replace('_', ' ')}"
                )
                for factor, score in factors.items()
            },
            explanation=explanation,
            key_strengths=strengths,
            potential_challenges=challenges,
            recommendation=self._generate_recommendation(overall_score),
            confidence_level=self._calculate_match_confidence(factors)
        )
    
    async def _get_match_bio_data(self, bio_data_id: str) -> Dict[str, Any]:
        """Get bio data for a match (mock implementation)."""
        # In production, this would fetch from the vector store or database
        # For now, return mock data
        return {
            "id": bio_data_id,
            "name": "Sarah Chen",
            "age": 28,
            "location": "San Francisco, CA",
            "occupation": "UX Designer",
            "education": "Stanford University - Master's in HCI",
            "interests": ["Design", "Travel", "Photography", "Yoga"],
            "bio": "Passionate UX designer who loves creating meaningful digital experiences. Enjoy traveling, photography, and staying active through yoga and hiking."
        }
    
    async def _score_basic_demographics(
        self, 
        user_bio: Dict[str, Any], 
        match_bio: Dict[str, Any]
    ) -> float:
        """Score basic demographic compatibility."""
        score = 0.0
        factors = 0
        
        # Age compatibility
        if user_bio.get("age") and match_bio.get("age"):
            age_score = self._calculate_age_compatibility(
                user_bio["age"], match_bio["age"]
            )
            score += age_score
            factors += 1
        
        # Location compatibility
        if user_bio.get("location") and match_bio.get("location"):
            location_score = self._calculate_location_compatibility(
                user_bio["location"], match_bio["location"]
            )
            score += location_score
            factors += 1
        
        # Education level compatibility
        if user_bio.get("education") and match_bio.get("education"):
            education_score = self._calculate_education_compatibility(
                user_bio["education"], match_bio["education"]
            )
            score += education_score
            factors += 1
        
        return score / factors if factors > 0 else 0.5
    
    def _calculate_age_compatibility(self, user_age: int, match_age: int) -> float:
        """Calculate age compatibility score."""
        age_diff = abs(user_age - match_age)
        
        if age_diff <= self.age_compatibility_params["optimal_range"]:
            return 1.0
        elif age_diff <= self.age_compatibility_params["acceptable_range"]:
            # Linear decrease in acceptable range
            acceptable_range = self.age_compatibility_params["acceptable_range"]
            optimal_range = self.age_compatibility_params["optimal_range"]
            penalty_range = acceptable_range - optimal_range
            return 1.0 - ((age_diff - optimal_range) / penalty_range) * 0.3
        else:
            # Exponential decay beyond acceptable range
            excess_years = age_diff - self.age_compatibility_params["acceptable_range"]
            penalty = self.age_compatibility_params["penalty_factor"]
            return max(0.2, 0.7 * math.exp(-penalty * excess_years))
    
    def _calculate_location_compatibility(self, user_loc: str, match_loc: str) -> float:
        """Calculate location compatibility score."""
        # Simple string matching for now
        # In production, would use geographic distance calculation
        
        user_loc_lower = user_loc.lower()
        match_loc_lower = match_loc.lower()
        
        if user_loc_lower == match_loc_lower:
            return 1.0
        
        # Check for same city
        user_parts = user_loc_lower.split(", ")
        match_parts = match_loc_lower.split(", ")
        
        if len(user_parts) >= 2 and len(match_parts) >= 2:
            if user_parts[0] == match_parts[0]:  # Same city
                return 0.9
            elif user_parts[-1] == match_parts[-1]:  # Same state/country
                return 0.6
        
        return 0.3  # Different locations
    
    def _calculate_education_compatibility(self, user_edu: str, match_edu: str) -> float:
        """Calculate education compatibility score."""
        # Simple keyword-based matching
        # In production, would have more sophisticated education level parsing
        
        education_levels = {
            "phd": 5, "doctorate": 5, "ph.d": 5,
            "master": 4, "masters": 4, "mba": 4, "ms": 4, "ma": 4,
            "bachelor": 3, "bachelors": 3, "bs": 3, "ba": 3, "undergraduate": 3,
            "associate": 2, "associates": 2,
            "high school": 1, "diploma": 1
        }
        
        user_level = 3  # Default bachelor's
        match_level = 3
        
        user_edu_lower = user_edu.lower()
        match_edu_lower = match_edu.lower()
        
        for level_name, level_value in education_levels.items():
            if level_name in user_edu_lower:
                user_level = level_value
                break
        
        for level_name, level_value in education_levels.items():
            if level_name in match_edu_lower:
                match_level = level_value
                break
        
        # Calculate compatibility based on education level difference
        level_diff = abs(user_level - match_level)
        
        if level_diff == 0:
            return 1.0
        elif level_diff == 1:
            return 0.8
        elif level_diff == 2:
            return 0.6
        else:
            return 0.4
    
    async def _score_professional_alignment(
        self, 
        user_bio: Dict[str, Any], 
        user_analysis: Optional[Dict[str, Any]], 
        match_bio: Dict[str, Any], 
        match_analysis: Dict[str, Any]
    ) -> float:
        """Score professional and career alignment."""
        score = 0.0
        factors = 0
        
        # Industry/field alignment
        user_occupation = user_bio.get("occupation", "")
        match_occupation = match_bio.get("occupation", "")
        
        if user_occupation and match_occupation:
            field_score = await self._calculate_field_compatibility(
                user_occupation, match_occupation
            )
            score += field_score
            factors += 1
        
        # Career level alignment
        if user_analysis and match_analysis:
            user_prof = user_analysis.get("professional_insights", {})
            match_prof = match_analysis.get("professional_insights", {})
            
            user_level = user_prof.get("career_level", "unknown")
            match_level = match_prof.get("career_level", "unknown")
            
            if user_level != "unknown" and match_level != "unknown":
                level_score = self._calculate_career_level_compatibility(
                    user_level, match_level
                )
                score += level_score
                factors += 1
            
            # Professional interests alignment
            user_interests = user_prof.get("professional_interests", [])
            match_interests = match_prof.get("professional_interests", [])
            
            if user_interests and match_interests:
                interest_score = self._calculate_list_overlap(
                    user_interests, match_interests
                )
                score += interest_score
                factors += 1
        
        return score / factors if factors > 0 else 0.5
    
    async def _calculate_field_compatibility(self, user_field: str, match_field: str) -> float:
        """Calculate professional field compatibility."""
        # Use LLM to determine field relationship
        prompt = prompt_manager.get_llm_prompt(
            "field_compatibility",
            user_field=user_field,
            match_field=match_field
        )
        
        try:
            response = await self.generate_response(prompt, max_tokens=200, temperature=0.3)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group(0))
                return result.get("compatibility_score", 0.5)
        
        except Exception as e:
            self.logger.error(f"Field compatibility analysis failed: {e}")
        
        # Fallback to simple string matching
        if user_field.lower() == match_field.lower():
            return 0.9
        elif any(word in match_field.lower() for word in user_field.lower().split()):
            return 0.7
        else:
            return 0.5
    
    def _calculate_career_level_compatibility(self, user_level: str, match_level: str) -> float:
        """Calculate career level compatibility."""
        level_hierarchy = {"entry": 1, "mid": 2, "senior": 3, "executive": 4}
        
        user_rank = level_hierarchy.get(user_level, 2)
        match_rank = level_hierarchy.get(match_level, 2)
        
        level_diff = abs(user_rank - match_rank)
        
        if level_diff == 0:
            return 1.0
        elif level_diff == 1:
            return 0.8
        elif level_diff == 2:
            return 0.6
        else:
            return 0.4
    
    async def _score_interest_compatibility(
        self, 
        user_analysis: Optional[Dict[str, Any]], 
        match_analysis: Dict[str, Any]
    ) -> float:
        """Score interest and lifestyle compatibility."""
        if not user_analysis:
            return 0.5
        
        user_interests = user_analysis.get("interest_insights", {})
        match_interests = match_analysis.get("interest_insights", {})
        
        score = 0.0
        factors = 0
        
        # Primary interests overlap
        user_primary = user_interests.get("primary_interests", [])
        match_primary = match_interests.get("primary_interests", [])
        
        if user_primary and match_primary:
            overlap_score = self._calculate_list_overlap(user_primary, match_primary)
            score += overlap_score
            factors += 1
        
        # Lifestyle indicators compatibility
        user_lifestyle = user_interests.get("lifestyle_indicators", [])
        match_lifestyle = match_interests.get("lifestyle_indicators", [])
        
        if user_lifestyle and match_lifestyle:
            lifestyle_score = self._calculate_list_overlap(user_lifestyle, match_lifestyle)
            score += lifestyle_score
            factors += 1
        
        # Activity level compatibility
        user_activity = user_analysis.get("activity_level", "unknown")
        match_activity = match_analysis.get("activity_level", "unknown")
        
        if user_activity != "unknown" and match_activity != "unknown":
            activity_score = self._calculate_activity_compatibility(
                user_activity, match_activity
            )
            score += activity_score
            factors += 1
        
        return score / factors if factors > 0 else 0.5
    
    def _calculate_list_overlap(self, list1: List[str], list2: List[str]) -> float:
        """Calculate overlap score between two lists."""
        if not list1 or not list2:
            return 0.0
        
        set1 = set(item.lower() for item in list1)
        set2 = set(item.lower() for item in list2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_activity_compatibility(self, user_activity: str, match_activity: str) -> float:
        """Calculate social activity level compatibility."""
        activity_scores = {"none": 0, "low": 1, "medium": 2, "high": 3}
        
        user_score = activity_scores.get(user_activity, 1)
        match_score = activity_scores.get(match_activity, 1)
        
        difference = abs(user_score - match_score)
        
        if difference == 0:
            return 1.0
        elif difference == 1:
            return 0.8
        elif difference == 2:
            return 0.5
        else:
            return 0.3
    
    async def _score_personality_match(
        self, 
        user_analysis: Optional[Dict[str, Any]], 
        match_analysis: Dict[str, Any]
    ) -> float:
        """Score personality trait compatibility."""
        if not user_analysis:
            return 0.5
        
        user_traits = user_analysis.get("personality_traits", [])
        match_traits = match_analysis.get("personality_traits", [])
        
        if not user_traits or not match_traits:
            return 0.5
        
        # Calculate compatibility using trait pairs
        total_compatibility = 0.0
        comparisons = 0
        
        for user_trait in user_traits:
            for match_trait in match_traits:
                pair_key = tuple(sorted([user_trait.lower(), match_trait.lower()]))
                
                if pair_key in self.personality_compatibility:
                    total_compatibility += self.personality_compatibility[pair_key]
                    comparisons += 1
                elif user_trait.lower() == match_trait.lower():
                    total_compatibility += 0.8  # Same trait
                    comparisons += 1
        
        if comparisons > 0:
            return total_compatibility / comparisons
        
        # Fallback: simple overlap calculation
        return self._calculate_list_overlap(user_traits, match_traits)
    
    async def _score_values_alignment(
        self, 
        user_analysis: Optional[Dict[str, Any]], 
        match_analysis: Dict[str, Any]
    ) -> float:
        """Score core values and priorities alignment."""
        if not user_analysis:
            return 0.5
        
        user_interests = user_analysis.get("interest_insights", {})
        match_interests = match_analysis.get("interest_insights", {})
        
        user_values = user_interests.get("values", [])
        match_values = match_interests.get("values", [])
        
        if user_values and match_values:
            return self._calculate_list_overlap(user_values, match_values)
        
        # Fallback: infer values from other data
        return 0.5
    
    async def _score_social_compatibility(
        self, 
        user_analysis: Optional[Dict[str, Any]], 
        match_analysis: Dict[str, Any]
    ) -> float:
        """Score social media behavior compatibility."""
        if not user_analysis:
            return 0.5
        
        user_interests = user_analysis.get("interest_insights", {})
        match_interests = match_analysis.get("interest_insights", {})
        
        user_social_style = user_interests.get("social_style", "")
        match_social_style = match_interests.get("social_style", "")
        
        if user_social_style and match_social_style:
            # Simple text similarity
            if user_social_style.lower() == match_social_style.lower():
                return 1.0
            elif any(word in match_social_style.lower() 
                    for word in user_social_style.lower().split()):
                return 0.7
            else:
                return 0.4
        
        return 0.5
    
    async def _generate_compatibility_explanation(
        self, 
        user_bio: Dict[str, Any], 
        match_bio: Dict[str, Any], 
        factors: Dict[str, float], 
        overall_score: float
    ) -> str:
        """Generate detailed compatibility explanation."""
        
        # Find the strongest and weakest factors
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_factors[0]
        weakest = sorted_factors[-1]
        
        user_name = user_bio.get("name", "You")
        match_name = match_bio.get("name", "This person")
        
        prompt = f"""
Generate a personalized compatibility explanation for a dating match.

User: {user_name}
Match: {match_name}

Overall Compatibility Score: {overall_score:.2f}/1.0

Factor Scores:
{json.dumps(factors, indent=2)}

Strongest Factor: {strongest[0]} ({strongest[1]:.2f})
Weakest Factor: {weakest[0]} ({weakest[1]:.2f})

Write a warm, insightful explanation (2-3 sentences) that:
1. Highlights the strongest compatibility areas
2. Acknowledges any challenges
3. Gives an overall assessment
4. Uses encouraging, positive tone

Focus on what makes this match promising while being honest about areas to explore.
"""
        
        try:
            explanation = await self.generate_response(prompt, max_tokens=300, temperature=0.7)
            return explanation.strip()
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            return f"This match shows {overall_score:.0%} compatibility based on your profiles and interests."
    
    def _identify_strengths_and_challenges(
        self, 
        factors: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """Identify key strengths and potential challenges."""
        strengths = []
        challenges = []
        
        for factor, score in factors.items():
            factor_name = factor.replace("_", " ").title()
            
            if score >= 0.8:
                strengths.append(f"Strong {factor_name.lower()} alignment")
            elif score >= 0.6:
                strengths.append(f"Good {factor_name.lower()} compatibility")
            elif score < 0.4:
                challenges.append(f"Limited {factor_name.lower()} overlap")
        
        return strengths, challenges
    
    def _generate_recommendation(self, overall_score: float) -> str:
        """Generate match recommendation based on score."""
        if overall_score >= 0.8:
            return "Highly Recommended - Excellent compatibility potential"
        elif overall_score >= 0.7:
            return "Recommended - Strong compatibility with good potential"
        elif overall_score >= 0.6:
            return "Worth Exploring - Moderate compatibility with growth potential"
        elif overall_score >= 0.5:
            return "Consider Carefully - Some compatibility challenges to navigate"
        else:
            return "Limited Compatibility - Significant differences to consider"
    
    def _calculate_match_confidence(self, factors: Dict[str, float]) -> float:
        """Calculate confidence level for the compatibility assessment."""
        # Higher confidence when we have more data and consistent scores
        non_default_factors = sum(1 for score in factors.values() if score != 0.5)
        data_confidence = non_default_factors / len(factors)
        
        # Lower confidence when scores are very close to 0.5 (uncertain)
        score_variance = sum((score - 0.5) ** 2 for score in factors.values()) / len(factors)
        variance_confidence = min(1.0, score_variance * 4)  # Scale variance
        
        return (data_confidence + variance_confidence) / 2
    
    def _calculate_score_statistics(self, scores: List[float]) -> Dict[str, Any]:
        """Calculate statistics for the score distribution."""
        if not scores:
            return {}
        
        return {
            "mean_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "high_compatibility_count": sum(1 for s in scores if s >= 0.7),
            "medium_compatibility_count": sum(1 for s in scores if 0.5 <= s < 0.7),
            "low_compatibility_count": sum(1 for s in scores if s < 0.5)
        }
    
    def _calculate_scoring_confidence(self, compatibility_scores: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in the scoring results."""
        if not compatibility_scores:
            return 0.0
        
        confidence_levels = [
            score.get("confidence_level", 0.5) 
            for score in compatibility_scores
        ]
        
        return sum(confidence_levels) / len(confidence_levels)
