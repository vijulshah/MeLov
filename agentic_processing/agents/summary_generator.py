"""
Summary Generator Agent - Creates comprehensive match summaries and recommendations.
"""
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..models.agentic_models import (
    AgentTask, AgentResponse, AgentRole, MatchSummary, MatchRecommendation,
    CompatibilityScore
)
from .base_agent import BaseAgent
from ..prompts.prompt_manager import prompt_manager


class SummaryGeneratorAgent(BaseAgent):
    """Agent responsible for generating comprehensive match summaries and recommendations."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = AgentRole.SUMMARY_GENERATOR
        
        # Template structures for different summary types
        self.summary_templates = {
            "detailed": {
                "sections": [
                    "compatibility_overview",
                    "key_strengths",
                    "potential_challenges", 
                    "conversation_starters",
                    "relationship_potential",
                    "next_steps"
                ]
            },
            "brief": {
                "sections": [
                    "compatibility_overview",
                    "key_highlights",
                    "recommendation"
                ]
            }
        }
        
        # Conversation starter categories
        self.conversation_categories = {
            "professional": [
                "Ask about their work projects",
                "Discuss industry trends",
                "Share career experiences"
            ],
            "interests": [
                "Explore shared hobbies",
                "Discuss travel experiences",
                "Share favorite activities"
            ],
            "lifestyle": [
                "Talk about weekend routines",
                "Discuss fitness activities",
                "Share food preferences"
            ],
            "values": [
                "Discuss life goals",
                "Share personal values",
                "Talk about future plans"
            ]
        }
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for summary generation."""
        return prompt_manager.get_agent_prompt("summary_generator")
    
    async def process_task(self, task: AgentTask) -> AgentResponse:
        """Process summary generation task."""
        try:
            # Extract input data
            user_bio_data = task.input_data.get("user_bio_data")
            compatibility_scores = task.input_data.get("compatibility_scores", [])
            top_k = task.input_data.get("top_k", 10)
            summary_type = task.input_data.get("summary_type", "detailed")
            
            if not user_bio_data or not compatibility_scores:
                raise ValueError("User bio data and compatibility scores required")
            
            # Take top K matches
            top_matches = compatibility_scores[:top_k]
            
            # Generate individual match summaries
            match_summaries = []
            for i, score_data in enumerate(top_matches):
                summary = await self._generate_match_summary(
                    user_bio_data, score_data, rank=i+1, summary_type=summary_type
                )
                match_summaries.append(summary.dict())
            
            # Generate overall recommendations
            overall_recommendations = await self._generate_overall_recommendations(
                user_bio_data, top_matches
            )
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(top_matches)
            
            return AgentResponse(
                agent_name=self.name,
                agent_role=self.role,
                task_id=task.task_id,
                success=True,
                response_data={
                    "match_summaries": match_summaries,
                    "overall_recommendations": overall_recommendations,
                    "summary_statistics": summary_stats,
                    "generated_at": datetime.now().isoformat(),
                    "summary_type": summary_type
                },
                confidence_score=self._calculate_summary_confidence(match_summaries)
            )
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            raise
    
    async def _generate_match_summary(
        self, 
        user_bio_data: Dict[str, Any], 
        score_data: Dict[str, Any], 
        rank: int, 
        summary_type: str = "detailed"
    ) -> MatchSummary:
        """Generate comprehensive summary for a single match."""
        
        bio_data_id = score_data["bio_data_id"]
        overall_score = score_data["overall_score"]
        factor_scores = score_data.get("factor_scores", {})
        explanation = score_data.get("explanation", "")
        key_strengths = score_data.get("key_strengths", [])
        potential_challenges = score_data.get("potential_challenges", [])
        
        # Get match bio data (mock for now)
        match_bio_data = await self._get_match_bio_data(bio_data_id)
        
        # Generate conversation starters
        conversation_starters = await self._generate_conversation_starters(
            user_bio_data, match_bio_data, factor_scores
        )
        
        # Generate relationship potential assessment
        relationship_potential = await self._assess_relationship_potential(
            overall_score, factor_scores, key_strengths
        )
        
        # Generate next steps recommendation
        next_steps = await self._generate_next_steps(
            overall_score, key_strengths, potential_challenges
        )
        
        # Create detailed summary text
        if summary_type == "detailed":
            summary_text = await self._create_detailed_summary(
                user_bio_data, match_bio_data, score_data, conversation_starters
            )
        else:
            summary_text = await self._create_brief_summary(
                user_bio_data, match_bio_data, score_data
            )
        
        return MatchSummary(
            bio_data_id=bio_data_id,
            rank=rank,
            overall_score=overall_score,
            summary_text=summary_text,
            key_strengths=key_strengths,
            potential_challenges=potential_challenges,
            conversation_starters=conversation_starters,
            relationship_potential=relationship_potential,
            next_steps=next_steps,
            confidence_level=score_data.get("confidence_level", 0.5)
        )
    
    async def _get_match_bio_data(self, bio_data_id: str) -> Dict[str, Any]:
        """Get bio data for a match (mock implementation)."""
        # Mock data - in production would fetch from database
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
    
    async def _generate_conversation_starters(
        self, 
        user_bio_data: Dict[str, Any], 
        match_bio_data: Dict[str, Any], 
        factor_scores: Dict[str, Any]
    ) -> List[str]:
        """Generate personalized conversation starters."""
        
        starters = []
        
        # Get strongest compatibility factors
        sorted_factors = sorted(
            factor_scores.items(), 
            key=lambda x: x[1].get("score", 0) if isinstance(x[1], dict) else x[1], 
            reverse=True
        )
        
        top_factors = [factor for factor, _ in sorted_factors[:3]]
        
        # Generate starters based on top compatibility areas
        for factor in top_factors:
            if factor == "professional_alignment":
                starters.extend(await self._generate_professional_starters(
                    user_bio_data, match_bio_data
                ))
            elif factor == "interest_compatibility":
                starters.extend(await self._generate_interest_starters(
                    user_bio_data, match_bio_data
                ))
            elif factor == "personality_match":
                starters.extend(await self._generate_personality_starters(
                    user_bio_data, match_bio_data
                ))
        
        # Use LLM to generate additional personalized starters
        llm_starters = await self._llm_generate_conversation_starters(
            user_bio_data, match_bio_data, top_factors
        )
        starters.extend(llm_starters)
        
        # Remove duplicates and limit to 5-7 starters
        unique_starters = list(dict.fromkeys(starters))
        return unique_starters[:6]
    
    async def _generate_professional_starters(
        self, 
        user_bio_data: Dict[str, Any], 
        match_bio_data: Dict[str, Any]
    ) -> List[str]:
        """Generate professional conversation starters."""
        starters = []
        
        user_occupation = user_bio_data.get("occupation", "")
        match_occupation = match_bio_data.get("occupation", "")
        
        if user_occupation and match_occupation:
            starters.append(f"I'd love to hear more about your experience in {match_occupation}")
            starters.append("What's the most exciting project you're working on right now?")
            starters.append("How did you get started in your field?")
        
        return starters
    
    async def _generate_interest_starters(
        self, 
        user_bio_data: Dict[str, Any], 
        match_bio_data: Dict[str, Any]
    ) -> List[str]:
        """Generate interest-based conversation starters."""
        starters = []
        
        user_interests = user_bio_data.get("interests", [])
        match_interests = match_bio_data.get("interests", [])
        
        # Find shared interests
        shared_interests = set(i.lower() for i in user_interests) & set(i.lower() for i in match_interests)
        
        for interest in list(shared_interests)[:2]:
            starters.append(f"I noticed we both enjoy {interest} - what got you into it?")
            starters.append(f"Do you have any favorite {interest} spots or recommendations?")
        
        return starters
    
    async def _generate_personality_starters(
        self, 
        user_bio_data: Dict[str, Any], 
        match_bio_data: Dict[str, Any]
    ) -> List[str]:
        """Generate personality-based conversation starters."""
        return [
            "What's something you're passionate about that others might find surprising?",
            "If you could have any superpower, what would it be and why?",
            "What's the best advice you've ever received?"
        ]
    
    async def _llm_generate_conversation_starters(
        self, 
        user_bio_data: Dict[str, Any], 
        match_bio_data: Dict[str, Any], 
        top_factors: List[str]
    ) -> List[str]:
        """Use LLM to generate personalized conversation starters."""
        
        prompt = prompt_manager.get_llm_prompt(
            "conversation_starters",
            user_name=user_bio_data.get('name', 'User'),
            user_occupation=user_bio_data.get('occupation', 'Unknown'),
            user_interests=', '.join(user_bio_data.get('interests', [])),
            user_bio=user_bio_data.get('bio', ''),
            match_name=match_bio_data.get('name', 'Match'),
            match_occupation=match_bio_data.get('occupation', 'Unknown'),
            match_interests=', '.join(match_bio_data.get('interests', [])),
            match_bio=match_bio_data.get('bio', ''),
            top_factors=', '.join(top_factors)
        )
        
        try:
            response = await self.generate_response(prompt, max_tokens=300, temperature=0.7)
            
            # Extract JSON array from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
                
        except Exception as e:
            self.logger.error(f"LLM conversation starter generation failed: {e}")
        
        return []
    
    async def _assess_relationship_potential(
        self, 
        overall_score: float, 
        factor_scores: Dict[str, Any], 
        key_strengths: List[str]
    ) -> str:
        """Assess long-term relationship potential."""
        
        if overall_score >= 0.8:
            return "High potential for a meaningful, long-term connection with strong compatibility across multiple dimensions."
        elif overall_score >= 0.7:
            return "Good potential for a solid relationship with strong foundations and room for growth together."
        elif overall_score >= 0.6:
            return "Moderate potential that could develop well with open communication and mutual understanding."
        elif overall_score >= 0.5:
            return "Some potential, but may require extra effort to navigate differences and build compatibility."
        else:
            return "Limited long-term potential due to significant compatibility challenges, though meaningful friendship is possible."
    
    async def _generate_next_steps(
        self, 
        overall_score: float, 
        key_strengths: List[str], 
        potential_challenges: List[str]
    ) -> List[str]:
        """Generate recommended next steps."""
        
        steps = []
        
        if overall_score >= 0.7:
            steps.append("Reach out with a thoughtful message referencing shared interests")
            steps.append("Suggest a low-pressure activity you both enjoy")
            steps.append("Be authentic and share your genuine interests")
        elif overall_score >= 0.5:
            steps.append("Start with friendly conversation to explore compatibility")
            steps.append("Focus on areas where you align well")
            steps.append("Be open about differences and see how they respond")
        else:
            steps.append("Consider starting as friends to see if compatibility develops")
            steps.append("Focus on getting to know them better before pursuing romantically")
        
        # Add personalized steps based on strengths
        if "professional" in " ".join(key_strengths).lower():
            steps.append("Discuss career goals and professional interests")
        
        if "interest" in " ".join(key_strengths).lower():
            steps.append("Plan activities around your shared hobbies")
        
        return steps[:4]  # Limit to 4 steps
    
    async def _create_detailed_summary(
        self, 
        user_bio_data: Dict[str, Any], 
        match_bio_data: Dict[str, Any], 
        score_data: Dict[str, Any], 
        conversation_starters: List[str]
    ) -> str:
        """Create detailed match summary."""
        
        user_name = user_bio_data.get("name", "You")
        match_name = match_bio_data.get("name", "This person")
        overall_score = score_data["overall_score"]
        explanation = score_data.get("explanation", "")
        
        prompt = prompt_manager.get_llm_prompt(
            "detailed_summary",
            user_name=user_name,
            match_name=match_name,
            compatibility_score=f"{overall_score:.0%}",
            match_age=match_bio_data.get('age'),
            match_occupation=match_bio_data.get('occupation'),
            match_location=match_bio_data.get('location'),
            match_interests=', '.join(match_bio_data.get('interests', [])),
            match_bio=match_bio_data.get('bio', ''),
            explanation=explanation,
            conversation_starters=', '.join(conversation_starters[:3])
        )
        
        try:
            summary = await self.generate_response(prompt, max_tokens=400, temperature=0.7)
            return summary.strip()
        except Exception as e:
            self.logger.error(f"Detailed summary generation failed: {e}")
            return f"{match_name} shows {overall_score:.0%} compatibility with you based on shared interests and values."
    
    async def _create_brief_summary(
        self, 
        user_bio_data: Dict[str, Any], 
        match_bio_data: Dict[str, Any], 
        score_data: Dict[str, Any]
    ) -> str:
        """Create brief match summary."""
        
        match_name = match_bio_data.get("name", "This person")
        overall_score = score_data["overall_score"]
        key_strengths = score_data.get("key_strengths", [])
        
        highlights = key_strengths[:2] if key_strengths else ["good compatibility"]
        
        return f"{match_name} shows {overall_score:.0%} compatibility with strong alignment in {' and '.join(highlights).lower()}. Worth exploring for a meaningful connection."
    
    async def _generate_overall_recommendations(
        self, 
        user_bio_data: Dict[str, Any], 
        top_matches: List[Dict[str, Any]]
    ) -> List[MatchRecommendation]:
        """Generate overall recommendations across all matches."""
        
        recommendations = []
        
        if not top_matches:
            recommendations.append(MatchRecommendation(
                type="no_matches",
                title="No Matches Found",
                description="Consider expanding your search criteria or updating your profile.",
                priority="high"
            ))
            return recommendations
        
        # Analyze match quality distribution
        high_quality_matches = [m for m in top_matches if m["overall_score"] >= 0.7]
        medium_quality_matches = [m for m in top_matches if 0.5 <= m["overall_score"] < 0.7]
        
        # Generate recommendations based on match quality
        if high_quality_matches:
            recommendations.append(MatchRecommendation(
                type="high_quality_matches",
                title=f"Focus on Your Top {len(high_quality_matches)} Matches",
                description=f"You have {len(high_quality_matches)} highly compatible matches. Prioritize reaching out to these connections first.",
                priority="high"
            ))
        
        if len(medium_quality_matches) > 3:
            recommendations.append(MatchRecommendation(
                type="explore_medium_matches",
                title="Explore Additional Matches",
                description=f"Consider exploring {len(medium_quality_matches)} additional matches that show good potential for meaningful connections.",
                priority="medium"
            ))
        
        # Profile improvement recommendations
        recommendations.extend(await self._generate_profile_recommendations(user_bio_data, top_matches))
        
        return recommendations
    
    async def _generate_profile_recommendations(
        self, 
        user_bio_data: Dict[str, Any], 
        top_matches: List[Dict[str, Any]]
    ) -> List[MatchRecommendation]:
        """Generate profile improvement recommendations."""
        
        recommendations = []
        
        # Analyze common weak factors across matches
        factor_scores = {}
        for match in top_matches:
            for factor, score_data in match.get("factor_scores", {}).items():
                score = score_data.get("score", 0) if isinstance(score_data, dict) else score_data
                if factor not in factor_scores:
                    factor_scores[factor] = []
                factor_scores[factor].append(score)
        
        # Find consistently weak areas
        weak_factors = []
        for factor, scores in factor_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < 0.5:
                weak_factors.append(factor)
        
        # Generate improvement recommendations
        if "interest_compatibility" in weak_factors:
            recommendations.append(MatchRecommendation(
                type="profile_improvement",
                title="Add More Interests to Your Profile",
                description="Consider adding more specific hobbies and interests to help find better matches.",
                priority="medium"
            ))
        
        if "professional_alignment" in weak_factors:
            recommendations.append(MatchRecommendation(
                type="profile_improvement", 
                title="Enhance Professional Information",
                description="Adding more details about your career and goals could improve professional compatibility.",
                priority="low"
            ))
        
        return recommendations
    
    def _calculate_summary_statistics(self, top_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for the matches."""
        
        if not top_matches:
            return {"total_matches": 0}
        
        scores = [match["overall_score"] for match in top_matches]
        
        return {
            "total_matches": len(top_matches),
            "average_compatibility": sum(scores) / len(scores),
            "highest_compatibility": max(scores),
            "high_quality_count": sum(1 for s in scores if s >= 0.7),
            "medium_quality_count": sum(1 for s in scores if 0.5 <= s < 0.7),
            "matches_to_focus_on": min(3, sum(1 for s in scores if s >= 0.6))
        }
    
    def _calculate_summary_confidence(self, match_summaries: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the summary quality."""
        
        if not match_summaries:
            return 0.0
        
        confidence_levels = [
            summary.get("confidence_level", 0.5) 
            for summary in match_summaries
        ]
        
        # Higher confidence when we have good data quality across matches
        avg_confidence = sum(confidence_levels) / len(confidence_levels)
        
        # Bonus for having multiple high-quality matches
        high_quality_bonus = min(0.2, len(match_summaries) * 0.05)
        
        return min(1.0, avg_confidence + high_quality_bonus)
