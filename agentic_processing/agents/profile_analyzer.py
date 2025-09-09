"""
Profile Analyzer Agent - Analyzes social media profiles and extracts relevant data.
"""
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from ..models.agentic_models import (
    AgentTask, AgentResponse, AgentRole, SocialProfile, LinkedInProfile, 
    SocialAnalysis, SocialPlatform
)
from .base_agent import BaseAgent
from ..prompts.prompt_manager import prompt_manager


class ProfileAnalyzerAgent(BaseAgent):
    """Agent responsible for analyzing social media profiles and extracting data."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = AgentRole.PROFILE_ANALYZER
        
        # Keywords for personality trait extraction
        self.personality_keywords = {
            "outgoing": ["outgoing", "social", "extrovert", "party", "networking", "meetups"],
            "adventurous": ["adventure", "travel", "explore", "hiking", "climbing", "extreme"],
            "creative": ["creative", "art", "design", "music", "writing", "photography"],
            "analytical": ["analytical", "data", "research", "science", "logic", "problem"],
            "ambitious": ["ambitious", "goals", "leadership", "entrepreneur", "startup", "growth"],
            "caring": ["caring", "volunteer", "charity", "helping", "community", "support"],
            "fitness-oriented": ["fitness", "gym", "workout", "running", "health", "sports"],
            "intellectual": ["books", "reading", "learning", "education", "philosophy", "debate"]
        }
        
        # Professional level indicators
        self.professional_levels = {
            "entry": ["intern", "junior", "associate", "entry", "trainee"],
            "mid": ["senior", "lead", "specialist", "coordinator", "analyst"],
            "senior": ["manager", "director", "head", "principal", "architect"],
            "executive": ["vp", "ceo", "cto", "founder", "president", "executive"]
        }
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for profile analysis."""
        return prompt_manager.get_agent_prompt("profile_analyzer")
    
    async def process_task(self, task: AgentTask) -> AgentResponse:
        """Process profile analysis task."""
        try:
            # Extract input data
            social_results = task.input_data.get("social_results", [])
            
            if not social_results:
                raise ValueError("No social results provided")
            
            # Analyze each person's social profiles
            analysis_results = []
            for social_result in social_results:
                bio_data_id = social_result["bio_data_id"]
                profiles = social_result.get("profiles", [])
                
                # Convert profile dicts to SocialProfile objects
                social_profiles = []
                for profile_dict in profiles:
                    social_profiles.append(SocialProfile(**profile_dict))
                
                # Perform analysis
                analysis = await self._analyze_social_profiles(bio_data_id, social_profiles)
                analysis_results.append(analysis.dict())
            
            return AgentResponse(
                agent_name=self.name,
                agent_role=self.role,
                task_id=task.task_id,
                success=True,
                response_data={
                    "analysis_results": analysis_results,
                    "total_analyzed": len(analysis_results)
                },
                confidence_score=self._calculate_analysis_confidence(analysis_results)
            )
            
        except Exception as e:
            self.logger.error(f"Profile analysis failed: {e}")
            raise
    
    async def _analyze_social_profiles(
        self, 
        bio_data_id: str, 
        profiles: List[SocialProfile]
    ) -> SocialAnalysis:
        """Analyze social profiles for a single person."""
        
        # Separate profiles by platform
        linkedin_profiles = [p for p in profiles if p.platform == SocialPlatform.LINKEDIN]
        instagram_profiles = [p for p in profiles if p.platform == SocialPlatform.INSTAGRAM]
        
        # Analyze LinkedIn profile (most detailed)
        linkedin_data = None
        if linkedin_profiles:
            linkedin_data = await self._analyze_linkedin_profile(linkedin_profiles[0])
        
        # Extract professional insights
        professional_insights = await self._extract_professional_insights(profiles, linkedin_data)
        
        # Extract interest insights
        interest_insights = await self._extract_interest_insights(profiles)
        
        # Extract personality traits
        personality_traits = await self._extract_personality_traits(profiles)
        
        # Assess activity level
        activity_level = self._assess_activity_level(profiles)
        
        # Calculate overall confidence
        confidence_score = self._calculate_profile_confidence(profiles, linkedin_data)
        
        return SocialAnalysis(
            bio_data_id=bio_data_id,
            profiles=profiles,
            linkedin_data=linkedin_data,
            professional_insights=professional_insights,
            interest_insights=interest_insights,
            personality_traits=personality_traits,
            activity_level=activity_level,
            confidence_score=confidence_score,
            data_freshness="moderate"  # Mock value
        )
    
    async def _analyze_linkedin_profile(self, profile: SocialProfile) -> LinkedInProfile:
        """Analyze LinkedIn profile and extract detailed data."""
        
        # In production, this would scrape or API call LinkedIn
        # For now, generate mock detailed LinkedIn data
        
        mock_experiences = [
            {
                "title": "Software Engineer",
                "company": "Tech Corp",
                "duration": "2022 - Present",
                "description": "Developing scalable web applications using React and Node.js"
            },
            {
                "title": "Junior Developer",
                "company": "Startup Inc",
                "duration": "2020 - 2022",
                "description": "Full-stack development with Python and JavaScript"
            }
        ]
        
        mock_education = [
            {
                "degree": "Bachelor of Science in Computer Science",
                "institution": "State University",
                "year": "2020",
                "gpa": "3.7"
            }
        ]
        
        mock_skills = [
            "Python", "JavaScript", "React", "Node.js", "AWS", 
            "Machine Learning", "Data Analysis", "Project Management"
        ]
        
        mock_posts = [
            {
                "date": "2024-09-01",
                "content": "Excited to share our latest project on sustainable tech solutions!",
                "engagement": {"likes": 45, "comments": 8}
            },
            {
                "date": "2024-08-15", 
                "content": "Great networking event at the AI conference. Looking forward to implementing new ideas!",
                "engagement": {"likes": 32, "comments": 5}
            }
        ]
        
        return LinkedInProfile(
            profile_url=profile.profile_url,
            headline=profile.bio,
            summary="Passionate software engineer with experience in full-stack development and emerging technologies.",
            current_position=mock_experiences[0] if mock_experiences else None,
            experience=mock_experiences,
            education=mock_education,
            skills=mock_skills,
            certifications=[],
            connections_count=profile.following_count,
            recent_posts=mock_posts,
            activity_score=0.7
        )
    
    async def _extract_professional_insights(
        self, 
        profiles: List[SocialProfile], 
        linkedin_data: Optional[LinkedInProfile]
    ) -> Dict[str, Any]:
        """Extract professional insights from profiles."""
        insights = {
            "career_level": "unknown",
            "industry_focus": [],
            "skills_mentioned": [],
            "leadership_indicators": [],
            "career_growth": "unknown",
            "professional_network_size": "unknown"
        }
        
        # Analyze LinkedIn data if available
        if linkedin_data:
            # Determine career level
            insights["career_level"] = self._determine_career_level(linkedin_data)
            
            # Extract skills
            insights["skills_mentioned"] = linkedin_data.skills[:10]  # Top 10 skills
            
            # Analyze career growth
            insights["career_growth"] = self._analyze_career_growth(linkedin_data.experience)
            
            # Network size assessment
            if linkedin_data.connections_count:
                if linkedin_data.connections_count > 500:
                    insights["professional_network_size"] = "large"
                elif linkedin_data.connections_count > 100:
                    insights["professional_network_size"] = "medium"
                else:
                    insights["professional_network_size"] = "small"
            
            # Leadership indicators
            insights["leadership_indicators"] = self._extract_leadership_indicators(linkedin_data)
        
        # Use LLM for deeper analysis
        llm_insights = await self._llm_professional_analysis(profiles)
        
        # Merge insights
        for key, value in llm_insights.items():
            if value and (key not in insights or not insights[key]):
                insights[key] = value
        
        return insights
    
    def _determine_career_level(self, linkedin_data: LinkedInProfile) -> str:
        """Determine career level from LinkedIn data."""
        if not linkedin_data.experience:
            return "unknown"
        
        current_position = linkedin_data.current_position
        if not current_position:
            return "entry"
        
        title = current_position.get("title", "").lower()
        
        for level, keywords in self.professional_levels.items():
            if any(keyword in title for keyword in keywords):
                return level
        
        return "mid"  # Default
    
    def _analyze_career_growth(self, experiences: List[Dict[str, Any]]) -> str:
        """Analyze career growth trajectory."""
        if len(experiences) < 2:
            return "insufficient_data"
        
        # Simple analysis based on titles
        levels = []
        for exp in experiences:
            title = exp.get("title", "").lower()
            for level, keywords in self.professional_levels.items():
                if any(keyword in title for keyword in keywords):
                    levels.append(level)
                    break
            else:
                levels.append("mid")  # Default
        
        # Map levels to numbers for comparison
        level_scores = {"entry": 1, "mid": 2, "senior": 3, "executive": 4}
        scores = [level_scores.get(level, 2) for level in levels]
        
        if len(scores) >= 2:
            if scores[0] > scores[-1]:  # Most recent is higher
                return "ascending"
            elif scores[0] < scores[-1]:
                return "lateral_or_descending"
            else:
                return "stable"
        
        return "unknown"
    
    def _extract_leadership_indicators(self, linkedin_data: LinkedInProfile) -> List[str]:
        """Extract leadership indicators from LinkedIn data."""
        indicators = []
        
        # Check titles for leadership keywords
        for exp in linkedin_data.experience:
            title = exp.get("title", "").lower()
            description = exp.get("description", "").lower()
            
            leadership_keywords = [
                "lead", "manage", "direct", "supervise", "coordinate",
                "mentor", "train", "team", "project", "initiative"
            ]
            
            for keyword in leadership_keywords:
                if keyword in title or keyword in description:
                    indicators.append(f"Leadership experience: {keyword}")
        
        # Check posts for leadership content
        if linkedin_data.recent_posts:
            for post in linkedin_data.recent_posts:
                content = post.get("content", "").lower()
                if any(word in content for word in ["team", "project", "led", "managed"]):
                    indicators.append("Leadership content in posts")
                    break
        
        return list(set(indicators))  # Remove duplicates
    
    async def _extract_interest_insights(self, profiles: List[SocialProfile]) -> Dict[str, Any]:
        """Extract interest and hobby insights from profiles."""
        insights = {
            "primary_interests": [],
            "lifestyle_indicators": [],
            "social_engagement": "unknown",
            "content_themes": [],
            "activity_patterns": {}
        }
        
        # Analyze profile bios for interests
        all_bios = []
        for profile in profiles:
            if profile.bio:
                all_bios.append(profile.bio)
        
        if all_bios:
            combined_bio = " ".join(all_bios).lower()
            
            # Extract interests using keywords
            found_interests = []
            interest_categories = {
                "fitness": ["gym", "fitness", "workout", "running", "cycling", "yoga"],
                "travel": ["travel", "explore", "adventure", "wanderlust", "vacation"],
                "food": ["food", "cooking", "chef", "restaurant", "wine", "coffee"],
                "arts": ["art", "music", "photography", "design", "creative", "painting"],
                "tech": ["tech", "coding", "programming", "innovation", "startup"],
                "outdoors": ["hiking", "camping", "nature", "outdoor", "climbing", "skiing"]
            }
            
            for category, keywords in interest_categories.items():
                if any(keyword in combined_bio for keyword in keywords):
                    found_interests.append(category)
            
            insights["primary_interests"] = found_interests
        
        # Use LLM for deeper interest analysis
        llm_insights = await self._llm_interest_analysis(profiles)
        
        # Merge insights
        for key, value in llm_insights.items():
            if value and (key not in insights or not insights[key]):
                insights[key] = value
        
        return insights
    
    async def _extract_personality_traits(self, profiles: List[SocialProfile]) -> List[str]:
        """Extract personality traits from social profiles."""
        traits = []
        
        # Analyze bios for personality indicators
        all_content = []
        for profile in profiles:
            if profile.bio:
                all_content.append(profile.bio.lower())
        
        combined_content = " ".join(all_content)
        
        # Check for personality trait keywords
        for trait, keywords in self.personality_keywords.items():
            if any(keyword in combined_content for keyword in keywords):
                traits.append(trait)
        
        # Use LLM for personality analysis
        if combined_content:
            llm_traits = await self._llm_personality_analysis(combined_content)
            traits.extend(llm_traits)
        
        return list(set(traits))  # Remove duplicates
    
    def _assess_activity_level(self, profiles: List[SocialProfile]) -> str:
        """Assess overall social media activity level."""
        total_followers = sum(p.follower_count or 0 for p in profiles)
        total_following = sum(p.following_count or 0 for p in profiles)
        
        # Simple heuristic based on follower counts and platform presence
        if len(profiles) >= 3 or total_followers > 1000:
            return "high"
        elif len(profiles) >= 2 or total_followers > 200:
            return "medium"
        elif profiles:
            return "low"
        else:
            return "none"
    
    async def _llm_professional_analysis(self, profiles: List[SocialProfile]) -> Dict[str, Any]:
        """Use LLM to analyze professional aspects of profiles."""
        if not profiles:
            return {}
        
        profile_data = []
        for profile in profiles:
            profile_data.append({
                "platform": profile.platform,
                "bio": profile.bio,
                "follower_count": profile.follower_count
            })
        
        prompt = f"""
Analyze these social media profiles for professional insights:

{json.dumps(profile_data, indent=2)}

Extract the following professional insights:
1. Industry focus (what industry/field do they work in?)
2. Professional interests (what professional topics do they engage with?)
3. Career ambitions (are they career-focused, entrepreneurial, etc.?)
4. Professional communication style
5. Network engagement level

Respond with ONLY a JSON object:
{{
    "industry_focus": ["technology", "healthcare"],
    "professional_interests": ["artificial intelligence", "product management"],
    "career_ambitions": "entrepreneurial",
    "communication_style": "professional and engaging",
    "network_engagement": "active"
}}
"""
        
        try:
            response = await self.generate_response(prompt, max_tokens=500, temperature=0.3)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
                
        except Exception as e:
            self.logger.error(f"LLM professional analysis failed: {e}")
        
        return {}
    
    async def _llm_interest_analysis(self, profiles: List[SocialProfile]) -> Dict[str, Any]:
        """Use LLM to analyze interests and lifestyle from profiles."""
        if not profiles:
            return {}
        
        bios = [p.bio for p in profiles if p.bio]
        if not bios:
            return {}
        
        prompt = f"""
Analyze these social media profile bios for interests and lifestyle:

{json.dumps(bios, indent=2)}

Extract:
1. Primary hobbies and interests
2. Lifestyle indicators (active, social, creative, etc.)
3. Values and priorities
4. Social engagement style

Respond with ONLY a JSON object:
{{
    "primary_interests": ["photography", "travel", "fitness"],
    "lifestyle_indicators": ["active", "social", "health-conscious"],
    "values": ["adventure", "creativity", "personal growth"],
    "social_style": "outgoing and engaging"
}}
"""
        
        try:
            response = await self.generate_response(prompt, max_tokens=400, temperature=0.3)
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
                
        except Exception as e:
            self.logger.error(f"LLM interest analysis failed: {e}")
        
        return {}
    
    async def _llm_personality_analysis(self, content: str) -> List[str]:
        """Use LLM to analyze personality traits from content."""
        prompt = f"""
Analyze this social media content for personality traits:

"{content}"

Identify personality traits that can be inferred from the language, interests, and style.

Common traits to consider:
- Outgoing vs Introverted
- Adventurous vs Cautious
- Creative vs Analytical
- Ambitious vs Laid-back
- Social vs Independent
- Optimistic vs Realistic

Respond with ONLY a JSON array of personality traits:
["outgoing", "adventurous", "creative"]
"""
        
        try:
            response = await self.generate_response(prompt, max_tokens=200, temperature=0.3)
            
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
                
        except Exception as e:
            self.logger.error(f"LLM personality analysis failed: {e}")
        
        return []
    
    def _calculate_profile_confidence(
        self, 
        profiles: List[SocialProfile], 
        linkedin_data: Optional[LinkedInProfile]
    ) -> float:
        """Calculate confidence score for profile analysis."""
        score = 0.0
        
        # Base score for having profiles
        if profiles:
            score += 0.3
        
        # Score for profile quality
        for profile in profiles:
            if profile.bio and len(profile.bio) > 20:
                score += 0.1
            if profile.verified:
                score += 0.1
            if profile.follower_count and profile.follower_count > 50:
                score += 0.05
        
        # Bonus for LinkedIn data
        if linkedin_data:
            score += 0.3
            if linkedin_data.experience:
                score += 0.1
            if linkedin_data.skills:
                score += 0.1
        
        return min(1.0, score)
    
    def _calculate_analysis_confidence(self, analysis_results: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence for all analysis results."""
        if not analysis_results:
            return 0.0
        
        total_confidence = sum(
            result.get("confidence_score", 0.5) 
            for result in analysis_results
        )
        
        return total_confidence / len(analysis_results)
