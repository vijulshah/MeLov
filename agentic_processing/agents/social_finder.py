"""
Sociafrom ..models.agentic_models import (
    AgentTask, AgentResponse, AgentRole, SocialProfile, SocialPlatform
)
from .base_agent import BaseAgent
from ..prompts.prompt_manager import prompt_managernder Agent - Finds social media profiles for matched bio data.
"""
import asyncio
import re
import json
from typing import Dict, Any, List, Optional
import aiohttp
from urllib.parse import quote

from ..models.agentic_models import (
    AgentTask, AgentResponse, AgentRole, BioMatch, SocialProfile, SocialPlatform
)
from .base_agent import BaseAgent


class SocialFinderAgent(BaseAgent):
    """Agent responsible for finding social media profiles for bio data matches."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = AgentRole.SOCIAL_FINDER
        
        # Social media search patterns
        self.linkedin_patterns = [
            r"linkedin\.com/in/([a-zA-Z0-9\-]+)",
            r"linkedin\.com/pub/([a-zA-Z0-9\-\s]+)"
        ]
        
        self.instagram_patterns = [
            r"instagram\.com/([a-zA-Z0-9\._]+)",
            r"@([a-zA-Z0-9\._]+)"
        ]
        
        # Mock data for demonstration (in production, use real APIs)
        self.use_mock_data = True
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for social media finding."""
        return prompt_manager.get_agent_prompt("social_finder")
    
    async def process_task(self, task: AgentTask) -> AgentResponse:
        """Process social media profile finding task."""
        try:
            # Extract input data
            bio_matches = task.input_data.get("bio_matches", [])
            
            if not bio_matches:
                raise ValueError("No bio matches provided")
            
            # Convert to BioMatch objects if they're dicts
            if bio_matches and isinstance(bio_matches[0], dict):
                bio_matches = [BioMatch(**match) for match in bio_matches]
            
            # Find social profiles for each match
            social_results = []
            for bio_match in bio_matches:
                profiles = await self._find_social_profiles(bio_match)
                social_results.append({
                    "bio_data_id": bio_match.bio_data_id,
                    "profiles": [profile.dict() for profile in profiles]
                })
            
            return AgentResponse(
                agent_name=self.name,
                agent_role=self.role,
                task_id=task.task_id,
                success=True,
                response_data={
                    "social_results": social_results,
                    "total_profiles_found": sum(len(result["profiles"]) for result in social_results),
                    "search_method": "mock" if self.use_mock_data else "api"
                },
                confidence_score=self._calculate_search_confidence(social_results)
            )
            
        except Exception as e:
            self.logger.error(f"Social profile finding failed: {e}")
            raise
    
    async def _find_social_profiles(self, bio_match: BioMatch) -> List[SocialProfile]:
        """Find social media profiles for a bio match."""
        profiles = []
        
        # Extract search parameters
        name = bio_match.personal_info.get("name", "")
        location = bio_match.personal_info.get("location", "")
        profession = ""
        if bio_match.professional:
            profession = bio_match.professional.get("current_job", "")
        
        # Search LinkedIn
        linkedin_profiles = await self._search_linkedin(name, profession, location)
        profiles.extend(linkedin_profiles)
        
        # Search Instagram
        instagram_profiles = await self._search_instagram(name, bio_match.interests)
        profiles.extend(instagram_profiles)
        
        # Search Facebook (limited due to API restrictions)
        facebook_profiles = await self._search_facebook(name, location)
        profiles.extend(facebook_profiles)
        
        return profiles
    
    async def _search_linkedin(
        self, 
        name: str, 
        profession: str, 
        location: str
    ) -> List[SocialProfile]:
        """Search for LinkedIn profiles."""
        if self.use_mock_data:
            return self._generate_mock_linkedin_profiles(name, profession, location)
        
        # In production, use LinkedIn API or web scraping (with proper permissions)
        # For now, return mock data
        return self._generate_mock_linkedin_profiles(name, profession, location)
    
    async def _search_instagram(
        self, 
        name: str, 
        interests: Optional[Dict[str, Any]]
    ) -> List[SocialProfile]:
        """Search for Instagram profiles."""
        if self.use_mock_data:
            return self._generate_mock_instagram_profiles(name, interests)
        
        # In production, use Instagram Basic Display API
        # For now, return mock data
        return self._generate_mock_instagram_profiles(name, interests)
    
    async def _search_facebook(self, name: str, location: str) -> List[SocialProfile]:
        """Search for Facebook profiles."""
        # Facebook API has strict limitations for profile search
        # Return empty list for now
        return []
    
    def _generate_mock_linkedin_profiles(
        self, 
        name: str, 
        profession: str, 
        location: str
    ) -> List[SocialProfile]:
        """Generate mock LinkedIn profiles for testing."""
        profiles = []
        
        if name and profession:
            # Generate 1-2 potential LinkedIn profiles
            for i in range(1, 3):
                username = self._generate_username_from_name(name, i)
                profile = SocialProfile(
                    platform=SocialPlatform.LINKEDIN,
                    profile_url=f"https://linkedin.com/in/{username}",
                    username=username,
                    display_name=name if i == 1 else f"{name} (Profile {i})",
                    bio=f"{profession} at Mock Company {i}" if profession else None,
                    follower_count=150 + (i * 50),
                    following_count=200 + (i * 30),
                    verified=i == 1,  # First profile is verified
                    confidence_score=0.8 - (i * 0.2)  # Decreasing confidence
                )
                profiles.append(profile)
        
        return profiles
    
    def _generate_mock_instagram_profiles(
        self, 
        name: str, 
        interests: Optional[Dict[str, Any]]
    ) -> List[SocialProfile]:
        """Generate mock Instagram profiles for testing."""
        profiles = []
        
        if name:
            # Generate 0-1 Instagram profile (not everyone has Instagram)
            if hash(name) % 3 != 0:  # 2/3 chance of having Instagram
                username = self._generate_instagram_username(name)
                
                # Generate bio based on interests
                bio_parts = []
                if interests:
                    for category, items in interests.items():
                        if items and isinstance(items, list) and items:
                            bio_parts.append(f"📸 {items[0]}")
                            if len(bio_parts) >= 3:
                                break
                
                bio = " | ".join(bio_parts) if bio_parts else "Living life to the fullest 🌟"
                
                profile = SocialProfile(
                    platform=SocialPlatform.INSTAGRAM,
                    profile_url=f"https://instagram.com/{username}",
                    username=username,
                    display_name=name,
                    bio=bio,
                    follower_count=180 + hash(name) % 500,
                    following_count=120 + hash(name) % 300,
                    verified=False,
                    confidence_score=0.7
                )
                profiles.append(profile)
        
        return profiles
    
    def _generate_username_from_name(self, name: str, variant: int = 1) -> str:
        """Generate potential username from name."""
        # Clean name
        clean_name = re.sub(r'[^a-zA-Z\s]', '', name.lower())
        parts = clean_name.split()
        
        if len(parts) >= 2:
            first_name = parts[0]
            last_name = parts[-1]
            
            if variant == 1:
                return f"{first_name}-{last_name}"
            elif variant == 2:
                return f"{first_name}{last_name[0]}"
            else:
                return f"{first_name[0]}{last_name}"
        else:
            return clean_name.replace(" ", "")
    
    def _generate_instagram_username(self, name: str) -> str:
        """Generate Instagram username from name."""
        clean_name = re.sub(r'[^a-zA-Z\s]', '', name.lower())
        parts = clean_name.split()
        
        if len(parts) >= 2:
            # Various Instagram username patterns
            patterns = [
                f"{parts[0]}_{parts[-1]}",
                f"{parts[0]}{parts[-1]}",
                f"{parts[0]}.{parts[-1]}",
                f"{parts[0]}{parts[-1]}{hash(name) % 100:02d}"
            ]
            return patterns[hash(name) % len(patterns)]
        else:
            return clean_name.replace(" ", "_")
    
    async def _validate_profile_with_llm(
        self, 
        profile: SocialProfile, 
        bio_match: BioMatch
    ) -> float:
        """Use LLM to validate if profile matches the bio data."""
        prompt = f"""
Analyze whether this social media profile matches the given bio data:

Bio Data:
- Name: {bio_match.personal_info.get('name', 'Unknown')}
- Profession: {bio_match.professional.get('current_job', 'Unknown') if bio_match.professional else 'Unknown'}
- Location: {bio_match.personal_info.get('location', 'Unknown')}
- Interests: {json.dumps(bio_match.interests, indent=2) if bio_match.interests else 'Unknown'}

Social Profile:
- Platform: {profile.platform}
- Username: {profile.username}
- Display Name: {profile.display_name}
- Bio: {profile.bio}
- URL: {profile.profile_url}

Please assess the likelihood (0.0 to 1.0) that this profile belongs to the same person as the bio data.

Consider:
1. Name similarity
2. Professional information alignment
3. Location consistency
4. Interest/lifestyle matches
5. Profile completeness and authenticity

Respond with ONLY a number between 0.0 and 1.0.
"""
        
        try:
            response = await self.generate_response(prompt, max_tokens=50, temperature=0.1)
            
            # Extract number from response
            import re
            number_match = re.search(r'(\d*\.?\d+)', response.strip())
            if number_match:
                score = float(number_match.group(1))
                return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"LLM profile validation failed: {e}")
        
        # Fallback to simple heuristics
        return self._simple_profile_validation(profile, bio_match)
    
    def _simple_profile_validation(self, profile: SocialProfile, bio_match: BioMatch) -> float:
        """Simple heuristic-based profile validation."""
        score = 0.0
        
        # Name similarity
        bio_name = bio_match.personal_info.get('name', '').lower()
        profile_name = (profile.display_name or '').lower()
        
        if bio_name and profile_name:
            # Simple name matching
            bio_parts = set(bio_name.split())
            profile_parts = set(profile_name.split())
            
            if bio_parts & profile_parts:  # Any overlap
                score += 0.4
                
            if bio_parts == profile_parts:  # Exact match
                score += 0.3
        
        # Platform-specific validation
        if profile.platform == SocialPlatform.LINKEDIN:
            # LinkedIn should match professional info
            if bio_match.professional:
                job = bio_match.professional.get('current_job', '').lower()
                bio_text = (profile.bio or '').lower()
                
                if job and any(word in bio_text for word in job.split()):
                    score += 0.3
        
        elif profile.platform == SocialPlatform.INSTAGRAM:
            # Instagram should match interests
            if bio_match.interests and profile.bio:
                bio_text = profile.bio.lower()
                interest_matches = 0
                total_interests = 0
                
                for category, items in bio_match.interests.items():
                    if items and isinstance(items, list):
                        for item in items:
                            total_interests += 1
                            if item.lower() in bio_text:
                                interest_matches += 1
                
                if total_interests > 0:
                    score += (interest_matches / total_interests) * 0.3
        
        return min(1.0, score)
    
    def _calculate_search_confidence(self, social_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for social media search results."""
        if not social_results:
            return 0.0
        
        total_confidence = 0.0
        total_profiles = 0
        
        for result in social_results:
            profiles = result.get("profiles", [])
            for profile_dict in profiles:
                total_profiles += 1
                # Get confidence from profile dict
                confidence = profile_dict.get("confidence_score", 0.5)
                total_confidence += confidence
        
        if total_profiles == 0:
            return 0.0
        
        avg_confidence = total_confidence / total_profiles
        
        # Boost confidence if we found profiles for most matches
        coverage = len([r for r in social_results if r.get("profiles")]) / len(social_results)
        
        return min(1.0, avg_confidence * (0.5 + 0.5 * coverage))
