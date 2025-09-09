"""
Main Agentic Processing Interface - Demonstrates the complete bio matching workflow.
"""
import asyncio
import json
from typing import Dict, Any, Optional

from .workflow_orchestrator import AgenticWorkflowOrchestrator, run_bio_matching_workflow
from .models.agentic_models import BioData


class AgenticBioMatcher:
    """Main interface for the agentic bio matching system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the bio matcher."""
        self.orchestrator = AgenticWorkflowOrchestrator(config_path)
    
    async def find_matches(
        self, 
        user_query: str, 
        user_bio: Dict[str, Any],
        include_social_search: bool = True,
        max_results: int = 10,
        detailed_summaries: bool = True
    ) -> Dict[str, Any]:
        """
        Find bio matches using the complete agentic workflow.
        
        Args:
            user_query: Natural language description of what user is looking for
            user_bio: User's bio data dictionary
            include_social_search: Whether to search social media profiles
            max_results: Maximum number of results to return
            detailed_summaries: Whether to generate detailed summaries
            
        Returns:
            Complete workflow results with matches and summaries
        """
        
        return await run_bio_matching_workflow(
            user_query=user_query,
            user_bio_data=user_bio,
            enable_social_search=include_social_search,
            max_results=max_results
        )
    
    async def simple_bio_matching(
        self, 
        user_bio: Dict[str, Any],
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Simplified bio matching without social media analysis.
        
        Args:
            user_bio: User's bio data dictionary
            max_results: Maximum number of results to return
            
        Returns:
            Basic bio matching results
        """
        
        return await self.orchestrator.execute_simple_bio_matching(
            user_bio_data=user_bio,
            max_matches=max_results
        )
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get the current workflow status."""
        return self.orchestrator.get_workflow_status()


async def demo_agentic_workflow():
    """Demonstrate the agentic bio matching workflow."""
    
    print("🤖 Agentic Bio Matching System Demo")
    print("=" * 50)
    
    # Sample user bio data
    user_bio = {
        "id": "user_001",
        "name": "Alex Johnson",
        "age": 29,
        "location": "San Francisco, CA",
        "occupation": "Software Engineer",
        "education": "Stanford University - Computer Science",
        "interests": ["Technology", "Hiking", "Photography", "Coffee", "Travel"],
        "bio": "Passionate software engineer who loves building innovative products. Enjoy hiking on weekends, exploring new coffee shops, and capturing moments through photography. Always excited to travel and experience new cultures.",
        "looking_for": "Someone who shares my love for technology and adventure"
    }
    
    # Sample user query
    user_query = """
    I'm looking for someone who:
    - Works in tech or a creative field
    - Enjoys outdoor activities like hiking
    - Is interested in travel and exploring new places
    - Values personal growth and learning
    - Lives in the Bay Area or willing to relocate
    - Age range 25-35
    """
    
    print(f"👤 User: {user_bio['name']}")
    print(f"📝 Query: {user_query.strip()}")
    print("\n🔄 Starting agentic workflow...")
    
    try:
        # Initialize the bio matcher
        bio_matcher = AgenticBioMatcher()
        
        # Run the complete workflow
        results = await bio_matcher.find_matches(
            user_query=user_query,
            user_bio=user_bio,
            include_social_search=True,
            max_results=5,
            detailed_summaries=True
        )
        
        print("\n✅ Workflow completed successfully!")
        print(f"🎯 Found {results.get('total_matches', 0)} potential matches")
        
        # Display results
        if "final_results" in results:
            final_results = results["final_results"]
            
            print("\n📊 Summary Statistics:")
            stats = final_results.get("summary_statistics", {})
            print(f"   • Average Compatibility: {stats.get('average_compatibility', 0):.1%}")
            print(f"   • High Quality Matches: {stats.get('high_quality_count', 0)}")
            print(f"   • Matches to Focus On: {stats.get('matches_to_focus_on', 0)}")
            
            # Show top matches
            match_summaries = final_results.get("match_summaries", [])
            
            print(f"\n🏆 Top {min(3, len(match_summaries))} Matches:")
            for i, match in enumerate(match_summaries[:3], 1):
                print(f"\n   {i}. Match Score: {match['overall_score']:.1%}")
                print(f"      Summary: {match['summary_text'][:200]}...")
                
                if match.get('conversation_starters'):
                    print(f"      💬 Conversation Starter: \"{match['conversation_starters'][0]}\"")
            
            # Show recommendations
            recommendations = final_results.get("overall_recommendations", [])
            if recommendations:
                print(f"\n💡 Recommendations:")
                for rec in recommendations[:2]:
                    print(f"   • {rec['title']}: {rec['description']}")
        
        print(f"\n⏱️  Total execution time: {results.get('execution_time', 0):.1f} seconds")
        
        # Show workflow steps
        if results.get('execution_log'):
            print("\n📋 Workflow Steps Completed:")
            completed_steps = [
                log['step'] for log in results.get('execution_log', []) 
                if log['step'].endswith('_completed')
            ]
            for step in completed_steps:
                step_name = step.replace('_completed', '').replace('_', ' ').title()
                print(f"   ✓ {step_name}")
        
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        print("This is expected in demo mode without actual data sources.")


async def demo_simple_matching():
    """Demonstrate simple bio matching without social media."""
    
    print("\n" + "=" * 50)
    print("🔍 Simple Bio Matching Demo (No Social Media)")
    print("=" * 50)
    
    user_bio = {
        "id": "user_002", 
        "name": "Sarah Wilson",
        "age": 26,
        "location": "New York, NY",
        "occupation": "Marketing Manager",
        "education": "NYU - Business Administration",
        "interests": ["Marketing", "Yoga", "Books", "Wine", "Art"],
        "bio": "Creative marketing professional passionate about brand storytelling. Love practicing yoga, reading fiction, and exploring art galleries. Always up for trying new restaurants and wine tastings."
    }
    
    print(f"👤 User: {user_bio['name']}")
    
    try:
        bio_matcher = AgenticBioMatcher()
        
        results = await bio_matcher.simple_bio_matching(
            user_bio=user_bio,
            max_results=5
        )
        
        print("✅ Simple matching completed!")
        print(f"🎯 Found {results.get('total_matches', 0)} bio-based matches")
        
        if "final_results" in results:
            match_summaries = results["final_results"].get("match_summaries", [])
            print(f"\n📝 Top {min(3, len(match_summaries))} Bio Matches:")
            
            for i, match in enumerate(match_summaries[:3], 1):
                print(f"   {i}. Compatibility: {match['overall_score']:.1%}")
                print(f"      {match['summary_text'][:150]}...")
        
    except Exception as e:
        print(f"❌ Simple matching failed: {e}")
        print("This is expected in demo mode without actual data sources.")


def main():
    """Main function to run demos."""
    print("🚀 Starting Agentic Bio Matching Demos...\n")
    
    # Run both demos
    asyncio.run(demo_agentic_workflow())
    asyncio.run(demo_simple_matching())
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print("\nTo use in your application:")
    print("```python")
    print("from agentic_processing.main import AgenticBioMatcher")
    print("")
    print("bio_matcher = AgenticBioMatcher()")
    print("results = await bio_matcher.find_matches(")
    print("    user_query='Looking for someone who...',")
    print("    user_bio={'name': 'John', 'age': 30, ...}")
    print(")")
    print("```")


if __name__ == "__main__":
    main()
