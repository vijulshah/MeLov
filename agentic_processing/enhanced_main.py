"""
Enhanced main entry point for the agentic bio matching system with LangGraph and MCP support.
"""
import asyncio
import json
import logging
import sys
from typing import Dict, Any, Optional
from pathlib import Path

from .workflow_orchestrator import (
    AgenticWorkflowOrchestrator,
    WorkflowOrchestrationMode,
    run_bio_matching_workflow,
    run_langgraph_workflow,
    run_legacy_workflow
)
from .models.agentic_models import WorkflowConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedAgenticBioMatcher:
    """Enhanced interface for the agentic bio matching system with LangGraph and MCP support."""
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        orchestration_mode: str = WorkflowOrchestrationMode.HYBRID,
        enable_mcp: bool = True
    ):
        """
        Initialize the enhanced bio matcher.
        
        Args:
            config_path: Path to configuration file
            orchestration_mode: Orchestration mode (legacy, langgraph, hybrid)
            enable_mcp: Whether to enable MCP integration
        """
        self.orchestrator = AgenticWorkflowOrchestrator(
            config_path=config_path,
            orchestration_mode=orchestration_mode,
            enable_mcp=enable_mcp
        )
        self.initialized = False
    
    async def initialize(self):
        """Initialize the bio matcher."""
        if not self.initialized:
            await self.orchestrator.initialize()
            self.initialized = True
    
    async def find_matches(
        self, 
        user_query: str, 
        user_bio: Dict[str, Any],
        include_social_search: bool = True,
        max_results: int = 10,
        detailed_summaries: bool = True,
        force_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find bio matches using the complete agentic workflow.
        
        Args:
            user_query: Natural language description of what user is looking for
            user_bio: User's bio data dictionary
            include_social_search: Whether to search social media profiles
            max_results: Maximum number of results to return
            detailed_summaries: Whether to generate detailed summaries
            force_mode: Force specific orchestration mode
            
        Returns:
            Complete workflow results with matches and summaries
        """
        await self.initialize()
        
        workflow_config = WorkflowConfig(
            enable_social_search=include_social_search,
            enable_profile_analysis=include_social_search,
            detailed_summaries=detailed_summaries,
            max_final_results=max_results
        )
        
        return await self.orchestrator.execute_workflow(
            user_query=user_query,
            user_bio_data=user_bio,
            workflow_config=workflow_config,
            force_mode=force_mode
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get bio matcher capabilities."""
        return self.orchestrator.get_capabilities()
    
    async def close(self):
        """Close the bio matcher."""
        if self.initialized:
            await self.orchestrator.close()
            self.initialized = False


async def run_example_workflow():
    """Run an example bio matching workflow."""
    logger.info("Starting example bio matching workflow")
    
    # Example user query and bio data
    user_query = """
    I'm looking for a life partner who is educated (preferably with a master's degree), 
    works in technology or finance, aged between 28-35, and shares similar interests 
    in travel, reading, and outdoor activities. Location preference is major cities.
    """
    
    user_bio_data = {
        "personal_info": {
            "name": "Alex Chen",
            "age": 30,
            "gender": "M",
            "location": "San Francisco, CA",
            "height": "5'10\"",
            "religion": "Non-religious"
        },
        "education": {
            "highest_degree": "Master's",
            "field_of_study": "Computer Science",
            "university": "Stanford University",
            "graduation_year": 2018
        },
        "professional": {
            "current_role": "Senior Software Engineer",
            "company": "Tech Startup",
            "industry": "Technology",
            "experience_years": 8,
            "annual_income": "$150,000"
        },
        "interests": {
            "hobbies": ["traveling", "reading", "hiking", "photography", "cooking"],
            "sports": ["tennis", "swimming"],
            "music": ["jazz", "classical", "indie rock"],
            "entertainment": ["documentaries", "foreign films", "theater"]
        },
        "lifestyle": {
            "fitness_level": "Active",
            "smoking": "Never",
            "drinking": "Socially",
            "diet": "Vegetarian",
            "pets": "Cat lover"
        },
        "relationship": {
            "looking_for": "Long-term relationship",
            "previous_marriages": 0,
            "wants_children": "Yes",
            "relationship_status": "Single"
        }
    }
    
    try:
        # Create enhanced bio matcher with hybrid mode
        bio_matcher = EnhancedAgenticBioMatcher(
            orchestration_mode=WorkflowOrchestrationMode.HYBRID,
            enable_mcp=True
        )
        
        await bio_matcher.initialize()
        
        # Display available capabilities
        capabilities = bio_matcher.get_capabilities()
        logger.info(f"Bio matcher capabilities: {json.dumps(capabilities, indent=2)}")
        
        # Execute workflow
        logger.info("Executing bio matching workflow...")
        result = await bio_matcher.find_matches(
            user_query=user_query,
            user_bio=user_bio_data,
            include_social_search=True,
            max_results=5,
            detailed_summaries=True
        )
        
        # Display results
        logger.info("Workflow execution completed!")
        logger.info(f"Status: {result.get('status', 'unknown')}")
        logger.info(f"Final matches found: {len(result.get('final_matches', []))}")
        
        # Display match summaries
        if result.get('final_matches'):
            logger.info("\nTop matches:")
            for i, match in enumerate(result['final_matches'][:3], 1):
                logger.info(f"{i}. {match.get('profile_summary', 'No summary available')}")
                logger.info(f"   Compatibility: {match.get('compatibility_score', 0):.1%}")
        
        # Save results
        output_file = "enhanced_workflow_results.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
        
        return result
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise
    
    finally:
        if 'bio_matcher' in locals():
            await bio_matcher.close()


async def run_comparison_test():
    """Run comparison test between different orchestration modes."""
    logger.info("Starting orchestration mode comparison test")
    
    # Simple test data
    user_query = "Find compatible matches for a tech professional"
    user_bio_data = {
        "personal_info": {"name": "Test User", "age": 30, "location": "NYC"},
        "professional": {"industry": "Technology", "role": "Software Engineer"},
        "interests": {"hobbies": ["reading", "travel"]}
    }
    
    modes_to_test = [
        WorkflowOrchestrationMode.LEGACY,
        WorkflowOrchestrationMode.LANGGRAPH,
        WorkflowOrchestrationMode.HYBRID
    ]
    
    results = {}
    
    for mode in modes_to_test:
        try:
            logger.info(f"Testing {mode} mode...")
            
            bio_matcher = EnhancedAgenticBioMatcher(
                orchestration_mode=mode,
                enable_mcp=(mode != WorkflowOrchestrationMode.LEGACY)
            )
            
            await bio_matcher.initialize()
            
            # Check if mode is available
            if mode not in bio_matcher.get_capabilities()["orchestration_modes"]:
                logger.warning(f"{mode} mode not available, skipping...")
                continue
            
            import time
            start_time = time.time()
            
            result = await bio_matcher.find_matches(
                user_query=user_query,
                user_bio=user_bio_data,
                include_social_search=False,  # Disable for faster testing
                max_results=3
            )
            
            execution_time = time.time() - start_time
            
            results[mode] = {
                "success": True,
                "execution_time": execution_time,
                "matches_found": len(result.get('final_matches', [])),
                "steps_completed": len(result.get('execution_log', []))
            }
            
            logger.info(f"{mode} completed in {execution_time:.2f}s")
            
            await bio_matcher.close()
            
        except Exception as e:
            logger.error(f"{mode} mode failed: {e}")
            results[mode] = {
                "success": False,
                "error": str(e)
            }
    
    # Display comparison results
    logger.info("\nComparison Results:")
    logger.info("-" * 50)
    for mode, result in results.items():
        logger.info(f"{mode.upper()}:")
        if result['success']:
            logger.info(f"  ✓ Success - {result['execution_time']:.2f}s")
            logger.info(f"  ✓ Matches: {result['matches_found']}")
            logger.info(f"  ✓ Steps: {result['steps_completed']}")
        else:
            logger.info(f"  ✗ Failed: {result['error']}")
        logger.info("")
    
    return results


async def interactive_demo():
    """Run interactive demo of the bio matching system."""
    logger.info("Starting interactive bio matching demo")
    
    print("\n" + "="*60)
    print("  ENHANCED AGENTIC BIO MATCHING SYSTEM DEMO")
    print("  LangGraph + MCP Integration")
    print("="*60)
    
    # Get user input
    print("\nEnter your bio matching query:")
    user_query = input("> ") or "Find me compatible matches for long-term relationship"
    
    print("\nChoose orchestration mode:")
    print("1. Hybrid (LangGraph + Legacy fallback)")
    print("2. LangGraph only")
    print("3. Legacy only")
    
    mode_choice = input("Choice (1-3): ") or "1"
    
    mode_map = {
        "1": WorkflowOrchestrationMode.HYBRID,
        "2": WorkflowOrchestrationMode.LANGGRAPH,
        "3": WorkflowOrchestrationMode.LEGACY
    }
    
    selected_mode = mode_map.get(mode_choice, WorkflowOrchestrationMode.HYBRID)
    
    # Simple user bio data for demo
    user_bio_data = {
        "personal_info": {
            "name": "Demo User",
            "age": 30,
            "location": "Demo City"
        },
        "interests": {
            "hobbies": ["demo", "testing"]
        }
    }
    
    try:
        print(f"\nExecuting workflow in {selected_mode} mode...")
        
        bio_matcher = EnhancedAgenticBioMatcher(
            orchestration_mode=selected_mode,
            enable_mcp=(selected_mode != WorkflowOrchestrationMode.LEGACY)
        )
        
        result = await bio_matcher.find_matches(
            user_query=user_query,
            user_bio=user_bio_data,
            max_results=3
        )
        
        print("\n" + "="*60)
        print("WORKFLOW RESULTS")
        print("="*60)
        
        print(f"Query: {user_query}")
        print(f"Mode: {selected_mode}")
        print(f"Status: {result.get('status', 'Unknown')}")
        print(f"Matches: {len(result.get('final_matches', []))}")
        
        if result.get('final_matches'):
            print("\nTop matches:")
            for i, match in enumerate(result['final_matches'], 1):
                print(f"{i}. {match.get('profile_summary', 'No summary')}")
        
        print("\n" + "="*60)
        
        await bio_matcher.close()
        
    except Exception as e:
        print(f"\nError: {e}")


async def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "example":
            await run_example_workflow()
        elif command == "compare":
            await run_comparison_test()
        elif command == "demo":
            await interactive_demo()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: example, compare, demo")
    else:
        # Default: run example workflow
        await run_example_workflow()


if __name__ == "__main__":
    asyncio.run(main())
