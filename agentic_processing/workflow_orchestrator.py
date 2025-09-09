"""
Agentic Workflow Orchestrator - Coordinates the multi-agent bio matching system.

This module now supports both legacy orchestration and LangGraph-based workflows.
"""
import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .models.agentic_models import (
    AgentTask, AgentResponse, WorkflowConfig, BioData
)
from .agents.query_processor import QueryProcessorAgent
from .agents.bio_matcher import BioMatcherAgent  
from .agents.social_finder import SocialFinderAgent
from .agents.profile_analyzer import ProfileAnalyzerAgent
from .agents.compatibility_scorer import CompatibilityScorerAgent
from .agents.summary_generator import SummaryGeneratorAgent
from .utils.config_manager import AgenticConfigManager

# Import LangGraph orchestrator if available
try:
    from .langgraph_orchestrator import LangGraphWorkflowOrchestrator
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# Import MCP client if available
try:
    from .mcp import MCPClient, get_default_mcp_servers
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class WorkflowOrchestrationMode:
    """Workflow orchestration modes."""
    LEGACY = "legacy"
    LANGGRAPH = "langgraph"
    HYBRID = "hybrid"


class AgenticWorkflowOrchestrator:
    """Enhanced workflow orchestrator with LangGraph and MCP support."""
    
    def __init__(
        self, 
        config_path: str = None,
        orchestration_mode: str = WorkflowOrchestrationMode.HYBRID,
        enable_mcp: bool = True
    ):
        """
        Initialize the workflow orchestrator.
        
        Args:
            config_path: Path to configuration file
            orchestration_mode: Orchestration mode (legacy, langgraph, hybrid)
            enable_mcp: Whether to enable MCP integration
        """
        self.config_manager = AgenticConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.orchestration_mode = orchestration_mode
        self.enable_mcp = enable_mcp and MCP_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize LangGraph orchestrator if available and requested
        self.langgraph_orchestrator = None
        if LANGGRAPH_AVAILABLE and orchestration_mode in [WorkflowOrchestrationMode.LANGGRAPH, WorkflowOrchestrationMode.HYBRID]:
            try:
                self.langgraph_orchestrator = LangGraphWorkflowOrchestrator(config_path)
                self.logger.info("LangGraph orchestrator initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LangGraph orchestrator: {e}")
                if orchestration_mode == WorkflowOrchestrationMode.LANGGRAPH:
                    self.orchestration_mode = WorkflowOrchestrationMode.LEGACY
                    self.logger.info("Falling back to legacy orchestration mode")
        
        # Initialize MCP client if available and requested
        self.mcp_client = None
        if self.enable_mcp:
            try:
                self.mcp_client = MCPClient(get_default_mcp_servers())
                self.logger.info("MCP client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize MCP client: {e}")
                self.enable_mcp = False
        
        # Workflow state
        self.workflow_id = None
        self.workflow_state = {}
        self.execution_log = []
        
        self.logger.info(f"Workflow orchestrator initialized in {self.orchestration_mode} mode")
    
    async def initialize(self):
        """Initialize the orchestrator and its components."""
        try:
            # Initialize LangGraph orchestrator if available
            if self.langgraph_orchestrator:
                await self.langgraph_orchestrator.initialize()
            
            # Initialize MCP client if available
            if self.mcp_client:
                await self.mcp_client.initialize()
            
            self.logger.info("Orchestrator initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise
        
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents with configuration."""
        agent_configs = self.config.agent_configs
        
        agents = {
            "query_processor": QueryProcessorAgent(
                name="QueryProcessor",
                config=agent_configs.query_processor
            ),
            "bio_matcher": BioMatcherAgent(
                name="BioMatcher", 
                config=agent_configs.bio_matcher
            ),
            "social_finder": SocialFinderAgent(
                name="SocialFinder",
                config=agent_configs.social_finder
            ),
            "profile_analyzer": ProfileAnalyzerAgent(
                name="ProfileAnalyzer",
                config=agent_configs.profile_analyzer
            ),
            "compatibility_scorer": CompatibilityScorerAgent(
                name="CompatibilityScorer",
                config=agent_configs.compatibility_scorer
            ),
            "summary_generator": SummaryGeneratorAgent(
                name="SummaryGenerator",
                config=agent_configs.summary_generator
            )
        }
        
        return agents
    
    async def execute_workflow(
        self, 
        user_query: str, 
        user_bio_data: Dict[str, Any],
        workflow_config: Optional[WorkflowConfig] = None,
        force_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete bio matching workflow.
        
        Args:
            user_query: User's bio matching query
            user_bio_data: User's bio data
            workflow_config: Workflow configuration
            force_mode: Force specific orchestration mode
            
        Returns:
            Workflow execution results
        """
        # Determine orchestration mode
        execution_mode = force_mode or self.orchestration_mode
        
        # Use LangGraph orchestrator if available and requested
        if execution_mode == WorkflowOrchestrationMode.LANGGRAPH and self.langgraph_orchestrator:
            self.logger.info("Executing workflow using LangGraph orchestrator")
            return await self.langgraph_orchestrator.execute_workflow(
                user_query, user_bio_data, workflow_config
            )
        
        # Use hybrid mode (LangGraph with legacy fallback)
        elif execution_mode == WorkflowOrchestrationMode.HYBRID and self.langgraph_orchestrator:
            try:
                self.logger.info("Attempting workflow execution using LangGraph orchestrator")
                return await self.langgraph_orchestrator.execute_workflow(
                    user_query, user_bio_data, workflow_config
                )
            except Exception as e:
                self.logger.warning(f"LangGraph execution failed, falling back to legacy: {e}")
                return await self._execute_legacy_workflow(user_query, user_bio_data, workflow_config)
        
        # Use legacy orchestrator
        else:
            self.logger.info("Executing workflow using legacy orchestrator")
            return await self._execute_legacy_workflow(user_query, user_bio_data, workflow_config)
    
    async def _execute_legacy_workflow(
        self, 
        user_query: str, 
        user_bio_data: Dict[str, Any],
        workflow_config: Optional[WorkflowConfig] = None
    ) -> Dict[str, Any]:
        """Execute the legacy workflow implementation."""
        
        # Initialize workflow
        self.workflow_id = str(uuid.uuid4())
        self.workflow_state = {
            "workflow_id": self.workflow_id,
            "start_time": datetime.now().isoformat(),
            "user_query": user_query,
            "user_bio_data": user_bio_data,
            "status": "running"
        }
        
        try:
            # Use provided config or default
            if workflow_config is None:
                workflow_config = self.config.workflow_config
            
            self._log_step("workflow_started", {"workflow_id": self.workflow_id})
            
            # Step 1: Process Query
            query_result = await self._execute_query_processing(user_query)
            
            # Step 2: Find Bio Matches 
            bio_matches = await self._execute_bio_matching(user_bio_data, query_result)
            
            # Step 3: Find Social Profiles (if enabled)
            social_results = None
            if workflow_config.enable_social_search:
                social_results = await self._execute_social_finding(bio_matches)
            
            # Step 4: Analyze Profiles (if social data available)
            analysis_results = None
            if social_results and workflow_config.enable_profile_analysis:
                analysis_results = await self._execute_profile_analysis(social_results)
            
            # Step 5: Calculate Compatibility Scores
            compatibility_scores = await self._execute_compatibility_scoring(
                user_bio_data, bio_matches, analysis_results
            )
            
            # Step 6: Generate Summaries
            final_summaries = await self._execute_summary_generation(
                user_bio_data, compatibility_scores, workflow_config
            )
            
            # Complete workflow
            self.workflow_state.update({
                "status": "completed",
                "end_time": datetime.now().isoformat(),
                "total_matches": len(bio_matches.get("matches", [])),
                "final_results": final_summaries
            })
            
            self._log_step("workflow_completed", {
                "total_matches": len(bio_matches.get("matches", [])),
                "execution_time": self._calculate_execution_time()
            })
            
            return self.workflow_state
            
        except Exception as e:
            self.workflow_state.update({
                "status": "failed",
                "error": str(e),
                "end_time": datetime.now().isoformat()
            })
            
            self._log_step("workflow_failed", {"error": str(e)})
            raise
    
    async def _execute_query_processing(self, user_query: str) -> Dict[str, Any]:
        """Execute query processing step."""
        self._log_step("query_processing_started", {"query": user_query})
        
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            agent_name="QueryProcessor",
            task_type="process_query",
            input_data={"user_query": user_query},
            priority="high"
        )
        
        result = await self.agents["query_processor"].process_task(task)
        
        if not result.success:
            raise Exception(f"Query processing failed: {result.error_message}")
        
        self.workflow_state["query_processing"] = result.response_data
        self._log_step("query_processing_completed", result.response_data)
        
        return result.response_data
    
    async def _execute_bio_matching(
        self, 
        user_bio_data: Dict[str, Any], 
        query_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute bio matching step."""
        self._log_step("bio_matching_started", {})
        
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            agent_name="BioMatcher",
            task_type="find_matches",
            input_data={
                "user_bio_data": user_bio_data,
                "requirements": query_result.get("requirements", {}),
                "filters": query_result.get("filters", {}),
                "max_matches": query_result.get("max_matches", 20)
            },
            priority="high"
        )
        
        result = await self.agents["bio_matcher"].process_task(task)
        
        if not result.success:
            raise Exception(f"Bio matching failed: {result.error_message}")
        
        self.workflow_state["bio_matching"] = result.response_data
        self._log_step("bio_matching_completed", {
            "matches_found": len(result.response_data.get("matches", []))
        })
        
        return result.response_data
    
    async def _execute_social_finding(self, bio_matches: Dict[str, Any]) -> Dict[str, Any]:
        """Execute social profile finding step."""
        self._log_step("social_finding_started", {})
        
        matches = bio_matches.get("matches", [])
        
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            agent_name="SocialFinder",
            task_type="find_social_profiles",
            input_data={"bio_matches": matches},
            priority="medium"
        )
        
        result = await self.agents["social_finder"].process_task(task)
        
        if not result.success:
            raise Exception(f"Social finding failed: {result.error_message}")
        
        self.workflow_state["social_finding"] = result.response_data
        self._log_step("social_finding_completed", {
            "profiles_found": result.response_data.get("total_profiles_found", 0)
        })
        
        return result.response_data
    
    async def _execute_profile_analysis(self, social_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute profile analysis step."""
        self._log_step("profile_analysis_started", {})
        
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            agent_name="ProfileAnalyzer", 
            task_type="analyze_profiles",
            input_data={"social_results": social_results.get("social_results", [])},
            priority="medium"
        )
        
        result = await self.agents["profile_analyzer"].process_task(task)
        
        if not result.success:
            raise Exception(f"Profile analysis failed: {result.error_message}")
        
        self.workflow_state["profile_analysis"] = result.response_data
        self._log_step("profile_analysis_completed", {
            "profiles_analyzed": result.response_data.get("total_analyzed", 0)
        })
        
        return result.response_data
    
    async def _execute_compatibility_scoring(
        self, 
        user_bio_data: Dict[str, Any], 
        bio_matches: Dict[str, Any], 
        analysis_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute compatibility scoring step."""
        self._log_step("compatibility_scoring_started", {})
        
        # Prepare match analyses data
        match_analyses = []
        if analysis_results:
            match_analyses = analysis_results.get("analysis_results", [])
        else:
            # Create basic analysis for bio matches without social data
            for match in bio_matches.get("matches", []):
                match_analyses.append({
                    "bio_data_id": match["id"],
                    "professional_insights": {},
                    "interest_insights": {},
                    "personality_traits": [],
                    "activity_level": "unknown",
                    "confidence_score": 0.3
                })
        
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            agent_name="CompatibilityScorer",
            task_type="calculate_scores",
            input_data={
                "user_bio_data": user_bio_data,
                "user_analysis": analysis_results.get("analysis_results", [{}])[0] if analysis_results else None,
                "match_analyses": match_analyses
            },
            priority="high"
        )
        
        result = await self.agents["compatibility_scorer"].process_task(task)
        
        if not result.success:
            raise Exception(f"Compatibility scoring failed: {result.error_message}")
        
        self.workflow_state["compatibility_scoring"] = result.response_data
        self._log_step("compatibility_scoring_completed", {
            "scores_calculated": result.response_data.get("total_matches", 0)
        })
        
        return result.response_data
    
    async def _execute_summary_generation(
        self, 
        user_bio_data: Dict[str, Any], 
        compatibility_scores: Dict[str, Any], 
        workflow_config: WorkflowConfig
    ) -> Dict[str, Any]:
        """Execute summary generation step."""
        self._log_step("summary_generation_started", {})
        
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            agent_name="SummaryGenerator",
            task_type="generate_summaries",
            input_data={
                "user_bio_data": user_bio_data,
                "compatibility_scores": compatibility_scores.get("compatibility_scores", []),
                "top_k": workflow_config.max_final_results,
                "summary_type": "detailed" if workflow_config.detailed_summaries else "brief"
            },
            priority="high"
        )
        
        result = await self.agents["summary_generator"].process_task(task)
        
        if not result.success:
            raise Exception(f"Summary generation failed: {result.error_message}")
        
        self.workflow_state["summary_generation"] = result.response_data
        self._log_step("summary_generation_completed", {
            "summaries_generated": len(result.response_data.get("match_summaries", []))
        })
        
        return result.response_data
    
    def _log_step(self, step_name: str, data: Dict[str, Any]):
        """Log workflow execution step."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "workflow_id": self.workflow_id,
            "step": step_name,
            "data": data
        }
        
        self.execution_log.append(log_entry)
        
        # Optional: write to file or external logging system
        if self.config.logging_config.log_to_file:
            self._write_log_to_file(log_entry)
    
    def _write_log_to_file(self, log_entry: Dict[str, Any]):
        """Write log entry to file."""
        try:
            import os
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, f"workflow_{self.workflow_id}.log")
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")
    
    def _calculate_execution_time(self) -> float:
        """Calculate total workflow execution time."""
        if "start_time" not in self.workflow_state:
            return 0.0
        
        start_time = datetime.fromisoformat(self.workflow_state["start_time"])
        end_time = datetime.now()
        
        return (end_time - start_time).total_seconds()
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            "workflow_id": self.workflow_id,
            "status": self.workflow_state.get("status", "unknown"),
            "current_step": self._get_current_step(),
            "progress": self._calculate_progress(),
            "execution_time": self._calculate_execution_time(),
            "error": self.workflow_state.get("error")
        }
    
    def _get_current_step(self) -> str:
        """Get the current execution step."""
        if not self.execution_log:
            return "not_started"
        
        latest_step = self.execution_log[-1]["step"]
        
        if latest_step.endswith("_started"):
            return latest_step.replace("_started", "")
        elif latest_step.endswith("_completed"):
            return "completed"
        else:
            return latest_step
    
    def _calculate_progress(self) -> float:
        """Calculate workflow progress percentage."""
        total_steps = 6  # Total number of main steps
        completed_steps = 0
        
        step_keywords = [
            "query_processing_completed",
            "bio_matching_completed", 
            "social_finding_completed",
            "profile_analysis_completed",
            "compatibility_scoring_completed",
            "summary_generation_completed"
        ]
        
        for keyword in step_keywords:
            if any(keyword in log["step"] for log in self.execution_log):
                completed_steps += 1
        
        return completed_steps / total_steps
    
    async def close(self):
        """Close the orchestrator and its components."""
        try:
            # Close LangGraph orchestrator
            if self.langgraph_orchestrator:
                await self.langgraph_orchestrator.close()
            
            # Close MCP client
            if self.mcp_client:
                await self.mcp_client.close()
            
            self.logger.info("Orchestrator closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing orchestrator: {e}")
    
    def get_available_modes(self) -> List[str]:
        """Get available orchestration modes."""
        modes = [WorkflowOrchestrationMode.LEGACY]
        
        if LANGGRAPH_AVAILABLE and self.langgraph_orchestrator:
            modes.extend([
                WorkflowOrchestrationMode.LANGGRAPH,
                WorkflowOrchestrationMode.HYBRID
            ])
        
        return modes
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get orchestrator capabilities."""
        return {
            "orchestration_modes": self.get_available_modes(),
            "mcp_enabled": self.enable_mcp,
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "mcp_available": MCP_AVAILABLE,
            "current_mode": self.orchestration_mode,
            "agents": list(self.agents.keys())
        }


# Enhanced convenience functions
async def run_bio_matching_workflow(
    user_query: str,
    user_bio_data: Dict[str, Any],
    config_path: str = None,
    enable_social_search: bool = True,
    max_results: int = 10,
    orchestration_mode: str = WorkflowOrchestrationMode.HYBRID,
    enable_mcp: bool = True
) -> Dict[str, Any]:
    """
    Enhanced convenience function to run the bio matching workflow.
    
    Args:
        user_query: User's bio matching query
        user_bio_data: User's bio data
        config_path: Path to configuration file
        enable_social_search: Whether to enable social media search
        max_results: Maximum number of results to return
        orchestration_mode: Orchestration mode to use
        enable_mcp: Whether to enable MCP integration
        
    Returns:
        Workflow execution results
    """
    orchestrator = AgenticWorkflowOrchestrator(
        config_path=config_path,
        orchestration_mode=orchestration_mode,
        enable_mcp=enable_mcp
    )
    
    try:
        await orchestrator.initialize()
        
        workflow_config = WorkflowConfig(
            enable_social_search=enable_social_search,
            enable_profile_analysis=enable_social_search,
            detailed_summaries=True,
            max_final_results=max_results
        )
        
        result = await orchestrator.execute_workflow(user_query, user_bio_data, workflow_config)
        return result
        
    finally:
        await orchestrator.close()


async def run_langgraph_workflow(
    user_query: str,
    user_bio_data: Dict[str, Any],
    config_path: str = None,
    enable_social_search: bool = True,
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Run workflow specifically using LangGraph orchestrator.
    
    Args:
        user_query: User's bio matching query
        user_bio_data: User's bio data
        config_path: Path to configuration file
        enable_social_search: Whether to enable social media search
        max_results: Maximum number of results to return
        
    Returns:
        Workflow execution results
    """
    return await run_bio_matching_workflow(
        user_query=user_query,
        user_bio_data=user_bio_data,
        config_path=config_path,
        enable_social_search=enable_social_search,
        max_results=max_results,
        orchestration_mode=WorkflowOrchestrationMode.LANGGRAPH,
        enable_mcp=True
    )


async def run_legacy_workflow(
    user_query: str,
    user_bio_data: Dict[str, Any],
    config_path: str = None,
    enable_social_search: bool = True,
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Run workflow specifically using legacy orchestrator.
    
    Args:
        user_query: User's bio matching query
        user_bio_data: User's bio data
        config_path: Path to configuration file
        enable_social_search: Whether to enable social media search
        max_results: Maximum number of results to return
        
    Returns:
        Workflow execution results
    """
    return await run_bio_matching_workflow(
        user_query=user_query,
        user_bio_data=user_bio_data,
        config_path=config_path,
        enable_social_search=enable_social_search,
        max_results=max_results,
        orchestration_mode=WorkflowOrchestrationMode.LEGACY,
        enable_mcp=False
    )
