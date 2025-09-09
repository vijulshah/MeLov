"""
LangGraph-based workflow orchestrator for agentic bio matching system.
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Annotated, TypedDict
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..models.agentic_models import (
    AgentTask, AgentResponse, WorkflowConfig, BioData, WorkflowExecution
)
from ..mcp import MCPClient, get_default_mcp_servers, get_all_tools
from ..agents.query_processor import QueryProcessorAgent
from ..agents.bio_matcher import BioMatcherAgent
from ..agents.social_finder import SocialFinderAgent
from ..agents.profile_analyzer import ProfileAnalyzerAgent
from ..agents.compatibility_scorer import CompatibilityScorerAgent
from ..agents.summary_generator import SummaryGeneratorAgent
from ..utils.config_manager import AgenticConfigManager


class WorkflowState(TypedDict):
    """State for the LangGraph workflow."""
    # Input data
    user_query: str
    user_bio_data: Dict[str, Any]
    workflow_config: Dict[str, Any]
    
    # Workflow metadata
    workflow_id: str
    execution_id: str
    current_step: str
    status: str
    start_time: str
    
    # Step results
    query_processing_result: Optional[Dict[str, Any]]
    bio_matching_result: Optional[Dict[str, Any]]
    social_finding_result: Optional[Dict[str, Any]]
    profile_analysis_result: Optional[Dict[str, Any]]
    compatibility_scoring_result: Optional[Dict[str, Any]]
    summary_generation_result: Optional[Dict[str, Any]]
    
    # Final outputs
    final_matches: List[Dict[str, Any]]
    execution_log: List[Dict[str, Any]]
    error_messages: List[str]
    
    # Messages for LLM conversation
    messages: List[BaseMessage]


class WorkflowStepEnum(str, Enum):
    """Workflow step enumeration."""
    QUERY_PROCESSING = "query_processing"
    BIO_MATCHING = "bio_matching"
    SOCIAL_FINDING = "social_finding"
    PROFILE_ANALYSIS = "profile_analysis"
    COMPATIBILITY_SCORING = "compatibility_scoring"
    SUMMARY_GENERATION = "summary_generation"
    WORKFLOW_COMPLETE = "workflow_complete"


class LangGraphWorkflowOrchestrator:
    """LangGraph-based workflow orchestrator with MCP integration."""
    
    def __init__(self, config_path: str = None):
        """Initialize the LangGraph workflow orchestrator."""
        self.config_manager = AgenticConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize MCP client
        self.mcp_client = MCPClient(get_default_mcp_servers())
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Create workflow graph
        self.workflow_graph = self._create_workflow_graph()
        
        # Memory for checkpointing
        self.memory = MemorySaver()
        
        # Compile the workflow
        self.compiled_workflow = self.workflow_graph.compile(checkpointer=self.memory)
        
        self.logger.info("LangGraph workflow orchestrator initialized")
    
    async def initialize(self):
        """Initialize the orchestrator and MCP client."""
        await self.mcp_client.initialize()
        self.logger.info("MCP client initialized with available tools")
    
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
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the LangGraph workflow graph."""
        # Create state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each workflow step
        workflow.add_node("initialize_workflow", self._initialize_workflow_node)
        workflow.add_node("query_processing", self._query_processing_node)
        workflow.add_node("bio_matching", self._bio_matching_node)
        workflow.add_node("social_finding", self._social_finding_node)
        workflow.add_node("profile_analysis", self._profile_analysis_node)
        workflow.add_node("compatibility_scoring", self._compatibility_scoring_node)
        workflow.add_node("summary_generation", self._summary_generation_node)
        workflow.add_node("finalize_workflow", self._finalize_workflow_node)
        
        # Add conditional routing
        workflow.add_node("route_workflow", self._route_workflow_node)
        
        # Define workflow edges
        workflow.set_entry_point("initialize_workflow")
        
        workflow.add_edge("initialize_workflow", "query_processing")
        workflow.add_edge("query_processing", "bio_matching")
        workflow.add_edge("bio_matching", "route_workflow")
        
        # Conditional edges from routing node
        workflow.add_conditional_edges(
            "route_workflow",
            self._should_continue_workflow,
            {
                "social_finding": "social_finding",
                "compatibility_scoring": "compatibility_scoring",
                "end": END
            }
        )
        
        workflow.add_edge("social_finding", "profile_analysis")
        workflow.add_edge("profile_analysis", "compatibility_scoring")
        workflow.add_edge("compatibility_scoring", "summary_generation")
        workflow.add_edge("summary_generation", "finalize_workflow")
        workflow.add_edge("finalize_workflow", END)
        
        return workflow
    
    async def execute_workflow(
        self, 
        user_query: str, 
        user_bio_data: Dict[str, Any],
        workflow_config: Optional[WorkflowConfig] = None
    ) -> Dict[str, Any]:
        """Execute the complete bio matching workflow using LangGraph."""
        
        # Generate unique IDs
        workflow_id = str(uuid.uuid4())
        execution_id = str(uuid.uuid4())
        
        # Use provided config or default
        if workflow_config is None:
            workflow_config = self.config.workflow_config
        
        # Initialize workflow state
        initial_state: WorkflowState = {
            "user_query": user_query,
            "user_bio_data": user_bio_data,
            "workflow_config": workflow_config.dict() if hasattr(workflow_config, 'dict') else workflow_config,
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "current_step": "initializing",
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "query_processing_result": None,
            "bio_matching_result": None,
            "social_finding_result": None,
            "profile_analysis_result": None,
            "compatibility_scoring_result": None,
            "summary_generation_result": None,
            "final_matches": [],
            "execution_log": [],
            "error_messages": [],
            "messages": [
                SystemMessage(content="Bio matching workflow initiated"),
                HumanMessage(content=user_query)
            ]
        }
        
        try:
            self.logger.info(f"Starting workflow execution {execution_id}")
            
            # Execute workflow
            config = {"configurable": {"thread_id": execution_id}}
            result = await self.compiled_workflow.ainvoke(initial_state, config=config)
            
            self.logger.info(f"Workflow execution {execution_id} completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow execution {execution_id} failed: {e}")
            raise
    
    async def _initialize_workflow_node(self, state: WorkflowState) -> WorkflowState:
        """Initialize the workflow."""
        self.logger.info(f"Initializing workflow {state['workflow_id']}")
        
        state["current_step"] = "query_processing"
        state["execution_log"].append({
            "timestamp": datetime.now().isoformat(),
            "step": "workflow_initialized",
            "workflow_id": state["workflow_id"],
            "execution_id": state["execution_id"]
        })
        
        return state
    
    async def _query_processing_node(self, state: WorkflowState) -> WorkflowState:
        """Process the user query using MCP tools."""
        self.logger.info("Executing query processing step")
        
        try:
            state["current_step"] = "query_processing"
            
            # Use MCP tool for query analysis
            query_result = await self.mcp_client.call_tool(
                "analyze_user_query",
                {
                    "user_query": state["user_query"],
                    "user_context": state["user_bio_data"]
                },
                server_name="bio_processing"
            )
            
            # Fallback to agent if MCP fails
            if not query_result:
                task = AgentTask(
                    task_id=str(uuid.uuid4()),
                    agent_name="QueryProcessor",
                    task_type="process_query",
                    input_data={"user_query": state["user_query"]},
                    priority="high"
                )
                
                agent_result = await self.agents["query_processor"].process_task(task)
                query_result = agent_result.response_data if agent_result.success else {}
            
            state["query_processing_result"] = query_result
            state["execution_log"].append({
                "timestamp": datetime.now().isoformat(),
                "step": "query_processing_completed",
                "result_summary": f"Processed query with {len(query_result.get('requirements', {}))} requirements"
            })
            
            self.logger.info("Query processing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            state["error_messages"].append(f"Query processing error: {str(e)}")
            state["query_processing_result"] = {}
        
        return state
    
    async def _bio_matching_node(self, state: WorkflowState) -> WorkflowState:
        """Find bio matches using vector search."""
        self.logger.info("Executing bio matching step")
        
        try:
            state["current_step"] = "bio_matching"
            
            query_result = state.get("query_processing_result", {})
            
            # Use MCP vector search tool
            search_result = await self.mcp_client.call_tool(
                "search_similar_profiles",
                {
                    "query_text": state["user_query"],
                    "top_k": query_result.get("max_matches", 20),
                    "filters": query_result.get("filters", {}),
                    "similarity_threshold": 0.7
                },
                server_name="vector_search"
            )
            
            # Fallback to agent if MCP fails
            if not search_result:
                task = AgentTask(
                    task_id=str(uuid.uuid4()),
                    agent_name="BioMatcher",
                    task_type="find_matches",
                    input_data={
                        "user_bio_data": state["user_bio_data"],
                        "requirements": query_result.get("requirements", {}),
                        "filters": query_result.get("filters", {}),
                        "max_matches": query_result.get("max_matches", 20)
                    },
                    priority="high"
                )
                
                agent_result = await self.agents["bio_matcher"].process_task(task)
                search_result = agent_result.response_data if agent_result.success else {"matches": []}
            
            state["bio_matching_result"] = search_result
            state["execution_log"].append({
                "timestamp": datetime.now().isoformat(),
                "step": "bio_matching_completed",
                "result_summary": f"Found {len(search_result.get('matches', []))} potential matches"
            })
            
            self.logger.info(f"Bio matching completed with {len(search_result.get('matches', []))} matches")
            
        except Exception as e:
            self.logger.error(f"Bio matching failed: {e}")
            state["error_messages"].append(f"Bio matching error: {str(e)}")
            state["bio_matching_result"] = {"matches": []}
        
        return state
    
    async def _route_workflow_node(self, state: WorkflowState) -> WorkflowState:
        """Route workflow based on configuration and results."""
        workflow_config = state.get("workflow_config", {})
        bio_matches = state.get("bio_matching_result", {}).get("matches", [])
        
        if not bio_matches:
            state["current_step"] = "workflow_complete"
        elif workflow_config.get("enable_social_search", True):
            state["current_step"] = "social_finding"
        else:
            state["current_step"] = "compatibility_scoring"
        
        return state
    
    def _should_continue_workflow(self, state: WorkflowState) -> str:
        """Determine next step in workflow."""
        current_step = state.get("current_step", "")
        workflow_config = state.get("workflow_config", {})
        bio_matches = state.get("bio_matching_result", {}).get("matches", [])
        
        if not bio_matches:
            return "end"
        elif current_step == "social_finding" or workflow_config.get("enable_social_search", True):
            return "social_finding"
        else:
            return "compatibility_scoring"
    
    async def _social_finding_node(self, state: WorkflowState) -> WorkflowState:
        """Find social media profiles for matches."""
        self.logger.info("Executing social finding step")
        
        try:
            state["current_step"] = "social_finding"
            
            bio_matches = state.get("bio_matching_result", {}).get("matches", [])
            
            # Use MCP social search tool
            social_result = await self.mcp_client.call_tool(
                "search_social_profiles",
                {
                    "candidates": bio_matches[:10],  # Limit for performance
                    "platforms": ["linkedin"],
                    "confidence_threshold": 0.8
                },
                server_name="social_analysis"
            )
            
            # Fallback to agent if MCP fails
            if not social_result:
                task = AgentTask(
                    task_id=str(uuid.uuid4()),
                    agent_name="SocialFinder",
                    task_type="find_social_profiles",
                    input_data={"bio_matches": bio_matches},
                    priority="medium"
                )
                
                agent_result = await self.agents["social_finder"].process_task(task)
                social_result = agent_result.response_data if agent_result.success else {}
            
            state["social_finding_result"] = social_result
            state["execution_log"].append({
                "timestamp": datetime.now().isoformat(),
                "step": "social_finding_completed",
                "result_summary": f"Found {social_result.get('total_profiles_found', 0)} social profiles"
            })
            
            self.logger.info("Social finding completed successfully")
            
        except Exception as e:
            self.logger.error(f"Social finding failed: {e}")
            state["error_messages"].append(f"Social finding error: {str(e)}")
            state["social_finding_result"] = {}
        
        return state
    
    async def _profile_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Analyze social media profiles."""
        self.logger.info("Executing profile analysis step")
        
        try:
            state["current_step"] = "profile_analysis"
            
            social_results = state.get("social_finding_result", {})
            
            # Use MCP analysis tool
            analysis_result = await self.mcp_client.call_tool(
                "analyze_linkedin_profile",
                {
                    "social_results": social_results.get("social_results", []),
                    "analysis_depth": "detailed"
                },
                server_name="social_analysis"
            )
            
            # Fallback to agent if MCP fails
            if not analysis_result:
                task = AgentTask(
                    task_id=str(uuid.uuid4()),
                    agent_name="ProfileAnalyzer",
                    task_type="analyze_profiles",
                    input_data={"social_results": social_results.get("social_results", [])},
                    priority="medium"
                )
                
                agent_result = await self.agents["profile_analyzer"].process_task(task)
                analysis_result = agent_result.response_data if agent_result.success else {}
            
            state["profile_analysis_result"] = analysis_result
            state["execution_log"].append({
                "timestamp": datetime.now().isoformat(),
                "step": "profile_analysis_completed",
                "result_summary": f"Analyzed {analysis_result.get('total_analyzed', 0)} profiles"
            })
            
            self.logger.info("Profile analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Profile analysis failed: {e}")
            state["error_messages"].append(f"Profile analysis error: {str(e)}")
            state["profile_analysis_result"] = {}
        
        return state
    
    async def _compatibility_scoring_node(self, state: WorkflowState) -> WorkflowState:
        """Calculate compatibility scores."""
        self.logger.info("Executing compatibility scoring step")
        
        try:
            state["current_step"] = "compatibility_scoring"
            
            bio_matches = state.get("bio_matching_result", {}).get("matches", [])
            analysis_results = state.get("profile_analysis_result", {})
            
            # Use MCP compatibility tool
            scoring_result = await self.mcp_client.call_tool(
                "batch_calculate_compatibility",
                {
                    "user_profile": state["user_bio_data"],
                    "candidate_profiles": bio_matches,
                    "social_analysis": analysis_results.get("analysis_results", []),
                    "max_candidates": 50
                },
                server_name="compatibility_scoring"
            )
            
            # Fallback to agent if MCP fails
            if not scoring_result:
                # Prepare match analyses data
                match_analyses = []
                if analysis_results:
                    match_analyses = analysis_results.get("analysis_results", [])
                else:
                    # Create basic analysis for bio matches without social data
                    for match in bio_matches:
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
                        "user_bio_data": state["user_bio_data"],
                        "user_analysis": analysis_results.get("analysis_results", [{}])[0] if analysis_results else None,
                        "match_analyses": match_analyses
                    },
                    priority="high"
                )
                
                agent_result = await self.agents["compatibility_scorer"].process_task(task)
                scoring_result = agent_result.response_data if agent_result.success else {}
            
            state["compatibility_scoring_result"] = scoring_result
            state["execution_log"].append({
                "timestamp": datetime.now().isoformat(),
                "step": "compatibility_scoring_completed",
                "result_summary": f"Calculated scores for {scoring_result.get('total_matches', 0)} matches"
            })
            
            self.logger.info("Compatibility scoring completed successfully")
            
        except Exception as e:
            self.logger.error(f"Compatibility scoring failed: {e}")
            state["error_messages"].append(f"Compatibility scoring error: {str(e)}")
            state["compatibility_scoring_result"] = {}
        
        return state
    
    async def _summary_generation_node(self, state: WorkflowState) -> WorkflowState:
        """Generate final summaries."""
        self.logger.info("Executing summary generation step")
        
        try:
            state["current_step"] = "summary_generation"
            
            compatibility_scores = state.get("compatibility_scoring_result", {})
            workflow_config = state.get("workflow_config", {})
            
            # Use MCP summary tool
            summary_result = await self.mcp_client.call_tool(
                "generate_profile_summary",
                {
                    "user_bio_data": state["user_bio_data"],
                    "compatibility_scores": compatibility_scores.get("compatibility_scores", []),
                    "top_k": workflow_config.get("max_final_results", 10),
                    "summary_type": "detailed" if workflow_config.get("detailed_summaries", True) else "brief"
                },
                server_name="bio_processing"
            )
            
            # Fallback to agent if MCP fails
            if not summary_result:
                task = AgentTask(
                    task_id=str(uuid.uuid4()),
                    agent_name="SummaryGenerator",
                    task_type="generate_summaries",
                    input_data={
                        "user_bio_data": state["user_bio_data"],
                        "compatibility_scores": compatibility_scores.get("compatibility_scores", []),
                        "top_k": workflow_config.get("max_final_results", 10),
                        "summary_type": "detailed" if workflow_config.get("detailed_summaries", True) else "brief"
                    },
                    priority="high"
                )
                
                agent_result = await self.agents["summary_generator"].process_task(task)
                summary_result = agent_result.response_data if agent_result.success else {}
            
            state["summary_generation_result"] = summary_result
            state["final_matches"] = summary_result.get("match_summaries", [])
            
            state["execution_log"].append({
                "timestamp": datetime.now().isoformat(),
                "step": "summary_generation_completed",
                "result_summary": f"Generated {len(state['final_matches'])} final summaries"
            })
            
            self.logger.info("Summary generation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            state["error_messages"].append(f"Summary generation error: {str(e)}")
            state["summary_generation_result"] = {}
            state["final_matches"] = []
        
        return state
    
    async def _finalize_workflow_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize the workflow."""
        self.logger.info("Finalizing workflow")
        
        state["current_step"] = "workflow_complete"
        state["status"] = "completed"
        
        end_time = datetime.now().isoformat()
        state["execution_log"].append({
            "timestamp": end_time,
            "step": "workflow_completed",
            "total_matches": len(state["final_matches"]),
            "execution_time": self._calculate_execution_time(state["start_time"], end_time),
            "errors": len(state["error_messages"])
        })
        
        self.logger.info(f"Workflow {state['workflow_id']} completed successfully")
        
        return state
    
    def _calculate_execution_time(self, start_time: str, end_time: str) -> float:
        """Calculate execution time in seconds."""
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            return (end - start).total_seconds()
        except:
            return 0.0
    
    async def get_workflow_status(self, execution_id: str) -> Dict[str, Any]:
        """Get workflow status by execution ID."""
        try:
            config = {"configurable": {"thread_id": execution_id}}
            state = await self.compiled_workflow.aget_state(config)
            
            return {
                "execution_id": execution_id,
                "current_step": state.values.get("current_step", "unknown"),
                "status": state.values.get("status", "unknown"),
                "progress": self._calculate_progress(state.values),
                "error_count": len(state.values.get("error_messages", [])),
                "matches_found": len(state.values.get("final_matches", []))
            }
        except Exception as e:
            self.logger.error(f"Error getting workflow status: {e}")
            return {"execution_id": execution_id, "status": "error", "error": str(e)}
    
    def _calculate_progress(self, state: Dict[str, Any]) -> float:
        """Calculate workflow progress percentage."""
        total_steps = 6
        completed_steps = 0
        
        step_results = [
            "query_processing_result",
            "bio_matching_result",
            "social_finding_result", 
            "profile_analysis_result",
            "compatibility_scoring_result",
            "summary_generation_result"
        ]
        
        for step_result in step_results:
            if state.get(step_result) is not None:
                completed_steps += 1
        
        return completed_steps / total_steps
    
    async def close(self):
        """Close the orchestrator and MCP client."""
        await self.mcp_client.close()
        self.logger.info("LangGraph workflow orchestrator closed")


# Convenience function for easy workflow execution
async def run_langgraph_bio_matching_workflow(
    user_query: str,
    user_bio_data: Dict[str, Any],
    config_path: str = None,
    enable_social_search: bool = True,
    max_results: int = 10
) -> Dict[str, Any]:
    """Convenience function to run the LangGraph bio matching workflow."""
    
    orchestrator = LangGraphWorkflowOrchestrator(config_path)
    await orchestrator.initialize()
    
    try:
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
