"""
Enhanced Agentic Processing Package - Multi-Agent Bio Matching System

This package provides a sophisticated multi-agent system for bio data matching
using open-source language models, LangGraph workflows, and MCP integration.

Key Components:
- Enhanced multi-agent workflow with LangGraph orchestration
- Model Context Protocol (MCP) for standardized tool management
- Open-source LLM integration (Phi-3, Llama-3.2, Mistral-7B, Falcon-7B)
- Social media profile discovery and analysis
- Comprehensive compatibility scoring
- Intelligent match summaries and recommendations

New Features in v2.0:
- LangGraph workflow orchestration with state management
- MCP (Model Context Protocol) integration for tools and resources
- Hybrid orchestration modes (Legacy, LangGraph, Hybrid)
- Enhanced performance monitoring and debugging
- Automatic fallback mechanisms for reliability

Usage:
    # Basic usage (legacy compatible)
    from agentic_processing import AgenticBioMatcher
    
    bio_matcher = AgenticBioMatcher()
    results = await bio_matcher.find_matches(
        user_query="Looking for someone who loves technology and travel",
        user_bio={"name": "John", "age": 30, "occupation": "Engineer", ...}
    )
    
    # Enhanced usage with LangGraph and MCP
    from agentic_processing.enhanced_main import EnhancedAgenticBioMatcher
    from agentic_processing.workflow_orchestrator import WorkflowOrchestrationMode
    
    bio_matcher = EnhancedAgenticBioMatcher(
        orchestration_mode=WorkflowOrchestrationMode.HYBRID,
        enable_mcp=True
    )
    
    await bio_matcher.initialize()
    results = await bio_matcher.find_matches(
        user_query="Looking for compatible matches",
        user_bio=user_bio_data
    )
    await bio_matcher.close()
"""

# Legacy imports (backward compatibility)
from .main import AgenticBioMatcher
from .workflow_orchestrator import (
    AgenticWorkflowOrchestrator, 
    run_bio_matching_workflow,
    WorkflowOrchestrationMode
)
from .models.agentic_models import (
    BioData, AgentTask, AgentResponse, CompatibilityScore, MatchSummary, WorkflowConfig
)

# Enhanced imports (new features)
try:
    from .enhanced_main import EnhancedAgenticBioMatcher
    from .langgraph_orchestrator import LangGraphWorkflowOrchestrator
    from .workflow_orchestrator import (
        run_langgraph_workflow,
        run_legacy_workflow
    )
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False

# MCP imports (if available)
try:
    from .mcp import (
        MCPClient, 
        MCPServer, 
        MCPTool, 
        MCPResource, 
        MCPPrompt,
        get_default_mcp_servers
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Import all agents for advanced usage
from .agents.query_processor import QueryProcessorAgent
from .agents.bio_matcher import BioMatcherAgent
from .agents.social_finder import SocialFinderAgent  
from .agents.profile_analyzer import ProfileAnalyzerAgent
from .agents.compatibility_scorer import CompatibilityScorerAgent
from .agents.summary_generator import SummaryGeneratorAgent

__version__ = "2.0.0"
__author__ = "MeLov Agentic Processing Team"

# Basic exports (always available)
__all__ = [
    # Main interfaces
    "AgenticBioMatcher",
    "AgenticWorkflowOrchestrator", 
    "run_bio_matching_workflow",
    "WorkflowOrchestrationMode",
    
    # Data models
    "BioData",
    "AgentTask", 
    "AgentResponse",
    "CompatibilityScore",
    "MatchSummary",
    "WorkflowConfig",
    
    # Individual agents (for advanced usage)
    "QueryProcessorAgent",
    "BioMatcherAgent", 
    "SocialFinderAgent",
    "ProfileAnalyzerAgent",
    "CompatibilityScorerAgent",
    "SummaryGeneratorAgent"
]

# Add enhanced features if available
if ENHANCED_FEATURES_AVAILABLE:
    __all__.extend([
        "EnhancedAgenticBioMatcher",
        "LangGraphWorkflowOrchestrator",
        "run_langgraph_workflow",
        "run_legacy_workflow"
    ])

# Add MCP features if available
if MCP_AVAILABLE:
    __all__.extend([
        "MCPClient",
        "MCPServer", 
        "MCPTool",
        "MCPResource",
        "MCPPrompt",
        "get_default_mcp_servers"
    ])

# Feature availability flags
__features__ = {
    "langgraph": ENHANCED_FEATURES_AVAILABLE,
    "mcp": MCP_AVAILABLE,
    "legacy": True,
    "hybrid_mode": ENHANCED_FEATURES_AVAILABLE
}
