"""
Model Context Protocol (MCP) integration for agentic bio matching.
"""

from .mcp_client import MCPClient, MCPServer, MCPTool, MCPResource, MCPPrompt, get_default_mcp_servers
from .mcp_tools import (
    get_all_tools, 
    get_all_resources, 
    get_all_prompts, 
    get_tools_by_category,
    # Individual tools
    BIO_EXTRACTION_TOOL,
    BIO_STANDARDIZATION_TOOL,
    BIO_VALIDATION_TOOL,
    VECTOR_SEARCH_TOOL,
    VECTOR_INDEX_TOOL,
    VECTOR_EMBEDDING_TOOL,
    SOCIAL_PROFILE_SEARCH_TOOL,
    LINKEDIN_ANALYSIS_TOOL,
    SOCIAL_ACTIVITY_TOOL,
    COMPATIBILITY_SCORING_TOOL,
    BATCH_SCORING_TOOL,
    FACTOR_ANALYSIS_TOOL,
    PROFILE_SUMMARY_TOOL,
    MATCH_EXPLANATION_TOOL
)

__all__ = [
    "MCPClient",
    "MCPServer", 
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "get_default_mcp_servers",
    "get_all_tools",
    "get_all_resources", 
    "get_all_prompts",
    "get_tools_by_category",
    # Tools
    "BIO_EXTRACTION_TOOL",
    "BIO_STANDARDIZATION_TOOL", 
    "BIO_VALIDATION_TOOL",
    "VECTOR_SEARCH_TOOL",
    "VECTOR_INDEX_TOOL",
    "VECTOR_EMBEDDING_TOOL",
    "SOCIAL_PROFILE_SEARCH_TOOL",
    "LINKEDIN_ANALYSIS_TOOL",
    "SOCIAL_ACTIVITY_TOOL",
    "COMPATIBILITY_SCORING_TOOL",
    "BATCH_SCORING_TOOL",
    "FACTOR_ANALYSIS_TOOL",
    "PROFILE_SUMMARY_TOOL",
    "MATCH_EXPLANATION_TOOL"
]
