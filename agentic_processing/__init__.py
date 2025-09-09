"""
Agentic Processing Package - Multi-Agent Bio Matching System

This package provides a sophisticated multi-agent system for bio data matching
using open-source language models and social media integration.

Key Components:
- Multi-agent workflow with 6 specialized agents
- Open-source LLM integration (Phi-3, Llama-3.2, Mistral-7B, Falcon-7B)
- Social media profile discovery and analysis
- Comprehensive compatibility scoring
- Intelligent match summaries and recommendations

Usage:
    from agentic_processing import AgenticBioMatcher
    
    bio_matcher = AgenticBioMatcher()
    results = await bio_matcher.find_matches(
        user_query="Looking for someone who loves technology and travel",
        user_bio={"name": "John", "age": 30, "occupation": "Engineer", ...}
    )
"""

from .main import AgenticBioMatcher
from .workflow_orchestrator import AgenticWorkflowOrchestrator, run_bio_matching_workflow
from .models.agentic_models import (
    BioData, AgentTask, AgentResponse, CompatibilityScore, MatchSummary
)

# Import all agents for advanced usage
from .agents.query_processor import QueryProcessorAgent
from .agents.bio_matcher import BioMatcherAgent
from .agents.social_finder import SocialFinderAgent  
from .agents.profile_analyzer import ProfileAnalyzerAgent
from .agents.compatibility_scorer import CompatibilityScorerAgent
from .agents.summary_generator import SummaryGeneratorAgent

__version__ = "1.0.0"
__author__ = "MeLov Agentic Processing Team"

__all__ = [
    # Main interfaces
    "AgenticBioMatcher",
    "AgenticWorkflowOrchestrator", 
    "run_bio_matching_workflow",
    
    # Data models
    "BioData",
    "AgentTask", 
    "AgentResponse",
    "CompatibilityScore",
    "MatchSummary",
    
    # Individual agents (for advanced usage)
    "QueryProcessorAgent",
    "BioMatcherAgent", 
    "SocialFinderAgent",
    "ProfileAnalyzerAgent",
    "CompatibilityScorerAgent",
    "SummaryGeneratorAgent"
]
