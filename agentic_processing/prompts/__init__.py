"""
Prompts package for agentic processing system.
"""

from .prompt_manager import PromptManager, prompt_manager
from .agent_prompts import *

__all__ = [
    "PromptManager",
    "prompt_manager",
    # Agent prompts
    "QUERY_PROCESSOR_PROMPT",
    "BIO_MATCHER_PROMPT", 
    "SOCIAL_FINDER_PROMPT",
    "PROFILE_ANALYZER_PROMPT",
    "COMPATIBILITY_SCORER_PROMPT",
    "SUMMARY_GENERATOR_PROMPT",
    # LLM prompts
    "LLM_PROFESSIONAL_ANALYSIS_PROMPT",
    "LLM_INTEREST_ANALYSIS_PROMPT",
    "LLM_PERSONALITY_ANALYSIS_PROMPT",
    "LLM_FIELD_COMPATIBILITY_PROMPT",
    "LLM_CONVERSATION_STARTERS_PROMPT",
    "LLM_DETAILED_SUMMARY_PROMPT"
]
