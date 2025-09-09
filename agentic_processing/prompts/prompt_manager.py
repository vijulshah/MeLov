"""
Prompt Manager for Agentic Processing System.
Handles loading and formatting of system prompts.
"""
from typing import Dict, Any
from .agent_prompts import *


class PromptManager:
    """Manages system prompts for all agents."""
    
    def __init__(self):
        """Initialize prompt manager."""
        self.agent_prompts = {
            "query_processor": QUERY_PROCESSOR_PROMPT,
            "bio_matcher": BIO_MATCHER_PROMPT,
            "social_finder": SOCIAL_FINDER_PROMPT,
            "profile_analyzer": PROFILE_ANALYZER_PROMPT,
            "compatibility_scorer": COMPATIBILITY_SCORER_PROMPT,
            "summary_generator": SUMMARY_GENERATOR_PROMPT
        }
        
        self.llm_prompts = {
            "professional_analysis": LLM_PROFESSIONAL_ANALYSIS_PROMPT,
            "interest_analysis": LLM_INTEREST_ANALYSIS_PROMPT,
            "personality_analysis": LLM_PERSONALITY_ANALYSIS_PROMPT,
            "field_compatibility": LLM_FIELD_COMPATIBILITY_PROMPT,
            "conversation_starters": LLM_CONVERSATION_STARTERS_PROMPT,
            "detailed_summary": LLM_DETAILED_SUMMARY_PROMPT
        }
    
    def get_agent_prompt(self, agent_name: str) -> str:
        """Get system prompt for an agent."""
        return self.agent_prompts.get(agent_name, "")
    
    def get_llm_prompt(self, prompt_name: str, **kwargs) -> str:
        """Get formatted LLM prompt with variables."""
        template = self.llm_prompts.get(prompt_name, "")
        
        if kwargs:
            try:
                return template.format(**kwargs)
            except KeyError as e:
                print(f"Warning: Missing variable {e} for prompt {prompt_name}")
                return template
        
        return template
    
    def list_agent_prompts(self) -> list:
        """List available agent prompts."""
        return list(self.agent_prompts.keys())
    
    def list_llm_prompts(self) -> list:
        """List available LLM prompts."""
        return list(self.llm_prompts.keys())
    
    def add_custom_prompt(self, name: str, prompt: str, prompt_type: str = "agent"):
        """Add a custom prompt."""
        if prompt_type == "agent":
            self.agent_prompts[name] = prompt
        elif prompt_type == "llm":
            self.llm_prompts[name] = prompt
        else:
            raise ValueError("prompt_type must be 'agent' or 'llm'")
    
    def update_prompt(self, name: str, prompt: str, prompt_type: str = "agent"):
        """Update an existing prompt."""
        self.add_custom_prompt(name, prompt, prompt_type)


# Global prompt manager instance
prompt_manager = PromptManager()
