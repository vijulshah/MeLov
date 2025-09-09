"""
Configuration Manager for Agentic Processing System.
"""
import os
import yaml
from typing import Dict, Any, Optional

from ..models.agentic_models import AgenticConfig


class AgenticConfigManager:
    """Manages configuration for the agentic processing system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize config manager."""
        if config_path is None:
            # Default config path
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "config", 
                "agentic_config.yaml"
            )
        
        self.config_path = config_path
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config_data = yaml.safe_load(file)
                self._config = AgenticConfig(**config_data)
        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}. Using default config.")
            self._config = self._get_default_config()
        except Exception as e:
            print(f"Warning: Error loading config: {e}. Using default config.")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> AgenticConfig:
        """Get default configuration."""
        from ..models.agentic_models import (
            WorkflowConfig, AgentConfigs, AgentConfig, ModelConfig, LoggingConfig
        )
        
        # Default model config
        default_model = ModelConfig(
            model_name="microsoft/DialoGPT-medium",
            model_type="huggingface", 
            max_tokens=500,
            temperature=0.7,
            api_key="",
            timeout=30
        )
        
        # Default agent configs
        agent_config = AgentConfig(
            model_config=default_model,
            max_retries=3,
            timeout=60,
            custom_params={}
        )
        
        agent_configs = AgentConfigs(
            query_processor=agent_config,
            bio_matcher=agent_config,
            social_finder=agent_config,
            profile_analyzer=agent_config,
            compatibility_scorer=agent_config,
            summary_generator=agent_config
        )
        
        # Default workflow config
        workflow_config = WorkflowConfig(
            enable_social_search=True,
            enable_profile_analysis=True,
            detailed_summaries=True,
            max_final_results=10,
            parallel_processing=False
        )
        
        # Default logging config
        logging_config = LoggingConfig(
            log_level="INFO",
            log_to_file=True,
            log_to_console=True
        )
        
        return AgenticConfig(
            agent_configs=agent_configs,
            workflow_config=workflow_config,
            logging_config=logging_config
        )
    
    def get_config(self) -> AgenticConfig:
        """Get the current configuration."""
        return self._config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        # This would be implemented to update specific config sections
        pass
    
    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to file."""
        output_path = output_path or self.config_path
        
        try:
            config_dict = self._config.dict()
            
            with open(output_path, 'w') as file:
                yaml.dump(config_dict, file, default_flow_style=False, indent=2)
                
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """Get configuration for a specific agent."""
        agent_configs = self._config.agent_configs
        
        if hasattr(agent_configs, agent_name.lower()):
            return getattr(agent_configs, agent_name.lower())
        else:
            # Return default config for unknown agents
            return agent_configs.query_processor
