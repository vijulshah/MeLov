"""
Configuration manager for bio-data extraction.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, validator


class ConfigManager:
    """Manages YAML configuration for bio-data extraction."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = self._get_default_config_path()
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        project_root = Path(__file__).parent.parent.parent
        return project_root / "data_processing" / "config" / "bio_extraction_config.yaml"
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key using dot notation.
        
        Args:
            key: Configuration key (e.g., 'pdf_processing.input_path')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_pdf_input_path(self, bio_type: Literal["my_biodata", "ppl_biodata"] = "my_biodata") -> str:
        """
        Get PDF input path for specific bio data type.
        
        Args:
            bio_type: Type of bio data ('my_biodata' or 'ppl_biodata')
            
        Returns:
            Input path for the specified bio data type
        """
        return self.get(f'pdf_processing.{bio_type}.input_path', f'data/{bio_type}')
    
    def get_output_path(self, bio_type: Literal["my_biodata", "ppl_biodata"] = "my_biodata") -> str:
        """
        Get output path for specific bio data type.
        
        Args:
            bio_type: Type of bio data ('my_biodata' or 'ppl_biodata')
            
        Returns:
            Output path for the specified bio data type
        """
        return self.get(f'pdf_processing.{bio_type}.output_path', f'data/processed/{bio_type}')
    
    def get_bio_fields(self) -> Dict[str, list]:
        """Get bio data fields configuration."""
        return self.get('bio_data_fields', {})
    
    def get_extraction_settings(self) -> Dict[str, Any]:
        """Get extraction settings."""
        return self.get('extraction', {})
    
    def get_output_settings(self) -> Dict[str, Any]:
        """Get output settings."""
        return self.get('output', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get('logging', {})
    
    def update_config(self, key: str, value: Any) -> None:
        """
        Update configuration value.
        
        Args:
            key: Configuration key using dot notation
            value: New value
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save_config(self) -> None:
        """Save current configuration back to file."""
        with open(self.config_path, 'w', encoding='utf-8') as file:
            yaml.dump(self._config, file, default_flow_style=False, indent=2)
    
    def validate_paths(self, bio_type: Optional[Literal["my_biodata", "ppl_biodata"]] = None) -> bool:
        """
        Validate that configured paths exist or can be created.
        
        Args:
            bio_type: Specific bio data type to validate, or None for all types
            
        Returns:
            True if paths are valid
        """
        bio_types = [bio_type] if bio_type else ["my_biodata", "ppl_biodata"]
        
        try:
            for bt in bio_types:
                input_path = Path(self.get_pdf_input_path(bt))
                output_path = Path(self.get_output_path(bt))
                
                # Create directories if they don't exist
                input_path.mkdir(parents=True, exist_ok=True)
                output_path.mkdir(parents=True, exist_ok=True)
            
            return True
        except Exception as e:
            print(f"Error creating directories: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"ConfigManager(config_path='{self.config_path}')"
