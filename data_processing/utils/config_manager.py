"""
Configuration manager for bio-data extraction.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml

# Handle imports for both script and module execution
try:
    # When run as module
    from ..models.bio_models import DataProcessingConfig
except ImportError:
    # When run as script or imported from script
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.bio_models import DataProcessingConfig


class ConfigManager:
    """Manages YAML configuration for bio-data extraction."""

    def __init__(self):
        """
        Initialize configuration manager.
        """
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "data_processing" / "config" / "bio_extraction_config.yaml"

        self.config_path = Path(config_path)
        self._config: DataProcessingConfig = self._load_config()

    def _load_config(self) -> DataProcessingConfig:
        """Load configuration from YAML file."""
        with open(self.config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return DataProcessingConfig(**config)

    def get_file_input_path(self, bio_type: Literal["my_biodata", "ppl_biodata"] = "my_biodata") -> str:
        """
        Get file input path for specific bio data type.

        Args:
            bio_type: Type of bio data ('my_biodata' or 'ppl_biodata')

        Returns:
            Input path for the specified bio data type
        """
        return getattr(self._config.file_processing, bio_type).input_path

    def get_pdf_input_path(self, bio_type: Literal["my_biodata", "ppl_biodata"] = "my_biodata") -> str:
        """
        Get PDF input path for specific bio data type (backward compatibility).

        Args:
            bio_type: Type of bio data ('my_biodata' or 'ppl_biodata')

        Returns:
            Input path for the specified bio data type
        """
        return self.get_file_input_path(bio_type)

    def get_output_path(self, bio_type: Literal["my_biodata", "ppl_biodata"] = "my_biodata") -> str:
        """
        Get output path for specific bio data type.

        Args:
            bio_type: Type of bio data ('my_biodata' or 'ppl_biodata')

        Returns:
            Output path for the specified bio data type
        """
        return getattr(self._config.file_processing, bio_type).output_path

    def get_images_output_path(self, bio_type: Literal["my_biodata", "ppl_biodata"] = "my_biodata") -> str:
        """
        Get images output path for specific bio data type.

        Args:
            bio_type: Type of bio data ('my_biodata' or 'ppl_biodata')

        Returns:
            Images output path for the specified bio data type
        """
        return getattr(self._config.file_processing, bio_type).images_output_path

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return self._config.file_processing.supported_formats

    def get_image_processing_config(self) -> Dict[str, Any]:
        """Get image processing configuration."""
        return self._config.image_processing

    def get_bio_fields(self) -> Dict[str, list]:
        """Get bio data fields configuration."""
        return self._config.bio_data_fields

    def get_extraction_settings(self) -> Dict[str, Any]:
        """Get extraction settings."""
        return self._config.extraction

    def get_output_settings(self) -> Dict[str, Any]:
        """Get output settings."""
        return self._config.output

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.logging

    def update_config(self, key: str, value: Any) -> None:
        """
        Update configuration value.

        Args:
            key: Configuration key using dot notation
            value: New value
        """
        keys = key.split(".")
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
        with open(self.config_path, "w", encoding="utf-8") as file:
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
