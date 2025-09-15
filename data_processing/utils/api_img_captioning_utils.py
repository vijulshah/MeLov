"""Simplified utility functions for API-based image captioning system - leveraging media understanding infrastructure."""

import logging
import os
from pathlib import Path
from typing import List

from data_models.api_img_captioning_data_models import (
    ApiCaptioningProcessingConfig,
    ApiCaptioningPromptConfig,
)


class CroppedImageDiscovery:
    """Handles discovery of cropped person images."""

    @staticmethod
    def find_cropped_images(base_path: Path, config=None) -> List[Path]:
        """Find all cropped person images in the processed data directory."""
        cropped_images = []
        cropped_dir_name = "cropped_persons"  # Use default name since we're simplifying

        # Recursively search for cropped_persons directories
        for root, dirs, files in os.walk(base_path):
            if cropped_dir_name in Path(root).name:
                # Add all PNG files in cropped_persons directories
                for file in files:
                    if file.lower().endswith(".png"):
                        image_path = Path(root) / file
                        cropped_images.append(image_path)

        logging.info(f"Found {len(cropped_images)} cropped person images")
        return sorted(cropped_images)


# Maintain minimal compatibility for existing code that might import these classes
# but redirect to use media understanding infrastructure instead


class ApiCaptioningConfigManager:
    """Legacy config manager - redirects to use media understanding config."""

    @staticmethod
    def load_config(config_path: str) -> ApiCaptioningProcessingConfig:
        """Load configuration - returns minimal config pointing to media understanding."""
        # Return a minimal config that uses media understanding
        from data_models.api_img_captioning_data_models import (
            ApiCaptioningConfig,
            ApiCaptioningProcessingConfig,
            OutputStructureConfig,
            PathsConfig,
            RateLimitsConfig,
        )

        # Create minimal config that points to media understanding
        api_config = ApiCaptioningConfig(
            api_endpoint="https://generativelanguage.googleapis.com/v1beta",
            api_key_env_var="GEMINI_API_KEY",
            model_name="gemini-2.5-flash",
            max_tokens=150,
            temperature=0.7,
            timeout=30,
            max_retries=3,
            retry_delay=1.0,
            batch_size=1,
            rate_limits=RateLimitsConfig(rpm=15, tpm=1000000, rpd=1500, max_concurrent_requests=5),
            overwrite_existing=False,
        )

        return ApiCaptioningProcessingConfig(
            api_captioning=api_config,
            paths=PathsConfig(processed_data_dir="./data/processed"),
            output_structure=OutputStructureConfig(cropped_persons_dir="cropped_persons"),
        )

    @staticmethod
    def save_config(config: ApiCaptioningProcessingConfig, config_path: str) -> None:
        """Legacy save method - no longer needed with media understanding integration."""
        logging.warning("Config saving disabled - using media understanding configuration instead")

    @staticmethod
    def load_prompt_config(config_path: str) -> ApiCaptioningPromptConfig:
        """Load prompt configuration for image captioning."""
        return ApiCaptioningPromptConfig(
            system_prompt="You are an expert at describing people in images. Provide concise, accurate descriptions focusing on visible characteristics.",
            user_prompt_template="Describe this person in detail, focusing on their appearance, clothing, and any visible characteristics. Keep the description under 50 words.",
        )


# Legacy classes that are no longer needed - provide deprecation warnings
class ImageEncoder:
    """Deprecated - use media understanding infrastructure instead."""

    def __init__(self):
        logging.warning("ImageEncoder is deprecated. Use MediaFileDetector from media understanding utils instead.")


class ApiRequestBuilder:
    """Deprecated - use media understanding infrastructure instead."""

    def __init__(self, config, prompt_config):
        logging.warning("ApiRequestBuilder is deprecated. Use media understanding request building instead.")


class ApiClient:
    """Deprecated - use media understanding infrastructure instead."""

    def __init__(self, config):
        logging.warning("ApiClient is deprecated. Use ApiMediaUnderstander from media understanding utils instead.")


class ApiImageCaptioner:
    """Deprecated - use media understanding infrastructure instead."""

    def __init__(self, config, prompt_config):
        logging.warning("ApiImageCaptioner is deprecated. Use ApiMediaUnderstandingProcessor instead.")
