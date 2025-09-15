"""Utility functions for the API-based image captioning system."""

import asyncio
import base64
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import aiohttp
import requests
import yaml
from data_models.api_img_captioning_data_models import (
    ApiCaptioningProcessingConfig,
    ApiCaptioningPromptConfig,
    ApiCaptioningResult,
    ApiRequestPayload,
    ApiResponse,
    ApiUsageStats,
)
from PIL import Image


class ApiCaptioningConfigManager:
    """Manages API captioning configuration loading and validation."""

    @staticmethod
    def load_config(config_path: str) -> ApiCaptioningProcessingConfig:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            return ApiCaptioningProcessingConfig(**config_data)
        else:
            # Return default configuration if file doesn't exist
            config = ApiCaptioningProcessingConfig()
            ApiCaptioningConfigManager.save_config(config, config_path)
            return config

    @staticmethod
    def save_config(config: ApiCaptioningProcessingConfig, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config.model_dump(), f, default_flow_style=False, indent=2)

    @staticmethod
    def load_prompt_config(config_path: str) -> ApiCaptioningPromptConfig:
        """Load prompt configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
                prompts_data = config_data.get("prompts", {})
            return ApiCaptioningPromptConfig(**prompts_data)
        else:
            return ApiCaptioningPromptConfig()


class ImageEncoder:
    """Handles image encoding for API requests."""

    @staticmethod
    def encode_image_to_base64(image_path: Path) -> str:
        """Encode image to base64 string for API request."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logging.error(f"Failed to encode image {image_path}: {str(e)}")
            raise

    @staticmethod
    def get_image_mime_type(image_path: Path) -> str:
        """Get MIME type for image based on file extension."""
        extension = image_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(extension, "image/png")

    @staticmethod
    def validate_image(image_path: Path) -> bool:
        """Validate that the image can be opened and processed."""
        try:
            with Image.open(image_path) as img:
                # Check if image is valid
                img.verify()
            return True
        except Exception as e:
            logging.error(f"Invalid image {image_path}: {str(e)}")
            return False


class ApiRequestBuilder:
    """Builds API requests for image captioning."""

    def __init__(
        self,
        config: ApiCaptioningProcessingConfig,
        prompt_config: ApiCaptioningPromptConfig,
    ):
        self.config = config
        self.prompt_config = prompt_config

    def build_request_payload(self, image_path: Path) -> ApiRequestPayload:
        """Build API request payload for image captioning."""
        # Encode image
        base64_image = ImageEncoder.encode_image_to_base64(image_path)
        mime_type = ImageEncoder.get_image_mime_type(image_path)

        # Build messages
        messages = [
            {"role": "system", "content": self.prompt_config.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt_config.user_prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                ],
            },
        ]

        return ApiRequestPayload(
            model=self.config.api_captioning.model_name,
            messages=messages,
            max_tokens=self.config.api_captioning.max_tokens,
            temperature=self.config.api_captioning.temperature,
        )

    def build_gemini_request(self, image_path: Path) -> dict:
        """Build Gemini API request for batch processing."""
        # Encode image
        base64_image = ImageEncoder.encode_image_to_base64(image_path)
        mime_type = ImageEncoder.get_image_mime_type(image_path)

        # Build Gemini-style request
        request = {
            "contents": [
                {
                    "parts": [
                        {"text": self.prompt_config.user_prompt_template},
                        {"inline_data": {"mime_type": mime_type, "data": base64_image}},
                    ],
                    "role": "user",
                }
            ],
            "system_instructions": {"parts": [{"text": self.prompt_config.system_prompt}]},
            "generation_config": {
                "temperature": self.config.api_captioning.temperature,
                "max_output_tokens": self.config.api_captioning.max_tokens,
            },
        }

        return request


class ApiClient:
    """Handles API communication for image captioning."""

    def __init__(self, config: ApiCaptioningProcessingConfig):
        self.config = config
        self.api_key = self._get_api_key()
        self.usage_stats = ApiUsageStats()

    def _get_api_key(self) -> str:
        """Get API key from environment variable."""
        api_key = os.getenv(self.config.api_captioning.api_key_env_var)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {self.config.api_captioning.api_key_env_var}")
        return api_key

    def _get_headers(self) -> dict:
        """Get headers for API request."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def make_request(self, payload: ApiRequestPayload) -> ApiResponse:
        """Make synchronous API request with retry logic."""
        start_time = time.time()
        last_exception = None

        for attempt in range(self.config.api_captioning.max_retries):
            try:
                response = requests.post(
                    str(self.config.api_captioning.api_endpoint),
                    headers=self._get_headers(),
                    json=payload.model_dump(),
                    timeout=self.config.api_captioning.timeout,
                )

                response_time = time.time() - start_time
                self._update_usage_stats(response_time, response.status_code == 200)

                if response.status_code == 200:
                    return ApiResponse(**response.json())
                else:
                    error_data = {
                        "status_code": response.status_code,
                        "message": response.text,
                    }
                    return ApiResponse(error=error_data)

            except Exception as e:
                last_exception = e
                logging.warning(f"API request attempt {attempt + 1} failed: {str(e)}")

                if attempt < self.config.api_captioning.max_retries - 1:
                    time.sleep(self.config.api_captioning.retry_delay)

        # All retries failed
        response_time = time.time() - start_time
        self._update_usage_stats(response_time, False)
        error_data = {
            "message": f"All {self.config.api_captioning.max_retries} retries failed. Last error: {str(last_exception)}"
        }
        return ApiResponse(error=error_data)

    async def make_request_async(self, payload: ApiRequestPayload) -> ApiResponse:
        """Make asynchronous API request with retry logic."""
        start_time = time.time()
        last_exception = None

        async with aiohttp.ClientSession() as session:
            for attempt in range(self.config.api_captioning.max_retries):
                try:
                    async with session.post(
                        str(self.config.api_captioning.api_endpoint),
                        headers=self._get_headers(),
                        json=payload.model_dump(),
                        timeout=aiohttp.ClientTimeout(total=self.config.api_captioning.timeout),
                    ) as response:
                        response_time = time.time() - start_time
                        self._update_usage_stats(response_time, response.status == 200)

                        if response.status == 200:
                            response_data = await response.json()
                            return ApiResponse(**response_data)
                        else:
                            response_text = await response.text()
                            error_data = {
                                "status_code": response.status,
                                "message": response_text,
                            }
                            return ApiResponse(error=error_data)

                except Exception as e:
                    last_exception = e
                    logging.warning(f"Async API request attempt {attempt + 1} failed: {str(e)}")

                    if attempt < self.config.api_captioning.max_retries - 1:
                        await asyncio.sleep(self.config.api_captioning.retry_delay)

            # All retries failed
            response_time = time.time() - start_time
            self._update_usage_stats(response_time, False)
            error_data = {
                "message": f"All {self.config.api_captioning.max_retries} retries failed. Last error: {str(last_exception)}"
            }
            return ApiResponse(error=error_data)

    def _update_usage_stats(self, response_time: float, success: bool) -> None:
        """Update usage statistics."""
        self.usage_stats.total_requests += 1
        if success:
            self.usage_stats.successful_requests += 1
        else:
            self.usage_stats.failed_requests += 1

        # Update average response time
        total_time = self.usage_stats.average_response_time * (self.usage_stats.total_requests - 1) + response_time
        self.usage_stats.average_response_time = total_time / self.usage_stats.total_requests


class ApiImageCaptioner:
    """Manages API-based image captioning."""

    def __init__(
        self,
        config: ApiCaptioningProcessingConfig,
        prompt_config: ApiCaptioningPromptConfig,
    ):
        self.config = config
        self.prompt_config = prompt_config
        self.request_builder = ApiRequestBuilder(config, prompt_config)
        self.api_client = ApiClient(config)

    def generate_caption(self, image_path: Path) -> ApiCaptioningResult:
        """Generate a caption for a single image using API synchronously."""
        start_time = time.time()

        try:
            # Validate image
            if not ImageEncoder.validate_image(image_path):
                return self._create_error_result(image_path, start_time, "Invalid image file")

            # Build request payload
            payload = self.request_builder.build_request_payload(image_path)

            # Make API request
            api_response = self.api_client.make_request(payload)

            # Process response
            if api_response.error:
                return self._create_error_result(
                    image_path,
                    start_time,
                    f"API error: {api_response.error}",
                    api_response,
                )

            # Extract caption from response
            caption = self._extract_caption_from_response(api_response)
            if not caption:
                return self._create_error_result(
                    image_path,
                    start_time,
                    "No caption found in API response",
                    api_response,
                )

            # Create caption file path
            caption_path = image_path.with_suffix(".txt")

            # Save caption
            success = self._save_caption(caption_path, caption)

            processing_time = time.time() - start_time

            return ApiCaptioningResult(
                image_path=image_path,
                caption=caption if success else None,
                caption_path=caption_path if success else None,
                processing_time=processing_time,
                api_response=api_response,
                success=success,
                retry_count=0,  # TODO: Track actual retry count from API client
            )

        except Exception as e:
            return self._create_error_result(image_path, start_time, str(e))

    async def generate_caption_async(self, image_path: Path) -> ApiCaptioningResult:
        """Generate a caption for a single image using API asynchronously."""
        start_time = time.time()

        try:
            # Validate image
            if not ImageEncoder.validate_image(image_path):
                return self._create_error_result(image_path, start_time, "Invalid image file")

            # Build request payload
            payload = self.request_builder.build_request_payload(image_path)

            # Make async API request
            api_response = await self.api_client.make_request_async(payload)

            # Process response
            if api_response.error:
                return self._create_error_result(
                    image_path,
                    start_time,
                    f"API error: {api_response.error}",
                    api_response,
                )

            # Extract caption from response
            caption = self._extract_caption_from_response(api_response)
            if not caption:
                return self._create_error_result(
                    image_path,
                    start_time,
                    "No caption found in API response",
                    api_response,
                )

            # Create caption file path
            caption_path = image_path.with_suffix(".txt")

            # Save caption
            success = self._save_caption(caption_path, caption)

            processing_time = time.time() - start_time

            return ApiCaptioningResult(
                image_path=image_path,
                caption=caption if success else None,
                caption_path=caption_path if success else None,
                processing_time=processing_time,
                api_response=api_response,
                success=success,
                retry_count=0,  # TODO: Track actual retry count from API client
            )

        except Exception as e:
            return self._create_error_result(image_path, start_time, str(e))

    def _extract_caption_from_response(self, response: ApiResponse) -> Optional[str]:
        """Extract caption text from API response."""
        try:
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"].strip()
        except Exception as e:
            logging.error(f"Failed to extract caption from response: {str(e)}")
        return None

    def _save_caption(self, caption_path: Path, caption: str) -> bool:
        """Save caption as a text file."""
        try:
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption)
            return True
        except Exception as e:
            logging.error(f"Failed to save caption to {caption_path}: {str(e)}")
            return False

    def _create_error_result(
        self,
        image_path: Path,
        start_time: float,
        error_message: str,
        api_response: Optional[ApiResponse] = None,
    ) -> ApiCaptioningResult:
        """Create an error result."""
        processing_time = time.time() - start_time
        logging.error(f"Failed to generate caption for {image_path}: {error_message}")

        return ApiCaptioningResult(
            image_path=image_path,
            processing_time=processing_time,
            api_response=api_response,
            success=False,
            error_message=error_message,
        )

    def should_process_image(self, image_path: Path) -> bool:
        """Check if image should be processed based on configuration."""
        caption_path = image_path.with_suffix(".txt")
        return self.config.api_captioning.overwrite_existing or not caption_path.exists()

    def get_usage_stats(self) -> ApiUsageStats:
        """Get current usage statistics."""
        return self.api_client.usage_stats


class CroppedImageDiscovery:
    """Handles discovery of cropped person images."""

    @staticmethod
    def find_cropped_images(base_path: Path, config: ApiCaptioningProcessingConfig) -> List[Path]:
        """Find all cropped person images in the processed data directory."""
        cropped_images = []
        cropped_dir_name = config.output_structure.cropped_persons_dir

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
