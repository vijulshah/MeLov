"""Utilities for API-based media understanding supporting Documents, Images, Videos, and Audio."""

import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from data_models.api_media_understanding_data_models import (
    ApiMediaUnderstandingCompleteConfig,
    ApiMediaUnderstandingConfig,
    ApiMediaUnderstandingOutputStructure,
    ApiMediaUnderstandingPathConfig,
    ApiMediaUnderstandingPromptConfig,
    ApiMediaUnderstandingResult,
    MediaFile,
    MediaType,
    ProcessingTask,
    TokenCalculator,
    VideoMetadata,
)
from google import genai
from google.genai import types
from PIL import Image


class ApiMediaUnderstandingConfigManager:
    """Manages configuration loading and validation for API-based media understanding."""

    @classmethod
    def load_config(cls, config_path: str) -> ApiMediaUnderstandingCompleteConfig:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config_data = yaml.safe_load(file)

            # Create configuration objects
            api_config = ApiMediaUnderstandingConfig(**config_data.get("api_media_understanding", {}))
            paths_config = ApiMediaUnderstandingPathConfig(**config_data.get("paths", {}))
            output_config = ApiMediaUnderstandingOutputStructure(**config_data.get("output_structure", {}))
            prompts_config = ApiMediaUnderstandingPromptConfig(**config_data.get("prompts", {}))

            return ApiMediaUnderstandingCompleteConfig(
                api_media_understanding=api_config,
                paths=paths_config,
                output_structure=output_config,
                prompts=prompts_config,
            )

        except Exception as e:
            logging.error(f"Failed to load configuration from {config_path}: {str(e)}")
            raise

    @classmethod
    def load_prompt_config(cls, config_path: str) -> ApiMediaUnderstandingPromptConfig:
        """Load prompt configuration from YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config_data = yaml.safe_load(file)

            prompts_data = config_data.get("prompts", {})
            return ApiMediaUnderstandingPromptConfig(**prompts_data)

        except Exception as e:
            logging.error(f"Failed to load prompt configuration from {config_path}: {str(e)}")
            raise


class MediaFileDetector:
    """Detects and analyzes media files for processing."""

    # Media type mappings based on MIME types
    MEDIA_TYPE_MAPPINGS = {
        # Images
        "image/jpeg": MediaType.IMAGE,
        "image/jpg": MediaType.IMAGE,
        "image/png": MediaType.IMAGE,
        "image/webp": MediaType.IMAGE,
        "image/heic": MediaType.IMAGE,
        "image/heif": MediaType.IMAGE,
        # Documents
        "application/pdf": MediaType.DOCUMENT,
        # Videos
        "video/mp4": MediaType.VIDEO,
        "video/mpeg": MediaType.VIDEO,
        "video/mov": MediaType.VIDEO,
        "video/avi": MediaType.VIDEO,
        "video/x-flv": MediaType.VIDEO,
        "video/mpg": MediaType.VIDEO,
        "video/webm": MediaType.VIDEO,
        "video/wmv": MediaType.VIDEO,
        "video/3gpp": MediaType.VIDEO,
        # Audio
        "audio/wav": MediaType.AUDIO,
        "audio/mp3": MediaType.AUDIO,
        "audio/aiff": MediaType.AUDIO,
        "audio/aac": MediaType.AUDIO,
        "audio/ogg": MediaType.AUDIO,
        "audio/flac": MediaType.AUDIO,
    }

    @classmethod
    def detect_media_type(cls, file_path: Path) -> Optional[MediaType]:
        """Detect media type from file path."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            return cls.MEDIA_TYPE_MAPPINGS.get(mime_type.lower())
        return None

    @classmethod
    def get_file_info(cls, file_path: Path) -> Optional[MediaFile]:
        """Get comprehensive file information."""
        if not file_path.exists():
            return None

        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            return None

        media_type = cls.MEDIA_TYPE_MAPPINGS.get(mime_type.lower())
        if not media_type:
            return None

        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        metadata = {
            "file_extension": file_path.suffix.lower(),
            "file_name": file_path.name,
            "last_modified": file_path.stat().st_mtime,
        }

        # Add media-specific metadata
        if media_type == MediaType.IMAGE:
            try:
                with Image.open(file_path) as img:
                    metadata.update(
                        {
                            "width": img.width,
                            "height": img.height,
                            "format": img.format,
                            "mode": img.mode,
                        }
                    )
            except Exception as e:
                logging.warning(f"Could not extract image metadata from {file_path}: {e}")

        return MediaFile(
            file_path=file_path,
            media_type=media_type,
            mime_type=mime_type,
            file_size_mb=file_size_mb,
            metadata=metadata,
        )

    @classmethod
    def find_media_files(
        cls,
        base_path: Path,
        media_types: Optional[List[MediaType]] = None,
        max_files: Optional[int] = None,
    ) -> List[MediaFile]:
        """Find all supported media files in a directory tree."""
        if not base_path.exists():
            return []

        media_files = []

        # Get all supported extensions
        supported_extensions = set()
        for mime_type in cls.MEDIA_TYPE_MAPPINGS.keys():
            # Get extension from mime type
            ext = mimetypes.guess_extension(mime_type)
            if ext:
                supported_extensions.add(ext.lower())

        # Search for files
        for file_path in base_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                media_file = cls.get_file_info(file_path)
                if media_file and (media_types is None or media_file.media_type in media_types):
                    media_files.append(media_file)

                    if max_files and len(media_files) >= max_files:
                        break

        return media_files


class ApiMediaUnderstander:
    """Handles API communication for media understanding with Google Gemini API."""

    def __init__(
        self,
        config: ApiMediaUnderstandingCompleteConfig,
        prompt_config: ApiMediaUnderstandingPromptConfig,
    ):
        self.config = config
        self.prompt_config = prompt_config
        self.api_key = self._get_api_key()

    def _get_api_key(self) -> str:
        """Get API key from environment variable."""
        api_key = os.getenv(self.config.api_media_understanding.api_key_env_var)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable: {self.config.api_media_understanding.api_key_env_var}"
            )
        return api_key

    def _should_use_files_api(self, media_file: MediaFile) -> bool:
        """Determine if Files API should be used based on file size."""
        return media_file.file_size_mb > self.config.api_media_understanding.use_files_api_threshold_mb

    def _upload_file_to_api(self, media_file: MediaFile) -> str:
        """Upload file using Files API and return file URI."""
        try:
            client = genai.Client(api_key=self.api_key)

            # Upload file
            uploaded_file = client.files.upload(file=str(media_file.file_path))

            logging.info(f"Uploaded {media_file.file_path} to Files API: {uploaded_file.name}")
            return uploaded_file.name

        except ImportError:
            raise ImportError("Google GenAI library not found. Please install with: pip install google-generativeai")
        except Exception as e:
            logging.error(f"Failed to upload file {media_file.file_path}: {str(e)}")
            raise

    def _prepare_content_parts(
        self,
        media_file: MediaFile,
        prompt: str,
        video_metadata: Optional[VideoMetadata] = None,
    ) -> List[Any]:
        """Prepare content parts for API request."""
        try:

            parts = []

            # Add text prompt
            parts.append(types.Part.from_text(prompt))

            # Handle file upload vs inline data
            if self._should_use_files_api(media_file):
                # Use Files API
                if not media_file.uploaded_file_uri:
                    media_file.uploaded_file_uri = self._upload_file_to_api(media_file)

                # Create file part
                if media_file.media_type == MediaType.VIDEO and video_metadata:
                    # Add video metadata
                    video_meta = types.VideoMetadata()
                    if video_metadata.fps:
                        video_meta.fps = video_metadata.fps
                    if video_metadata.start_offset:
                        video_meta.start_offset = video_metadata.start_offset
                    if video_metadata.end_offset:
                        video_meta.end_offset = video_metadata.end_offset

                    parts.append(
                        types.Part.from_uri(
                            file_uri=media_file.uploaded_file_uri,
                            video_metadata=video_meta,
                        )
                    )
                else:
                    parts.append(types.Part.from_uri(file_uri=media_file.uploaded_file_uri))
            else:
                # Use inline data
                with open(media_file.file_path, "rb") as f:
                    file_data = f.read()

                if media_file.media_type == MediaType.VIDEO and video_metadata:
                    # Add video metadata for inline data
                    video_meta = types.VideoMetadata()
                    if video_metadata.fps:
                        video_meta.fps = video_metadata.fps

                    parts.append(
                        types.Part.from_bytes(
                            data=file_data,
                            mime_type=media_file.mime_type,
                            video_metadata=video_meta,
                        )
                    )
                else:
                    parts.append(types.Part.from_bytes(data=file_data, mime_type=media_file.mime_type))

            return parts

        except ImportError:
            raise ImportError("Google GenAI library not found. Please install with: pip install google-generativeai")
        except Exception as e:
            logging.error(f"Failed to prepare content parts for {media_file.file_path}: {str(e)}")
            raise

    def _make_api_request(self, content_parts: List[Any]) -> Dict[str, Any]:
        """Make API request to Gemini."""
        try:

            client = genai.Client(api_key=self.api_key)

            # Create generation config
            generation_config = types.GenerateContentConfig(
                temperature=self.config.api_media_understanding.temperature,
                max_output_tokens=self.config.api_media_understanding.max_tokens,
            )

            # Make request
            response = client.models.generate_content(
                model=self.config.api_media_understanding.model_name,
                contents=content_parts,
                config=generation_config,
            )

            return {
                "text": response.text,
                "model": self.config.api_media_understanding.model_name,
                "success": True,
            }

        except Exception as e:
            logging.error(f"API request failed: {str(e)}")
            return {
                "text": "",
                "model": self.config.api_media_understanding.model_name,
                "success": False,
                "error": str(e),
            }

    def understand_media(
        self,
        media_file: MediaFile,
        task: ProcessingTask = ProcessingTask.DESCRIBE,
        custom_prompt: Optional[str] = None,
        video_metadata: Optional[VideoMetadata] = None,
    ) -> ApiMediaUnderstandingResult:
        """Understand a media file using the Gemini API."""
        start_time = time.time()

        try:
            # Get appropriate prompt
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = self._get_prompt_for_task(media_file.media_type, task)

            # Prepare content parts
            content_parts = self._prepare_content_parts(media_file, prompt, video_metadata)

            # Make API request
            response_data = self._make_api_request(content_parts)

            processing_time = time.time() - start_time

            # Calculate token usage (estimate)
            estimated_tokens = self._estimate_token_usage(media_file, response_data.get("text", ""))

            return ApiMediaUnderstandingResult(
                file_path=media_file.file_path,
                media_type=media_file.media_type,
                task=task,
                content=response_data.get("text", ""),
                processing_time=processing_time,
                success=response_data.get("success", False),
                error_message=response_data.get("error"),
                model_used=response_data.get("model", ""),
                tokens_used=estimated_tokens,
                file_size_mb=media_file.file_size_mb,
                uploaded_file_uri=media_file.uploaded_file_uri,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logging.error(f"Failed to understand media {media_file.file_path}: {str(e)}")

            return ApiMediaUnderstandingResult(
                file_path=media_file.file_path,
                media_type=media_file.media_type,
                task=task,
                content="",
                processing_time=processing_time,
                success=False,
                error_message=str(e),
                model_used=self.config.api_media_understanding.model_name,
                file_size_mb=media_file.file_size_mb,
            )

    def _get_prompt_for_task(self, media_type: MediaType, task: ProcessingTask) -> str:
        """Get appropriate prompt for media type and task."""
        # Get system prompt for media type
        system_prompt = self.prompt_config.system_prompts.get(
            media_type.value, self.prompt_config.system_prompts.get("multimodal", "")
        )

        # Get task-specific prompt template
        task_template = self.prompt_config.task_prompts.get(
            task.value, "Analyze this {media_type} and provide insights."
        )

        # Format the template
        task_prompt = task_template.format(media_type=media_type.value)

        # Combine system and task prompts
        if system_prompt:
            return f"{system_prompt}\n\n{task_prompt}"
        return task_prompt

    def _estimate_token_usage(self, media_file: MediaFile, response_text: str) -> int:
        """Estimate token usage for the request."""
        # Response tokens (rough estimate: 1 token per 4 characters)
        response_tokens = len(response_text) // 4

        # Media tokens based on type
        if media_file.media_type == MediaType.IMAGE:
            width = media_file.metadata.get("width", 1024)
            height = media_file.metadata.get("height", 1024)
            media_tokens = TokenCalculator.calculate_image_tokens(width, height)
        elif media_file.media_type == MediaType.VIDEO:
            # Estimate based on file size (rough approximation)
            estimated_duration = media_file.file_size_mb * 10  # Very rough estimate
            media_tokens = TokenCalculator.calculate_video_tokens(estimated_duration)
        elif media_file.media_type == MediaType.AUDIO:
            # Estimate based on file size (rough approximation)
            estimated_duration = media_file.file_size_mb * 30  # Very rough estimate
            media_tokens = TokenCalculator.calculate_audio_tokens(estimated_duration)
        elif media_file.media_type == MediaType.DOCUMENT:
            # Estimate pages based on file size (very rough)
            estimated_pages = max(1, int(media_file.file_size_mb))
            media_tokens = TokenCalculator.calculate_document_tokens(estimated_pages)
        else:
            media_tokens = 100  # Default estimate

        return media_tokens + response_tokens

    def should_process_media(self, media_file: MediaFile, task: ProcessingTask) -> bool:
        """Check if media file should be processed (not already processed)."""
        if not self.config.api_media_understanding.overwrite_existing:
            # Check if output file already exists
            output_path = self._get_output_path(media_file, task)
            return not output_path.exists()
        return True

    def _get_output_path(self, media_file: MediaFile, task: ProcessingTask) -> Path:
        """Get output path for processed media file."""
        base_dir = Path(self.config.paths.processed_data_dir)

        # Create subdirectory based on task
        if task == ProcessingTask.TRANSCRIBE:
            subdir = self.config.output_structure.transcripts_dir
        elif task == ProcessingTask.SUMMARIZE:
            subdir = self.config.output_structure.summaries_dir
        elif task == ProcessingTask.EXTRACT:
            subdir = self.config.output_structure.extracted_data_dir
        else:
            subdir = self.config.output_structure.analysis_results_dir

        output_dir = base_dir / subdir / media_file.media_type.value
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create output filename
        base_name = media_file.file_path.stem
        output_filename = f"{base_name}_{task.value}.txt"

        return output_dir / output_filename

    def save_result(self, result: ApiMediaUnderstandingResult) -> Path:
        """Save processing result to file."""
        output_path = self._get_output_path(result.file_path, result.task)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# Media Understanding Result\n")
                f.write(f"File: {result.file_path}\n")
                f.write(f"Media Type: {result.media_type.value}\n")
                f.write(f"Task: {result.task.value}\n")
                f.write(f"Model: {result.model_used}\n")
                f.write(f"Processing Time: {result.processing_time:.2f}s\n")
                f.write(f"Success: {result.success}\n")
                if result.error_message:
                    f.write(f"Error: {result.error_message}\n")
                f.write(f"\n## Content\n")
                f.write(result.content)

            logging.info(f"Saved result to: {output_path}")
            return output_path

        except Exception as e:
            logging.error(f"Failed to save result to {output_path}: {str(e)}")
            raise
