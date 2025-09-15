"""Pydantic models for the API-based media understanding system supporting Documents, Images, Videos, and Audio."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl


class MediaType(str, Enum):
    """Supported media types for understanding."""

    IMAGE = "image"
    DOCUMENT = "document"
    VIDEO = "video"
    AUDIO = "audio"


class ProcessingTask(str, Enum):
    """Available processing tasks."""

    DESCRIBE = "describe"
    ANALYZE = "analyze"
    EXTRACT = "extract"
    SUMMARIZE = "summarize"
    TRANSCRIBE = "transcribe"
    CAPTION = "caption"
    CLASSIFY = "classify"
    COMPARE = "compare"


class RateLimits(BaseModel):
    """Rate limiting configuration for API requests."""

    rpm: int = Field(default=15, ge=1, le=1000, description="Requests per minute")
    tpm: int = Field(default=1000000, ge=1000, le=10000000, description="Tokens per minute")
    rpd: int = Field(default=1500, ge=100, le=100000, description="Requests per day")
    max_concurrent_requests: int = Field(default=5, ge=1, le=20, description="Maximum concurrent requests")


class MediaTypeConfig(BaseModel):
    """Configuration for a specific media type."""

    enabled: bool = Field(default=True, description="Whether this media type is enabled")
    formats: List[str] = Field(default_factory=list, description="Supported MIME types")
    max_files_per_request: Optional[int] = Field(default=None, description="Maximum files per request")
    max_inline_size_mb: int = Field(default=20, description="Maximum size for inline data (MB)")
    token_cost_per_unit: int = Field(default=258, description="Base token cost per unit")


class VideoConfig(MediaTypeConfig):
    """Extended configuration for video processing."""

    max_duration_hours: float = Field(default=2.0, description="Maximum duration for default resolution")
    max_duration_hours_low_res: float = Field(default=6.0, description="Maximum duration for low resolution")
    token_cost_per_second: int = Field(default=300, description="Tokens per second (default resolution)")
    token_cost_per_second_low_res: int = Field(default=100, description="Tokens per second (low resolution)")
    default_fps: int = Field(default=1, description="Default frames per second sampling")


class AudioConfig(MediaTypeConfig):
    """Extended configuration for audio processing."""

    max_duration_hours: float = Field(default=9.5, description="Maximum audio duration")
    token_cost_per_second: int = Field(default=32, description="Tokens per second of audio")


class ApiMediaUnderstandingConfig(BaseModel):
    """Configuration for API-based media understanding."""

    api_endpoint: HttpUrl = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        description="API endpoint for media understanding",
    )
    api_key_env_var: str = Field(
        default="GEMINI_API_KEY",
        description="Environment variable name for API key",
    )
    model_name: str = Field(
        default="gemini-2.5-flash",
        description="Model name for media understanding",
    )
    max_tokens: int = Field(default=500, ge=10, le=2000, description="Maximum tokens in response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for generation")
    timeout: int = Field(default=60, ge=5, le=300, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum number of retries")
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0, description="Delay between retries in seconds")
    batch_size: int = Field(default=1, ge=1, le=5, description="Number of concurrent API calls")
    overwrite_existing: bool = Field(default=False, description="Whether to overwrite existing analysis files")
    use_files_api_threshold_mb: int = Field(default=15, description="Use Files API for files larger than this (MB)")
    rate_limits: RateLimits = Field(default_factory=RateLimits, description="Rate limiting configuration")


class MediaFile(BaseModel):
    """Represents a media file for processing."""

    file_path: Path = Field(description="Path to the media file")
    media_type: MediaType = Field(description="Type of media")
    mime_type: str = Field(description="MIME type of the file")
    file_size_mb: float = Field(description="File size in megabytes")
    uploaded_file_uri: Optional[str] = Field(default=None, description="URI if uploaded via Files API")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional file metadata")


class VideoMetadata(BaseModel):
    """Video-specific metadata."""

    duration_seconds: Optional[float] = Field(default=None, description="Video duration in seconds")
    fps: Optional[int] = Field(default=None, description="Custom frames per second")
    start_offset: Optional[str] = Field(default=None, description="Start time offset (e.g., '1250s')")
    end_offset: Optional[str] = Field(default=None, description="End time offset (e.g., '1570s')")
    resolution: Optional[str] = Field(default="default", description="Processing resolution (default/low)")


class AudioMetadata(BaseModel):
    """Audio-specific metadata."""

    duration_seconds: Optional[float] = Field(default=None, description="Audio duration in seconds")
    channels: Optional[int] = Field(default=None, description="Number of audio channels")
    sample_rate: Optional[int] = Field(default=None, description="Audio sample rate")


class ApiMediaUnderstandingResult(BaseModel):
    """Result from API-based media understanding."""

    file_path: Path = Field(description="Path to the processed media file")
    media_type: MediaType = Field(description="Type of media processed")
    task: ProcessingTask = Field(description="Processing task performed")
    content: str = Field(description="Generated understanding/analysis content")

    # Processing metadata
    processing_time: float = Field(description="Time taken for processing in seconds")
    success: bool = Field(description="Whether processing was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")

    # API metadata
    model_used: str = Field(description="Model name used for processing")
    tokens_used: Optional[int] = Field(default=None, description="Number of tokens used")
    api_cost: Optional[float] = Field(default=None, description="Estimated API cost")

    # File metadata
    file_size_mb: Optional[float] = Field(default=None, description="File size in megabytes")
    uploaded_file_uri: Optional[str] = Field(default=None, description="URI if uploaded via Files API")

    # Content-specific metadata
    extracted_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Structured extracted data")
    confidence_score: Optional[float] = Field(default=None, description="Confidence score if available")
    timestamps: Optional[List[str]] = Field(default_factory=list, description="Timestamps for video/audio content")


class ApiMediaUnderstandingBatchResult(BaseModel):
    """Result from batch processing multiple media files."""

    total_files: int = Field(description="Total number of files processed")
    successful_files: int = Field(description="Number of successfully processed files")
    failed_files: int = Field(description="Number of failed files")
    skipped_files: int = Field(description="Number of skipped files")

    results: List[ApiMediaUnderstandingResult] = Field(
        default_factory=list, description="Individual processing results"
    )

    # Batch metadata
    batch_processing_time: float = Field(description="Total time for batch processing")
    total_tokens_used: int = Field(default=0, description="Total tokens used in batch")
    total_api_cost: float = Field(default=0.0, description="Total estimated API cost")

    # Statistics by media type
    files_by_type: Dict[str, int] = Field(default_factory=dict, description="Count of files processed by media type")

    def add_result(self, result: ApiMediaUnderstandingResult) -> None:
        """Add a result to the batch results."""
        self.results.append(result)

        if result.success:
            self.successful_files += 1
        else:
            self.failed_files += 1

        if result.tokens_used:
            self.total_tokens_used += result.tokens_used

        if result.api_cost:
            self.total_api_cost += result.api_cost

        # Update type statistics
        media_type_str = result.media_type.value
        self.files_by_type[media_type_str] = self.files_by_type.get(media_type_str, 0) + 1


class ApiMediaUnderstandingPathConfig(BaseModel):
    """Configuration for media understanding paths."""

    processed_data_dir: str = Field(default="./data/processed", description="Directory for processed output")
    media_analysis_dir: str = Field(default="media_analysis", description="Directory for analysis results")


class ApiMediaUnderstandingOutputStructure(BaseModel):
    """Configuration for media understanding output directory structure."""

    analysis_results_dir: str = Field(
        default="analysis_results",
        description="Directory name for analysis results",
    )
    transcripts_dir: str = Field(
        default="transcripts",
        description="Directory name for transcripts",
    )
    summaries_dir: str = Field(
        default="summaries",
        description="Directory name for summaries",
    )
    extracted_data_dir: str = Field(
        default="extracted_data",
        description="Directory name for extracted data",
    )


class ApiMediaUnderstandingPromptConfig(BaseModel):
    """Configuration for media understanding prompts."""

    system_prompts: Dict[str, str] = Field(
        default_factory=dict,
        description="System prompts for different media types",
    )
    task_prompts: Dict[str, str] = Field(
        default_factory=dict,
        description="Task-specific prompt templates",
    )
    default_prompts: Dict[str, str] = Field(
        default_factory=dict,
        description="Default prompts for common use cases",
    )


class ApiMediaUnderstandingCompleteConfig(BaseModel):
    """Complete configuration combining all config sections."""

    api_media_understanding: ApiMediaUnderstandingConfig
    paths: ApiMediaUnderstandingPathConfig
    output_structure: ApiMediaUnderstandingOutputStructure
    prompts: ApiMediaUnderstandingPromptConfig


# Token cost calculators for different media types
class TokenCalculator:
    """Calculate token costs for different media types based on Gemini API specifications."""

    @staticmethod
    def calculate_image_tokens(width: int, height: int) -> int:
        """Calculate tokens for an image based on dimensions."""
        if width <= 384 and height <= 384:
            return 258

        # Calculate tiles for larger images
        crop_unit = max(1, min(width, height) // 2)  # Rough approximation
        tiles_x = (width + crop_unit - 1) // crop_unit
        tiles_y = (height + crop_unit - 1) // crop_unit
        return tiles_x * tiles_y * 258

    @staticmethod
    def calculate_video_tokens(duration_seconds: float, resolution: str = "default") -> int:
        """Calculate tokens for video based on duration and resolution."""
        if resolution == "low":
            return int(duration_seconds * 100)  # ~100 tokens per second
        return int(duration_seconds * 300)  # ~300 tokens per second

    @staticmethod
    def calculate_audio_tokens(duration_seconds: float) -> int:
        """Calculate tokens for audio based on duration."""
        return int(duration_seconds * 32)  # 32 tokens per second

    @staticmethod
    def calculate_document_tokens(pages: int) -> int:
        """Calculate tokens for document based on page count."""
        return pages * 258  # 258 tokens per page
