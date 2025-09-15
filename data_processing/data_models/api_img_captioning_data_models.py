"""Pydantic models for the API-based image captioning system."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl


class RateLimits(BaseModel):
    """Rate limiting configuration for API requests."""

    rpm: int = Field(default=15, ge=1, le=1000, description="Requests per minute")
    tpm: int = Field(default=1000000, ge=1000, le=10000000, description="Tokens per minute")
    rpd: int = Field(default=1500, ge=100, le=100000, description="Requests per day")
    max_concurrent_requests: int = Field(default=5, ge=1, le=20, description="Maximum concurrent requests")


class ApiCaptioningConfig(BaseModel):
    """Configuration for API-based image captioning."""

    api_endpoint: HttpUrl = Field(
        default="https://api.openai.com/v1/chat/completions",
        description="API endpoint for image captioning",
    )
    api_key_env_var: str = Field(
        default="GEMINI_API_KEY",
        description="Environment variable name for API key",
    )
    model_name: str = Field(
        default="gemini-2.5-flash",
        description="Model name for image captioning",
    )
    max_tokens: int = Field(default=150, ge=10, le=500, description="Maximum tokens in response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for generation")
    timeout: int = Field(default=30, ge=5, le=120, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum number of retries")
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0, description="Delay between retries in seconds")
    overwrite_existing: bool = Field(default=False, description="Whether to overwrite existing caption files")
    batch_size: int = Field(default=1, ge=1, le=5, description="Number of concurrent API calls")
    rate_limits: RateLimits = Field(default_factory=RateLimits, description="Rate limiting configuration")


class ApiCaptioningPathConfig(BaseModel):
    """Configuration for API captioning paths."""

    processed_data_dir: str = Field(default="./data/processed", description="Directory for processed output")


class ApiCaptioningOutputStructure(BaseModel):
    """Configuration for API captioning output directory structure."""

    cropped_persons_dir: str = Field(
        default="cropped_persons",
        description="Directory name for cropped person images",
    )


class ApiCaptioningProcessingConfig(BaseModel):
    """Main configuration class for the API-based image captioning system."""

    api_captioning: ApiCaptioningConfig = Field(default_factory=ApiCaptioningConfig)
    paths: ApiCaptioningPathConfig = Field(default_factory=ApiCaptioningPathConfig)
    output_structure: ApiCaptioningOutputStructure = Field(default_factory=ApiCaptioningOutputStructure)


class ApiCaptioningPromptConfig(BaseModel):
    """Configuration for API captioning prompts."""

    system_prompt: str = Field(
        default="You are an expert at describing people in images. Provide concise, accurate descriptions focusing on visible characteristics.",
        description="System prompt for the API",
    )
    user_prompt_template: str = Field(
        default="Describe this person in detail, focusing on their appearance, clothing, and any visible characteristics. Keep the description under 50 words.",
        description="User prompt template for image captioning",
    )


class ApiRequestPayload(BaseModel):
    """Model for API request payload."""

    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int
    temperature: float


class ApiResponse(BaseModel):
    """Model for API response."""

    id: Optional[str] = None
    object: Optional[str] = None
    created: Optional[int] = None
    model: Optional[str] = None
    choices: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None
    error: Optional[Dict[str, Any]] = None


class ApiCaptioningResult(BaseModel):
    """Model for API-based image captioning results."""

    image_path: Path
    caption: Optional[str] = None
    caption_path: Optional[Path] = None
    processing_time: float
    api_response: Optional[ApiResponse] = None
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0


class ApiCaptioningBatchResult(BaseModel):
    """Model for batch API captioning results."""

    total_images: int
    processed_images: int
    skipped_images: int
    failed_images: int
    total_processing_time: float
    total_api_calls: int
    total_api_cost: Optional[float] = None
    results: List[ApiCaptioningResult]


class ApiUsageStats(BaseModel):
    """Model for tracking API usage statistics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
