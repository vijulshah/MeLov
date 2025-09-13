"""Pydantic models for the image captioning system."""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class CaptioningConfig(BaseModel):
    """Configuration for image captioning."""

    model_name: str = Field(
        default="Salesforce/blip-image-captioning-base",
        description="Model name for image captioning",
    )
    max_length: int = Field(default=50, ge=10, le=200, description="Maximum caption length")
    num_beams: int = Field(default=5, ge=1, le=10, description="Number of beams for generation")
    device: Optional[str] = Field(default=None, description="Device for model inference (auto-detected if None)")
    batch_size: int = Field(default=4, ge=1, le=16, description="Batch size for captioning")
    overwrite_existing: bool = Field(default=False, description="Whether to overwrite existing caption files")
    use_fast_processor: bool = Field(
        default=True,
        description="Use fast tokenizer (recommended, will be default in transformers v4.52)",
    )


class CaptioningPathConfig(BaseModel):
    """Configuration for captioning paths."""

    processed_data_dir: str = Field(default="./data/processed", description="Directory for processed output")


class CaptioningOutputStructure(BaseModel):
    """Configuration for captioning output directory structure."""

    cropped_persons_dir: str = Field(
        default="cropped_persons",
        description="Directory name for cropped person images",
    )


class CaptioningProcessingConfig(BaseModel):
    """Main configuration class for the image captioning system."""

    captioning: CaptioningConfig = Field(default_factory=CaptioningConfig)
    paths: CaptioningPathConfig = Field(default_factory=CaptioningPathConfig)
    output_structure: CaptioningOutputStructure = Field(default_factory=CaptioningOutputStructure)


class CaptioningResult(BaseModel):
    """Model for image captioning results."""

    image_path: Path
    caption: Optional[str] = None
    caption_path: Optional[Path] = None
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class CaptioningBatchResult(BaseModel):
    """Model for batch captioning results."""

    total_images: int
    processed_images: int
    skipped_images: int
    failed_images: int
    total_processing_time: float
    results: List[CaptioningResult]
