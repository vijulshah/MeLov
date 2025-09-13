"""Pydantic models for the document processing system."""

from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class FileType(str, Enum):
    """Supported file types for processing."""

    PDF = "pdf"
    IMAGE = "image"


class DetectionConfig(BaseModel):
    """Configuration for object detection."""

    model_name: str = Field(default="facebook/detr-resnet-50", description="Model name for object detection")
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for person detection",
    )
    target_label: str = Field(default="person", description="Target label to detect and crop")
    dpi_scale: int = Field(default=300, ge=72, le=600, description="DPI for PDF rendering")
    device: Optional[str] = Field(default=None, description="Device for model inference (auto-detected if None)")
    dtype: str = Field(default="float16", description="Data type for model inference")
    batch_size: int = Field(default=8, ge=1, le=32, description="Batch size for processing multiple images")
    use_fast_tokenizer: bool = Field(
        default=True,
        description="Use fast tokenizer (recommended, will be default in transformers v4.52)",
    )


class VisualizationConfig(BaseModel):
    """Configuration for visualization and annotation."""

    bounding_box_color: str = Field(default="red", description="Color for bounding boxes")
    bounding_box_width: int = Field(default=2, ge=1, le=10, description="Width of bounding box lines")
    font_name: str = Field(default="arial.ttf", description="Font file name for annotations")
    font_size: int = Field(default=16, ge=8, le=32, description="Font size for annotations")
    padding_ratio: float = Field(default=0.1, ge=0.0, le=0.5, description="Padding ratio around detected persons")


class PathConfig(BaseModel):
    """Configuration for input and output paths."""

    raw_data_dir: str = Field(default="./data/raw", description="Directory containing raw data files")
    processed_data_dir: str = Field(default="./data/processed", description="Directory for processed output")
    supported_image_extensions: List[str] = Field(
        default=["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"],
        description="Supported image file extensions",
    )


class OutputStructure(BaseModel):
    """Configuration for output directory structure."""

    detections_dir: str = Field(default="detections", description="Directory name for annotated images")
    cropped_persons_dir: str = Field(
        default="cropped_persons",
        description="Directory name for cropped person images",
    )
    original_pages_dir: str = Field(default="original_pages", description="Directory name for original images")


class ProcessingConfig(BaseModel):
    """Main configuration class for the document processing system."""

    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    output_structure: OutputStructure = Field(default_factory=OutputStructure)


class BoundingBox(BaseModel):
    """Model for bounding box coordinates."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float


class Detection(BaseModel):
    """Model for a single detection result."""

    label: str
    score: float
    box: BoundingBox


class ProcessingResult(BaseModel):
    """Model for processing results."""

    file_path: Path
    file_type: FileType
    total_detections: int
    person_detections: int
    high_confidence_persons: int
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class BatchProcessingResult(BaseModel):
    """Model for batch processing results."""

    total_files: int
    successful_files: int
    failed_files: int
    total_processing_time: float
    results: List[ProcessingResult]
