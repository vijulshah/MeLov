"""Utility functions for the document processing system."""

from pathlib import Path
from typing import List, Optional, Tuple

import fitz
import torch
import yaml
from data_models.obj_detection_data_models import (
    BoundingBox,
    Detection,
    ProcessingConfig,
)
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline


class ConfigManager:
    """Manages configuration loading and validation."""

    @staticmethod
    def load_config(config_path: str) -> ProcessingConfig:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            return ProcessingConfig(**config_data)
        else:
            # Return default configuration if file doesn't exist
            config = ProcessingConfig()
            ConfigManager.save_config(config, config_path)
            return config

    @staticmethod
    def save_config(config: ProcessingConfig, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config.model_dump(), f, default_flow_style=False, indent=2)


class FileDiscovery:
    """Handles file discovery and classification."""

    @staticmethod
    def find_files(base_dir: Path, extensions: List[str]) -> List[Path]:
        """Find all files with given extensions recursively."""
        files = []
        for ext in extensions:
            files.extend(base_dir.glob(f"**/*.{ext}"))
            files.extend(base_dir.glob(f"**/*.{ext.upper()}"))
        return files

    @staticmethod
    def discover_all_files(config: ProcessingConfig) -> Tuple[List[Path], List[Path]]:
        """Discover all PDF and image files based on configuration."""
        raw_dir = Path(config.paths.raw_data_dir)

        pdf_files = FileDiscovery.find_files(raw_dir, ["pdf"])
        image_files = FileDiscovery.find_files(raw_dir, config.paths.supported_image_extensions)

        return pdf_files, image_files


class DirectoryManager:
    """Manages directory creation and organization."""

    @staticmethod
    def create_output_structure(base_path: Path, output_config: ProcessingConfig) -> Tuple[Path, Path, Path]:
        """Create output directory structure and return paths."""
        detections_dir = base_path / output_config.output_structure.detections_dir
        cropped_dir = base_path / output_config.output_structure.cropped_persons_dir
        original_dir = base_path / output_config.output_structure.original_pages_dir

        detections_dir.mkdir(parents=True, exist_ok=True)
        cropped_dir.mkdir(parents=True, exist_ok=True)
        original_dir.mkdir(parents=True, exist_ok=True)

        return detections_dir, cropped_dir, original_dir

    @staticmethod
    def get_output_base_path(file_path: Path, config: ProcessingConfig) -> Path:
        """Get base output path for a given file."""
        file_name = file_path.stem
        relative_path = file_path.relative_to(config.paths.raw_data_dir)
        relative_dir = relative_path.parent

        processed_dir = Path(config.paths.processed_data_dir)
        return processed_dir / relative_dir / file_name


class ImageProcessor:
    """Handles image processing operations."""

    @staticmethod
    def load_image(image_path: Path) -> Optional[Image.Image]:
        """Load and convert image to RGB format."""
        try:
            # Temporarily increase PIL's decompression bomb limit for loading images
            from PIL import Image as PILImage

            original_max_pixels = PILImage.MAX_IMAGE_PIXELS
            try:
                PILImage.MAX_IMAGE_PIXELS = 100_000_000
                image = Image.open(image_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Check if image is reasonable size
                pixels = image.width * image.height
                if pixels > 80_000_000:
                    print(f"    Warning: Large image detected ({pixels:,} pixels), consider reducing resolution")

                return image
            finally:
                PILImage.MAX_IMAGE_PIXELS = original_max_pixels
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    @staticmethod
    def pdf_page_to_image(page, dpi: int = 300) -> Image.Image:
        """Convert PDF page to PIL Image with adaptive DPI to prevent decompression bomb warnings."""
        # Get page dimensions in points (1 point = 1/72 inch)
        page_rect = page.rect
        page_width_points = page_rect.width
        page_height_points = page_rect.height

        # Calculate estimated pixel dimensions at given DPI
        estimated_width = int(page_width_points * dpi / 72)
        estimated_height = int(page_height_points * dpi / 72)
        estimated_pixels = estimated_width * estimated_height

        # PIL's default limit is around 89 million pixels
        # Set our limit to 88 million to be safe
        max_pixels = 88_000_000

        # If estimated size exceeds limit, reduce DPI
        if estimated_pixels > max_pixels:
            # Calculate the scale factor needed
            scale_factor = (max_pixels / estimated_pixels) ** 0.5
            adjusted_dpi = int(dpi * scale_factor)
            print(f"    Large page detected ({estimated_pixels:,} pixels), reducing DPI from {dpi} to {adjusted_dpi}")
            dpi = max(72, adjusted_dpi)  # Don't go below 72 DPI

        # Convert page to image
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("ppm")

        # Temporarily increase PIL's decompression bomb limit for this operation
        from PIL import Image as PILImage

        original_max_pixels = PILImage.MAX_IMAGE_PIXELS
        try:
            # Set a reasonable limit (100M pixels)
            PILImage.MAX_IMAGE_PIXELS = 100_000_000
            image = Image.open(fitz.io.BytesIO(img_data))

            # Ensure image is in RGB mode for consistency
            if image.mode != "RGB":
                image = image.convert("RGB")

            return image
        finally:
            # Restore original limit
            PILImage.MAX_IMAGE_PIXELS = original_max_pixels

    @staticmethod
    def crop_person(image: Image.Image, detection: Detection, padding_ratio: float = 0.1) -> Image.Image:
        """Crop person from image with padding."""
        box = detection.box

        # Calculate padding
        box_width = box.xmax - box.xmin
        box_height = box.ymax - box.ymin
        padding_x = int(box_width * padding_ratio)
        padding_y = int(box_height * padding_ratio)

        # Calculate crop coordinates with padding
        crop_xmin = max(0, int(box.xmin - padding_x))
        crop_ymin = max(0, int(box.ymin - padding_y))
        crop_xmax = min(image.width, int(box.xmax + padding_x))
        crop_ymax = min(image.height, int(box.ymax + padding_y))

        return image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))


class DetectionVisualizer:
    """Handles detection visualization and annotation."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.font = self._load_font()

    def _load_font(self) -> ImageFont.FreeTypeFont:
        """Load font for annotations."""
        try:
            return ImageFont.truetype(self.config.visualization.font_name, self.config.visualization.font_size)
        except:
            return ImageFont.load_default()

    def annotate_image(self, image: Image.Image, detections: List[Detection]) -> Image.Image:
        """Annotate image with detection bounding boxes and labels."""
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        for detection in detections:
            self._draw_detection(draw, detection)

        return annotated_image

    def _draw_detection(self, draw: ImageDraw.Draw, detection: Detection) -> None:
        """Draw a single detection on the image."""
        box = detection.box
        color = self.config.visualization.bounding_box_color
        width = self.config.visualization.bounding_box_width

        # Draw bounding box
        draw.rectangle([box.xmin, box.ymin, box.xmax, box.ymax], outline=color, width=width)

        # Draw label with confidence score
        text = f"{detection.label}: {detection.score:.2f}"
        bbox = draw.textbbox((box.xmin, box.ymin), text, font=self.font)

        # Draw background for text
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill=color)

        # Draw text
        draw.text((box.xmin, box.ymin), text, fill="white", font=self.font)


class ModelManager:
    """Manages object detection model initialization and inference."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.pipeline = self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the object detection pipeline."""
        device = self._get_device()
        dtype = getattr(torch, self.config.detection.dtype)

        return pipeline(
            "object-detection",
            model=self.config.detection.model_name,
            dtype=dtype,
            device_map=device,
            batch_size=self.config.detection.batch_size,
        )

    def _get_device(self) -> str:
        """Get the appropriate device for inference."""
        if self.config.detection.device:
            return self.config.detection.device
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def detect_objects(self, image: Image.Image) -> List[Detection]:
        """Perform object detection on an image."""
        raw_detections = self.pipeline(image)

        detections = []
        for detection in raw_detections:
            bbox = BoundingBox(**detection["box"])
            det = Detection(label=detection["label"], score=detection["score"], box=bbox)
            detections.append(det)

        return detections

    def detect_objects_batch(self, images: List[Image.Image]) -> List[List[Detection]]:
        """Perform object detection on a batch of images for better GPU utilization."""
        if not images:
            return []

        # Check if all images have the same dimensions
        if len(images) == 1:
            # Single image, process normally
            raw_detections_batch = self.pipeline(images)
        else:
            # Multiple images - check dimensions
            first_size = images[0].size
            all_same_size = all(img.size == first_size for img in images)

            if all_same_size:
                # All images same size, can process as batch
                try:
                    raw_detections_batch = self.pipeline(images)
                except Exception as e:
                    if "tensor" in str(e).lower() and "size" in str(e).lower():
                        # Fallback to individual processing if tensor size error occurs
                        print(
                            f"    Batch processing failed due to tensor size mismatch, falling back to individual processing..."
                        )
                        return self._process_images_individually(images)
                    else:
                        raise e
            else:
                # Different sizes, process individually for safety
                print(f"    Images have different dimensions, processing individually for stability...")
                return self._process_images_individually(images)

        batch_results = []
        for raw_detections in raw_detections_batch:
            detections = []
            for detection in raw_detections:
                bbox = BoundingBox(**detection["box"])
                det = Detection(label=detection["label"], score=detection["score"], box=bbox)
                detections.append(det)
            batch_results.append(detections)

        return batch_results

    def _process_images_individually(self, images: List[Image.Image]) -> List[List[Detection]]:
        """Process images individually when batch processing fails."""
        batch_results = []
        for i, image in enumerate(images):
            try:
                detections = self.detect_objects(image)
                batch_results.append(detections)
            except Exception as e:
                print(f"    Error processing image {i+1}/{len(images)}: {e}")
                # Return empty detections for failed image
                batch_results.append([])
        return batch_results

    def filter_high_confidence_persons(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections for high-confidence person detections."""
        return [
            det
            for det in detections
            if (
                det.label.lower() == self.config.detection.target_label.lower()
                and det.score > self.config.detection.confidence_threshold
            )
        ]
