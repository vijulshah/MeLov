"""
Image processing utilities using Hugging Face models for person detection and description.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image, ImageFile

try:
    from transformers import (
        BlipForConditionalGeneration,
        BlipProcessor,
        DetrForObjectDetection,
        DetrImageProcessor,
    )

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Handle imports for both script and module execution
try:
    # When run as module
    from ..models.bio_models import ImageInfo
    from .config_manager import ConfigManager
except ImportError:
    # When run as script or imported from script
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.bio_models import ImageInfo
    from utils.config_manager import ConfigManager

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageProcessor:
    """Processes images for person detection and description using Hugging Face models."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize image processor.

        Args:
            config_manager: Configuration manager instance
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "Hugging Face transformers library is required. "
                "Install with: pip install transformers torch torchvision"
            )

        self.config = config_manager or ConfigManager()
        self.logger = self._setup_logging()
        self.image_config = self.config.get_image_processing_config()

        # Setup device
        self.device = self._setup_device()

        # Initialize models
        self.person_detector = None
        self.image_captioner = None
        self._setup_models()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            log_config = self.config.get_logging_config()
            logger.setLevel(getattr(logging, log_config.get("level", "INFO")))

            formatter = logging.Formatter(
                log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    def _setup_device(self) -> str:
        """Setup compute device for models."""
        device_config = self.image_config.get("person_detection", {}).get("device", "auto")

        if device_config == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                self.logger.info("Using CUDA for image processing")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                self.logger.info("Using MPS for image processing")
            else:
                device = "cpu"
                self.logger.info("Using CPU for image processing")
        else:
            device = device_config
            self.logger.info(f"Using specified device: {device}")

        return device

    def _setup_models(self):
        """Initialize Hugging Face models."""
        try:
            # Setup cache directory
            cache_dir = self.image_config.get("cache_dir", "local/hf_cache")
            os.environ["TRANSFORMERS_CACHE"] = cache_dir

            # Person detection model
            person_config = self.image_config.get("person_detection", {})
            person_model_name = person_config.get("model", "facebook/detr-resnet-50")

            self.logger.info(f"Loading person detection model: {person_model_name}")
            self.person_processor = DetrImageProcessor.from_pretrained(person_model_name)
            self.person_model = DetrForObjectDetection.from_pretrained(person_model_name)
            self.person_model.to(self.device)
            self.person_model.eval()

            # Image captioning model
            caption_config = self.image_config.get("image_description", {})
            caption_model_name = caption_config.get("model", "Salesforce/blip-image-captioning-large")

            self.logger.info(f"Loading image captioning model: {caption_model_name}")
            self.caption_processor = BlipProcessor.from_pretrained(caption_model_name)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(caption_model_name)
            self.caption_model.to(self.device)
            self.caption_model.eval()

            self.logger.info("All models loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise

    def detect_person(self, image_path: str) -> Tuple[bool, float]:
        """
        Detect if image contains a person.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (contains_person, confidence_score)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            inputs = self.person_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.person_model(**inputs)

            # Process results
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.person_processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

            # Check for person class (COCO class 1 is person)
            person_config = self.image_config.get("person_detection", {})
            confidence_threshold = person_config.get("confidence_threshold", 0.7)

            person_scores = results["scores"][results["labels"] == 1]

            if len(person_scores) > 0:
                max_score = person_scores.max().item()
                contains_person = max_score >= confidence_threshold
                return contains_person, max_score
            else:
                return False, 0.0

        except Exception as e:
            self.logger.error(f"Error detecting person in {image_path}: {e}")
            return False, 0.0

    def describe_image(self, image_path: str) -> str:
        """
        Generate description for image.

        Args:
            image_path: Path to image file

        Returns:
            Generated description
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            inputs = self.caption_processor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate caption
            caption_config = self.image_config.get("image_description", {})
            max_length = caption_config.get("max_length", 150)

            with torch.no_grad():
                out = self.caption_model.generate(**inputs, max_length=max_length, num_beams=3, early_stopping=True)

            description = self.caption_processor.decode(out[0], skip_special_tokens=True)
            return description

        except Exception as e:
            self.logger.error(f"Error describing image {image_path}: {e}")
            return "Error generating description"

    def process_image(
        self,
        image_path: str,
        extracted_from_pdf: bool = False,
        page_number: Optional[int] = None,
        original_filename: Optional[str] = None,
    ) -> ImageInfo:
        """
        Complete processing pipeline for an image.

        Args:
            image_path: Path to image file
            extracted_from_pdf: Whether image was extracted from PDF
            page_number: PDF page number if applicable
            original_filename: Original filename if different

        Returns:
            ImageInfo with processing results
        """
        try:
            # Get image metadata
            image = Image.open(image_path)
            file_stats = Path(image_path).stat()

            # Person detection
            contains_person, person_confidence = self.detect_person(image_path)

            # Image description (only if contains person or is standalone image)
            description = None
            description_model = None

            if contains_person or not extracted_from_pdf:
                description = self.describe_image(image_path)
                caption_config = self.image_config.get("image_description", {})
                description_model = caption_config.get("model", "Salesforce/blip-image-captioning-large")

            # Create ImageInfo
            image_info = ImageInfo(
                file_path=str(image_path),
                original_filename=original_filename,
                extracted_from_pdf=extracted_from_pdf,
                page_number=page_number,
                contains_person=contains_person,
                person_confidence=person_confidence if contains_person else None,
                description=description,
                description_model=description_model,
                width=image.width,
                height=image.height,
                file_size_bytes=file_stats.st_size,
            )

            self.logger.info(
                f"Processed image {Path(image_path).name}: "
                f"person={contains_person}, confidence={person_confidence:.3f}"
            )

            return image_info

        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            # Return minimal ImageInfo with error info
            return ImageInfo(
                file_path=str(image_path),
                original_filename=original_filename,
                extracted_from_pdf=extracted_from_pdf,
                page_number=page_number,
                contains_person=False,
                description=f"Error processing image: {str(e)}",
            )

    def process_image_list(self, image_paths: List[str]) -> List[ImageInfo]:
        """
        Process multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of ImageInfo objects
        """
        results = []

        for image_path in image_paths:
            self.logger.info(f"Processing image: {Path(image_path).name}")
            image_info = self.process_image(image_path)
            results.append(image_info)

        return results

    def is_valid_image(self, file_path: str) -> bool:
        """
        Check if file is a valid image.

        Args:
            file_path: Path to file

        Returns:
            True if valid image
        """
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def get_supported_image_formats(self) -> List[str]:
        """Get list of supported image formats."""
        return [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, "person_model"):
            del self.person_model
        if hasattr(self, "caption_model"):
            del self.caption_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
