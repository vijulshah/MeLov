"""Utility functions for the image captioning system."""

import logging
import os
import time
from pathlib import Path
from typing import List

import torch
import yaml
from captioning_models import CaptioningProcessingConfig, CaptioningResult
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


class CaptioningConfigManager:
    """Manages captioning configuration loading and validation."""

    @staticmethod
    def load_config(config_path: str) -> CaptioningProcessingConfig:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            return CaptioningProcessingConfig(**config_data)
        else:
            # Return default configuration if file doesn't exist
            config = CaptioningProcessingConfig()
            CaptioningConfigManager.save_config(config, config_path)
            return config

    @staticmethod
    def save_config(config: CaptioningProcessingConfig, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config.model_dump(), f, default_flow_style=False, indent=2)


class ImageCaptioner:
    """Manages image captioning using BLIP model."""

    def __init__(self, config: CaptioningProcessingConfig):
        self.config = config
        self.processor = None
        self.model = None
        self.device = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the BLIP model and processor."""
        try:
            logging.info(f"Initializing BLIP model: {self.config.captioning.model_name}")

            # Set device
            self.device = self._get_device()
            logging.info(f"Using device: {self.device}")

            # Load processor and model
            self.processor = BlipProcessor.from_pretrained(self.config.captioning.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.config.captioning.model_name)
            self.model.to(self.device)

            logging.info("BLIP model and processor loaded successfully")

        except Exception as e:
            logging.error(f"Failed to initialize BLIP model: {str(e)}")
            raise

    def _get_device(self) -> str:
        """Get the appropriate device for inference."""
        if self.config.captioning.device:
            return self.config.captioning.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def generate_caption(self, image_path: Path) -> CaptioningResult:
        """Generate a caption for a single image."""
        start_time = time.time()

        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")

            # Process image for the model
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            # Generate caption
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_length=self.config.captioning.max_length,
                    num_beams=self.config.captioning.num_beams,
                )

            # Decode the generated caption
            caption = self.processor.decode(out[0], skip_special_tokens=True)

            # Create caption file path
            caption_path = image_path.with_suffix(".txt")

            # Save caption
            success = self._save_caption(caption_path, caption)

            processing_time = time.time() - start_time

            return CaptioningResult(
                image_path=image_path,
                caption=caption if success else None,
                caption_path=caption_path if success else None,
                processing_time=processing_time,
                success=success,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logging.error(f"Failed to generate caption for {image_path}: {str(e)}")

            return CaptioningResult(
                image_path=image_path,
                processing_time=processing_time,
                success=False,
                error_message=str(e),
            )

    def _save_caption(self, caption_path: Path, caption: str) -> bool:
        """Save caption as a text file."""
        try:
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption)
            return True
        except Exception as e:
            logging.error(f"Failed to save caption to {caption_path}: {str(e)}")
            return False

    def should_process_image(self, image_path: Path) -> bool:
        """Check if image should be processed based on configuration."""
        caption_path = image_path.with_suffix(".txt")
        return self.config.captioning.overwrite_existing or not caption_path.exists()


class CroppedImageDiscovery:
    """Handles discovery of cropped person images."""

    @staticmethod
    def find_cropped_images(base_path: Path, config: CaptioningProcessingConfig) -> List[Path]:
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
