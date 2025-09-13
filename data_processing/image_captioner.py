"""Image captioning processor for cropped person images using Salesforce BLIP model."""

import logging
import time
from pathlib import Path

from data_models.img_captioning_data_models import (
    CaptioningBatchResult,
    CaptioningResult,
)
from utils.img_captioning_utils import (
    CaptioningConfigManager,
    CroppedImageDiscovery,
    ImageCaptioner,
)


class ImageCaptioningProcessor:
    """Main processor for image captioning with BLIP model."""

    def __init__(self, config_path: str = "./data_processing/configs/img_captioning_config.yaml"):
        self.config = CaptioningConfigManager.load_config(config_path)
        self.captioner = ImageCaptioner(self.config)

        # Set up logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/image_captioning.log"),
                logging.StreamHandler(),
            ],
        )

    def process_single_image(self, image_path: Path) -> CaptioningResult:
        """Process a single cropped image for captioning."""
        return self.captioner.generate_caption(image_path)

    def process_batch(self, base_path: Path = None) -> CaptioningBatchResult:
        """Process all cropped images in the processed data directory."""
        start_time = time.time()

        if base_path is None:
            base_path = Path(self.config.paths.processed_data_dir)

        if not base_path.exists():
            logging.error(f"Base path does not exist: {base_path}")
            return self._create_empty_result()

        # Find all cropped images
        image_paths = CroppedImageDiscovery.find_cropped_images(base_path, self.config)

        if not image_paths:
            logging.warning(f"No cropped images found in {base_path}")
            return self._create_empty_result()

        # Initialize counters
        total_images = len(image_paths)
        processed_images = 0
        skipped_images = 0
        failed_images = 0
        results = []

        logging.info(f"Starting to process {total_images} images...")

        # Process each image
        for i, image_path in enumerate(image_paths, 1):
            try:
                # Check if we should process this image
                if not self.captioner.should_process_image(image_path):
                    skipped_images += 1
                    logging.debug(f"Caption already exists for {image_path.name}, skipping")
                    continue

                # Process the image
                result = self.process_single_image(image_path)
                results.append(result)

                if result.success:
                    processed_images += 1
                    logging.info(f"[{i}/{total_images}] Processed: {image_path.name}")
                    logging.debug(f"Generated caption: {result.caption}")
                else:
                    failed_images += 1
                    logging.error(f"[{i}/{total_images}] Failed: {image_path.name} - {result.error_message}")

                # Progress update every 10 images
                if i % 10 == 0:
                    self._log_progress(i, total_images, start_time)

            except Exception as e:
                failed_images += 1
                logging.error(f"Error processing {image_path}: {str(e)}")

                # Create error result
                error_result = CaptioningResult(
                    image_path=image_path,
                    processing_time=0.0,
                    success=False,
                    error_message=str(e),
                )
                results.append(error_result)

        total_processing_time = time.time() - start_time

        # Create and return batch result
        batch_result = CaptioningBatchResult(
            total_images=total_images,
            processed_images=processed_images,
            skipped_images=skipped_images,
            failed_images=failed_images,
            total_processing_time=total_processing_time,
            results=results,
        )

        self._print_batch_summary(batch_result)
        return batch_result

    def _create_empty_result(self) -> CaptioningBatchResult:
        """Create an empty batch result."""
        return CaptioningBatchResult(
            total_images=0,
            processed_images=0,
            skipped_images=0,
            failed_images=0,
            total_processing_time=0.0,
            results=[],
        )

    def _log_progress(self, current: int, total: int, start_time: float) -> None:
        """Log processing progress."""
        elapsed = time.time() - start_time
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0

        logging.info(
            f"Progress: {current}/{total} ({current/total*100:.1f}%) - " f"Rate: {rate:.1f} img/s - ETA: {eta:.0f}s"
        )

    def _print_batch_summary(self, batch_result: CaptioningBatchResult) -> None:
        """Print batch processing summary."""
        logging.info(f"\n{'='*60}")
        logging.info("IMAGE CAPTIONING COMPLETED")
        logging.info(f"{'='*60}")
        logging.info(f"Total images: {batch_result.total_images}")
        logging.info(f"Successfully processed: {batch_result.processed_images}")
        logging.info(f"Skipped (already exist): {batch_result.skipped_images}")
        logging.info(f"Failed: {batch_result.failed_images}")
        logging.info(f"Total processing time: {batch_result.total_processing_time:.2f}s")

        if batch_result.total_images > 0:
            avg_time = batch_result.total_processing_time / batch_result.total_images
            logging.info(f"Average time per image: {avg_time:.2f}s")

        logging.info(f"Results saved in: {self.config.paths.processed_data_dir}")
        logging.info(f"{'='*60}")


def main():
    """Main entry point for the image captioning system."""
    try:
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)

        # Initialize processor
        processor = ImageCaptioningProcessor()

        # Process all images
        processor.process_batch()

    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
