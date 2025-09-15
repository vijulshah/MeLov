"""API-based image captioning processor for cropped person images using media understanding infrastructure."""

import logging
import time
from pathlib import Path

# Import the media understanding infrastructure
from api_media_understander import ApiMediaUnderstandingProcessor
from data_models.api_img_captioning_data_models import (
    ApiCaptioningBatchResult,
    ApiCaptioningResult,
)
from data_models.api_media_understanding_data_models import ProcessingTask
from tqdm import tqdm
from utils.api_img_captioning_utils import CroppedImageDiscovery
from utils.api_media_understanding_utils import MediaFileDetector


class ApiImageCaptioningProcessor:
    """Main processor for API-based image captioning using media understanding infrastructure."""

    def __init__(
        self,
        config_path: str = "./data_processing/configs/api_media_understanding_config.yaml",
    ):
        # Use the media understanding processor as the underlying engine
        self.media_processor = ApiMediaUnderstandingProcessor(config_path)

        # Set up image captioning specific logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up logging configuration for image captioning."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/api_image_captioning.log"),
                logging.StreamHandler(),
            ],
        )

    def process_single_image(self, image_path: Path) -> ApiCaptioningResult:
        """Process a single cropped image for captioning synchronously."""
        # Convert to MediaFile format using MediaFileDetector
        media_file = MediaFileDetector.get_file_info(image_path)
        if not media_file:
            return ApiCaptioningResult(
                image_path=image_path,
                caption=None,
                caption_path=None,
                processing_time=0.0,
                success=False,
                error_message=f"Unable to detect media type for {image_path}",
                retry_count=0,
            )

        # Use media understanding with specific prompt for person description
        custom_prompt = "Describe this person in detail, focusing on their appearance, clothing, and any visible characteristics. Keep the description under 50 words."

        result = self.media_processor.understander.understand_media(
            media_file,
            task=ProcessingTask.DESCRIBE,
            custom_prompt=custom_prompt,
        )

        # Convert to captioning result format
        caption_path = image_path.with_suffix(".txt")

        if result.success and result.content:
            # Save caption to file
            try:
                with open(caption_path, "w", encoding="utf-8") as f:
                    f.write(result.content)
                caption_saved = True
            except Exception as e:
                logging.error(f"Failed to save caption to {caption_path}: {str(e)}")
                caption_saved = False
        else:
            caption_saved = False

        return ApiCaptioningResult(
            image_path=image_path,
            caption=result.content if result.success else None,
            caption_path=caption_path if caption_saved else None,
            processing_time=result.processing_time,
            success=result.success and caption_saved,
            error_message=result.error_message,
            retry_count=0,
        )

    def process_batch(self, base_path: Path = None) -> ApiCaptioningBatchResult:
        """Process all cropped images in the processed data directory."""
        start_time = time.time()

        if base_path is None:
            base_path = Path(self.media_processor.config.paths.processed_data_dir)

        if not base_path.exists():
            logging.error(f"Base path does not exist: {base_path}")
            return self._create_empty_result()

        # Find all cropped images using the discovery utility
        image_paths = CroppedImageDiscovery.find_cropped_images(
            base_path,
            type(
                "Config",
                (),
                {
                    "output_structure": type(
                        "OutputStructure",
                        (),
                        {"cropped_persons_dir": "cropped_persons"},
                    )()
                },
            )(),
        )

        if not image_paths:
            logging.warning(f"No cropped images found in {base_path}")
            return self._create_empty_result()

        # Filter images that need processing
        images_to_process = []
        skipped_images = 0

        for image_path in image_paths:
            caption_path = image_path.with_suffix(".txts")
            if not caption_path.exists():  # Only process if caption doesn't exist
                images_to_process.append(image_path)
            else:
                skipped_images += 1

        if not images_to_process:
            logging.info(f"All {len(image_paths)} images already have captions, nothing to process")
            return self._create_empty_result()

        total_images = len(image_paths)
        processed_images = 0
        failed_images = 0
        results = []

        logging.info(f"Starting to process {len(images_to_process)} images using media understanding API...")
        logging.info(f"API endpoint: {self.media_processor.config.api_media_understanding.api_endpoint}")
        logging.info(f"Model: {self.media_processor.config.api_media_understanding.model_name}")

        # Process each image with progress bar
        progress_bar = tqdm(
            images_to_process,
            desc="Processing images",
            unit="img",
            total=len(images_to_process),
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {desc}",
        )

        for image_path in progress_bar:
            try:
                # Update progress bar description with current image name
                progress_bar.set_description(f"Processing {image_path.name}")

                # Process the image
                result = self.process_single_image(image_path)
                results.append(result)

                if result.success:
                    processed_images += 1
                    logging.debug(f"Generated caption: {result.caption}")
                else:
                    failed_images += 1
                    logging.error(f"Failed: {image_path.name} - {result.error_message}")

                progress_bar.set_postfix(
                    {
                        "Processed": processed_images,
                        "Skipped": skipped_images,
                        "Failed": failed_images,
                    }
                )

                # Small delay between requests to respect rate limits
                time.sleep(0.1)

            except Exception as e:
                failed_images += 1
                progress_bar.set_postfix(
                    {
                        "Processed": processed_images,
                        "Skipped": skipped_images,
                        "Failed": failed_images,
                    }
                )
                logging.error(f"Error processing {image_path}: {str(e)}")

                # Create error result
                error_result = ApiCaptioningResult(
                    image_path=image_path,
                    processing_time=0.0,
                    success=False,
                    error_message=str(e),
                )
                results.append(error_result)

        # Close the progress bar
        progress_bar.close()

        total_processing_time = time.time() - start_time

        # Create and return batch result
        batch_result = ApiCaptioningBatchResult(
            total_images=total_images,
            processed_images=processed_images,
            skipped_images=skipped_images,
            failed_images=failed_images,
            total_processing_time=total_processing_time,
            total_api_calls=len(images_to_process),
            total_api_cost=None,
            results=results,
        )

        self._print_batch_summary(batch_result)
        return batch_result

    def _create_empty_result(self) -> ApiCaptioningBatchResult:
        """Create an empty batch result."""
        return ApiCaptioningBatchResult(
            total_images=0,
            processed_images=0,
            skipped_images=0,
            failed_images=0,
            total_processing_time=0.0,
            total_api_calls=0,
            total_api_cost=0.0,
            results=[],
        )

    def _print_batch_summary(self, batch_result: ApiCaptioningBatchResult) -> None:
        """Print batch processing summary."""
        logging.info(f"\n{'='*60}")
        logging.info("API IMAGE CAPTIONING COMPLETED")
        logging.info(f"{'='*60}")
        logging.info(f"Total images: {batch_result.total_images}")
        logging.info(f"Successfully processed: {batch_result.processed_images}")
        logging.info(f"Skipped (already exist): {batch_result.skipped_images}")
        logging.info(f"Failed: {batch_result.failed_images}")
        logging.info(f"Total processing time: {batch_result.total_processing_time:.2f}s")
        logging.info(f"Total API calls: {batch_result.total_api_calls}")

        if batch_result.total_images > 0:
            avg_time = batch_result.total_processing_time / batch_result.total_images
            logging.info(f"Average time per image: {avg_time:.2f}s")

        logging.info(f"\nResults saved in: {self.media_processor.config.paths.processed_data_dir}")
        logging.info(f"{'='*60}")


def main():
    """Main entry point for the API-based image captioning system."""
    try:
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)

        # Initialize processor (now using media understanding config)
        processor = ApiImageCaptioningProcessor()

        # Display rate limiting information
        rate_limits = processor.media_processor.config.api_media_understanding.rate_limits
        print(f"\n=== Rate Limiting Configuration ===")
        print(f"Requests per minute (RPM): {rate_limits.rpm}")
        print(f"Tokens per minute (TPM): {rate_limits.tpm:,}")
        print(f"Requests per day (RPD): {rate_limits.rpd}")
        print(f"Max concurrent requests: {rate_limits.max_concurrent_requests}")

        print("\n=== API Image Captioning Processing ===")
        print("Processing cropped person images for captioning...")

        # Process all cropped images
        result = processor.process_batch()

        if result.processed_images > 0:
            print(f"\nImage captioning completed: {result.processed_images} images processed successfully")
        else:
            print("\nNo images were processed")

    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
