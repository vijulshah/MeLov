"""API-based media understanding processor for analyzing Documents, Images, Videos, and Audio."""

import logging
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from data_models.api_media_understanding_data_models import (
    ApiMediaUnderstandingBatchResult,
    ApiMediaUnderstandingResult,
    MediaType,
    ProcessingTask,
    VideoMetadata,
)
from tqdm import tqdm
from utils.api_media_understanding_utils import (
    ApiMediaUnderstander,
    ApiMediaUnderstandingConfigManager,
    MediaFileDetector,
)


class RateLimiter:
    """Rate limiter to respect API limits (RPM, TPM, RPD)."""

    def __init__(self, rpm: int = 15, tpm: int = 1000000, rpd: int = 1500):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd

        # Track requests per minute
        self.request_times = deque()

        # Track tokens per minute
        self.token_times = deque()

        # Track requests per day
        self.daily_request_count = 0
        self.last_reset_date = datetime.now().date()

    def wait_if_needed(self, estimated_tokens: int = 100) -> None:
        """Wait if necessary to respect rate limits."""
        current_time = datetime.now()

        # Reset daily counter if needed
        if current_time.date() > self.last_reset_date:
            self.daily_request_count = 0
            self.last_reset_date = current_time.date()

        # Check RPD limit
        if self.daily_request_count >= self.rpd:
            logging.warning(f"Daily request limit ({self.rpd}) reached. Blocking further requests.")
            raise Exception(f"Daily request limit ({self.rpd}) exceeded")

        # Clean old request times (older than 1 minute)
        minute_ago = current_time - timedelta(minutes=1)
        while self.request_times and self.request_times[0] < minute_ago:
            self.request_times.popleft()

        # Clean old token times (older than 1 minute)
        while self.token_times and self.token_times[0][0] < minute_ago:
            self.token_times.popleft()

        # Check RPM limit
        if len(self.request_times) >= self.rpm:
            sleep_time = (self.request_times[0] + timedelta(minutes=1) - current_time).total_seconds()
            if sleep_time > 0:
                logging.info(f"Rate limiting: waiting {sleep_time:.2f} seconds (RPM limit: {self.rpm})")
                time.sleep(sleep_time)

        # Check TPM limit
        current_tokens = sum(tokens for _, tokens in self.token_times)
        if current_tokens + estimated_tokens > self.tpm:
            # Wait until oldest token request expires
            if self.token_times:
                sleep_time = (self.token_times[0][0] + timedelta(minutes=1) - current_time).total_seconds()
                if sleep_time > 0:
                    logging.info(f"Rate limiting: waiting {sleep_time:.2f} seconds (TPM limit: {self.tpm})")
                    time.sleep(sleep_time)

    def record_request(self, tokens_used: int = 100) -> None:
        """Record a request and tokens used."""
        current_time = datetime.now()
        self.request_times.append(current_time)
        self.token_times.append((current_time, tokens_used))
        self.daily_request_count += 1


class ApiMediaUnderstandingProcessor:
    """Main processor for API-based media understanding supporting Documents, Images, Videos, and Audio."""

    def __init__(
        self,
        config_path: str = "./data_processing/configs/api_media_understanding_config.yaml",
    ):
        self.config = ApiMediaUnderstandingConfigManager.load_config(config_path)
        self.prompt_config = ApiMediaUnderstandingConfigManager.load_prompt_config(config_path)
        self.understander = ApiMediaUnderstander(self.config, self.prompt_config)

        # Initialize rate limiter with config values
        self.rate_limiter = RateLimiter(
            rpm=self.config.api_media_understanding.rate_limits.rpm,
            tpm=self.config.api_media_understanding.rate_limits.tpm,
            rpd=self.config.api_media_understanding.rate_limits.rpd,
        )

        # Set up logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/api_media_understanding.log"),
                logging.StreamHandler(),
            ],
        )

    def process_single_media(
        self,
        media_path: Path,
        task: ProcessingTask = ProcessingTask.DESCRIBE,
        custom_prompt: Optional[str] = None,
        video_metadata: Optional[VideoMetadata] = None,
    ) -> ApiMediaUnderstandingResult:
        """Process a single media file for understanding."""
        # Get media file info
        media_file = MediaFileDetector.get_file_info(media_path)
        if not media_file:
            raise ValueError(f"Unsupported media file: {media_path}")

        # Apply rate limiting before making the request
        estimated_tokens = 200  # Conservative estimate for media + prompt tokens
        self.rate_limiter.wait_if_needed(estimated_tokens)

        result = self.understander.understand_media(
            media_file,
            task=task,
            custom_prompt=custom_prompt,
            video_metadata=video_metadata,
        )

        # Record the request for rate limiting
        actual_tokens = result.tokens_used or estimated_tokens
        self.rate_limiter.record_request(actual_tokens)

        # Save result if successful
        if result.success:
            self.understander.save_result(result)

        return result

    def process_batch(
        self,
        base_path: Path = None,
        media_types: Optional[List[MediaType]] = None,
        task: ProcessingTask = ProcessingTask.DESCRIBE,
        max_files: Optional[int] = None,
    ) -> ApiMediaUnderstandingBatchResult:
        """Process all supported media files in a directory."""
        start_time = time.time()

        if base_path is None:
            base_path = Path(self.config.paths.processed_data_dir)

        if not base_path.exists():
            logging.error(f"Base path does not exist: {base_path}")
            return self._create_empty_result()

        # Find all media files
        media_files = MediaFileDetector.find_media_files(base_path, media_types=media_types, max_files=max_files)

        if not media_files:
            logging.warning(f"No supported media files found in {base_path}")
            return self._create_empty_result()

        # Filter files that need processing
        files_to_process = []
        skipped_files = 0

        for media_file in media_files:
            if self.understander.should_process_media(media_file, task):
                files_to_process.append(media_file)
            else:
                skipped_files += 1

        if not files_to_process:
            logging.info("All media files already processed, nothing to do")
            return self._create_empty_result()

        total_files = len(media_files)
        logging.info(f"Starting batch processing of {len(files_to_process)} media files")
        logging.info(f"API endpoint: {self.config.api_media_understanding.api_endpoint}")
        logging.info(f"Model: {self.config.api_media_understanding.model_name}")

        # Process files
        processed_files = 0
        failed_files = 0
        results = []

        # Create progress bar
        progress_bar = tqdm(
            total=len(files_to_process),
            desc="Processing media files",
            unit="file",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {desc}",
        )

        for media_file in files_to_process:
            try:
                # Apply rate limiting
                estimated_tokens = 200
                self.rate_limiter.wait_if_needed(estimated_tokens)

                # Process the file
                result = self.understander.understand_media(media_file, task=task)

                # Record for rate limiting
                actual_tokens = result.tokens_used or estimated_tokens
                self.rate_limiter.record_request(actual_tokens)

                results.append(result)

                if result.success:
                    processed_files += 1
                    # Save result
                    self.understander.save_result(result)
                else:
                    failed_files += 1

                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        "Processed": processed_files,
                        "Failed": failed_files,
                        "Skipped": skipped_files,
                    }
                )

                # Small delay between requests
                time.sleep(0.1)

            except Exception as e:
                failed_files += 1
                progress_bar.set_postfix(
                    {
                        "Processed": processed_files,
                        "Failed": failed_files,
                        "Skipped": skipped_files,
                    }
                )
                logging.error(f"Error processing {media_file.file_path}: {str(e)}")

                # Create error result
                error_result = ApiMediaUnderstandingResult(
                    file_path=media_file.file_path,
                    media_type=media_file.media_type,
                    task=task,
                    content="",
                    processing_time=0.0,
                    success=False,
                    error_message=str(e),
                    model_used=self.config.api_media_understanding.model_name,
                    file_size_mb=media_file.file_size_mb,
                )
                results.append(error_result)

        progress_bar.close()

        # Create batch result
        batch_result = ApiMediaUnderstandingBatchResult(
            total_files=total_files,
            successful_files=processed_files,
            failed_files=failed_files,
            skipped_files=skipped_files,
            results=results,
            batch_processing_time=time.time() - start_time,
        )

        # Calculate statistics
        for result in results:
            if result.tokens_used:
                batch_result.total_tokens_used += result.tokens_used
            if result.api_cost:
                batch_result.total_api_cost += result.api_cost

        self._print_batch_summary(batch_result)
        return batch_result

    def process_images_only(self, base_path: Path = None) -> ApiMediaUnderstandingBatchResult:
        """Process only image files (for backward compatibility)."""
        return self.process_batch(
            base_path=base_path,
            media_types=[MediaType.IMAGE],
            task=ProcessingTask.DESCRIBE,
        )

    def process_documents_only(self, base_path: Path = None) -> ApiMediaUnderstandingBatchResult:
        """Process only document files."""
        return self.process_batch(
            base_path=base_path,
            media_types=[MediaType.DOCUMENT],
            task=ProcessingTask.ANALYZE,
        )

    def process_videos_only(self, base_path: Path = None) -> ApiMediaUnderstandingBatchResult:
        """Process only video files."""
        return self.process_batch(
            base_path=base_path,
            media_types=[MediaType.VIDEO],
            task=ProcessingTask.DESCRIBE,
        )

    def process_audio_only(self, base_path: Path = None) -> ApiMediaUnderstandingBatchResult:
        """Process only audio files."""
        return self.process_batch(
            base_path=base_path,
            media_types=[MediaType.AUDIO],
            task=ProcessingTask.TRANSCRIBE,
        )

    def _create_empty_result(self) -> ApiMediaUnderstandingBatchResult:
        """Create an empty batch result."""
        return ApiMediaUnderstandingBatchResult(
            total_files=0,
            successful_files=0,
            failed_files=0,
            skipped_files=0,
            results=[],
            batch_processing_time=0.0,
        )

    def _print_batch_summary(self, batch_result: ApiMediaUnderstandingBatchResult) -> None:
        """Print a summary of batch processing results."""
        logging.info("=" * 50)
        logging.info("BATCH PROCESSING SUMMARY")
        logging.info("=" * 50)
        logging.info(f"Total files found: {batch_result.total_files}")
        logging.info(f"Successfully processed: {batch_result.successful_files}")
        logging.info(f"Failed: {batch_result.failed_files}")
        logging.info(f"Skipped (already processed): {batch_result.skipped_files}")
        logging.info(f"Total processing time: {batch_result.batch_processing_time:.2f} seconds")

        if batch_result.total_tokens_used > 0:
            logging.info(f"Total tokens used: {batch_result.total_tokens_used:,}")

        if batch_result.total_api_cost > 0:
            logging.info(f"Estimated API cost: ${batch_result.total_api_cost:.4f}")

        # Print statistics by media type
        if batch_result.files_by_type:
            logging.info("\nFiles processed by type:")
            for media_type, count in batch_result.files_by_type.items():
                logging.info(f"  {media_type}: {count}")

        logging.info("=" * 50)


def main():
    """Main entry point for the API-based media understanding system."""
    try:
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)

        # Initialize processor
        processor = ApiMediaUnderstandingProcessor()

        # Display rate limiting information
        rate_limits = processor.config.api_media_understanding.rate_limits
        logging.info(f"Rate Limiting Configuration:")
        logging.info(f"  - Requests per minute (RPM): {rate_limits.rpm}")
        logging.info(f"  - Tokens per minute (TPM): {rate_limits.tpm:,}")
        logging.info(f"  - Requests per day (RPD): {rate_limits.rpd}")
        logging.info(f"  - Max concurrent requests: {rate_limits.max_concurrent_requests}")

        # Demonstrate different processing options
        print("\n=== API Media Understanding Processing Options ===")
        print("1. Process All Media Types")
        print("2. Process Images Only")
        print("3. Process Documents Only")
        print("4. Process Videos Only")
        print("5. Process Audio Only")
        print("6. Process Single Media File")

        choice = input("\nSelect processing method (1-6) [default: 1]: ").strip() or "1"

        if choice == "1":
            print("\n--- Processing All Media Types ---")
            processor.process_batch()
        elif choice == "2":
            print("\n--- Processing Images Only ---")
            processor.process_images_only()
        elif choice == "3":
            print("\n--- Processing Documents Only ---")
            processor.process_documents_only()
        elif choice == "4":
            print("\n--- Processing Videos Only ---")
            processor.process_videos_only()
        elif choice == "5":
            print("\n--- Processing Audio Only ---")
            processor.process_audio_only()
        elif choice == "6":
            print("\n--- Single Media File Processing ---")
            file_path = input("Enter path to media file: ").strip()
            if file_path and Path(file_path).exists():
                result = processor.process_single_media(Path(file_path))
                if result.success:
                    print(f"Analysis: {result.content[:200]}...")
                else:
                    print(f"Processing failed: {result.error_message}")
            else:
                print("Invalid file path")
        else:
            print("Invalid choice, processing all media types")
            processor.process_batch()

    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
