"""API-based image captioning processor for cropped person images."""

import asyncio
import json
import logging
import tempfile
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

# Load environment variables from .env file
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

from data_models.api_img_captioning_data_models import (
    ApiCaptioningBatchResult,
    ApiCaptioningResult,
)
from tqdm import tqdm
from utils.api_img_captioning_utils import (
    ApiCaptioningConfigManager,
    ApiImageCaptioner,
    CroppedImageDiscovery,
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


class ApiImageCaptioningProcessor:
    """Main processor for API-based image captioning."""

    def __init__(
        self,
        config_path: str = "./data_processing/configs/api_img_captioning_config.yaml",
    ):
        self.config = ApiCaptioningConfigManager.load_config(config_path)
        self.prompt_config = ApiCaptioningConfigManager.load_prompt_config(config_path)
        self.captioner = ApiImageCaptioner(self.config, self.prompt_config)

        # Initialize rate limiter with config values
        self.rate_limiter = RateLimiter(
            rpm=self.config.api_captioning.rate_limits.rpm,
            tpm=self.config.api_captioning.rate_limits.tpm,
            rpd=self.config.api_captioning.rate_limits.rpd,
        )

        # Set up logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
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
        # Apply rate limiting before making the request
        estimated_tokens = 100  # Conservative estimate for image + prompt tokens
        self.rate_limiter.wait_if_needed(estimated_tokens)

        result = self.captioner.generate_caption(image_path)

        # Record the request for rate limiting
        actual_tokens = estimated_tokens  # Could be enhanced to track actual tokens from response
        self.rate_limiter.record_request(actual_tokens)

        return result

    async def process_single_image_async(self, image_path: Path, semaphore: asyncio.Semaphore) -> ApiCaptioningResult:
        """Process a single cropped image for captioning asynchronously with semaphore control."""
        async with semaphore:
            # Apply rate limiting before making the request
            estimated_tokens = 100  # Conservative estimate for image + prompt tokens
            await asyncio.get_event_loop().run_in_executor(None, self.rate_limiter.wait_if_needed, estimated_tokens)

            # Run the synchronous captioning in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.captioner.generate_caption, image_path)

            # Record the request for rate limiting
            actual_tokens = estimated_tokens  # Could be enhanced to track actual tokens from response
            await loop.run_in_executor(None, self.rate_limiter.record_request, actual_tokens)

            return result

    async def process_batch_async(self, base_path: Path = None, max_concurrent: int = None) -> ApiCaptioningBatchResult:
        """Process all cropped images asynchronously with concurrency control."""
        start_time = time.time()

        if base_path is None:
            base_path = Path(self.config.paths.processed_data_dir)

        if not base_path.exists():
            logging.error(f"Base path does not exist: {base_path}")
            return self._create_empty_result()

        # Use config batch_size if max_concurrent not specified, but respect rate limits
        if max_concurrent is None:
            max_concurrent = self.config.api_captioning.batch_size

        # Ensure max_concurrent doesn't exceed rate limit configuration
        max_allowed = min(
            self.config.api_captioning.rate_limits.max_concurrent_requests,
            self.config.api_captioning.rate_limits.rpm,  # Don't exceed RPM either
        )
        if max_concurrent > max_allowed:
            logging.warning(f"Reducing max_concurrent from {max_concurrent} to {max_allowed} to respect rate limits")
            max_concurrent = max_allowed

        # Find all cropped images
        image_paths = CroppedImageDiscovery.find_cropped_images(base_path, self.config)

        if not image_paths:
            logging.warning(f"No cropped images found in {base_path}")
            return self._create_empty_result()

        # Filter images that need processing
        images_to_process = []
        skipped_images = 0

        for image_path in image_paths:
            if self.captioner.should_process_image(image_path):
                images_to_process.append(image_path)
            else:
                skipped_images += 1

        if not images_to_process:
            logging.info(f"All {len(images_to_process)} images already have captions, nothing to process")
            return self._create_empty_result()

        total_images = len(image_paths)
        logging.info(
            f"Starting async processing of {len(images_to_process)} images with max concurrency: {max_concurrent}"
        )
        logging.info(f"API endpoint: {self.config.api_captioning.api_endpoint}")
        logging.info(f"Model: {self.config.api_captioning.model_name}")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        # Create progress tracking
        processed_images = 0
        failed_images = 0
        results = []

        # Create progress bar
        progress_bar = tqdm(
            total=len(images_to_process),
            desc="Processing images (async)",
            unit="img",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {desc}",
        )

        async def process_with_progress(image_path: Path) -> ApiCaptioningResult:
            """Process image and update progress."""
            try:
                result = await self.process_single_image_async(image_path, semaphore)
                progress_bar.set_description(f"Processed {image_path.name}")
                progress_bar.update(1)
                return result
            except Exception as e:
                error_result = ApiCaptioningResult(
                    image_path=image_path,
                    processing_time=0.0,
                    success=False,
                    error_message=str(e),
                )
                progress_bar.update(1)
                return error_result

        # Process all images concurrently
        tasks = [process_with_progress(image_path) for image_path in images_to_process]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Close progress bar
        progress_bar.close()

        # Process results and count successes/failures
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                # Handle exceptions that weren't caught in process_with_progress
                failed_images += 1
                logging.error(f"Async processing exception: {str(result)}")
            elif isinstance(result, ApiCaptioningResult):
                final_results.append(result)
                if result.success:
                    processed_images += 1
                    logging.debug(f"Generated caption: {result.caption}")
                else:
                    failed_images += 1
                    logging.error(f"Failed: {result.image_path.name} - {result.error_message}")

        total_processing_time = time.time() - start_time

        # Get usage statistics
        usage_stats = self.captioner.get_usage_stats()

        # Create and return batch result
        batch_result = ApiCaptioningBatchResult(
            total_images=total_images,
            processed_images=processed_images,
            skipped_images=skipped_images,
            failed_images=failed_images,
            total_processing_time=total_processing_time,
            total_api_calls=len(images_to_process),  # Each image is one API call
            total_api_cost=None,
            results=final_results,
        )

        self._print_batch_summary(batch_result, usage_stats)
        return batch_result

    def process_batch(self, base_path: Path = None) -> ApiCaptioningBatchResult:
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
        total_api_calls = 0

        logging.info(f"Starting to process {total_images} images using API...")
        logging.info(f"API endpoint: {self.config.api_captioning.api_endpoint}")
        logging.info(f"Model: {self.config.api_captioning.model_name}")

        # Process each image with progress bar
        progress_bar = tqdm(
            image_paths,
            desc="Processing images",
            unit="img",
            total=total_images,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {desc}",
        )

        for image_path in progress_bar:
            try:
                # Update progress bar description with current image name
                progress_bar.set_description(f"Processing {image_path.name}")

                # Check if we should process this image
                if not self.captioner.should_process_image(image_path):
                    skipped_images += 1
                    progress_bar.set_postfix(
                        {
                            "Processed": processed_images,
                            "Skipped": skipped_images,
                            "Failed": failed_images,
                            "API Calls": total_api_calls,
                        }
                    )
                    logging.debug(f"Caption already exists for {image_path.name}, skipping")
                    continue

                # Process the image
                result = self.process_single_image(image_path)
                results.append(result)
                total_api_calls += 1

                if result.success:
                    processed_images += 1
                    progress_bar.set_postfix(
                        {
                            "Processed": processed_images,
                            "Skipped": skipped_images,
                            "Failed": failed_images,
                            "API Calls": total_api_calls,
                        }
                    )
                    logging.debug(f"Generated caption: {result.caption}")
                else:
                    failed_images += 1
                    progress_bar.set_postfix(
                        {
                            "Processed": processed_images,
                            "Skipped": skipped_images,
                            "Failed": failed_images,
                            "API Calls": total_api_calls,
                        }
                    )
                    logging.error(f"Failed: {image_path.name} - {result.error_message}")

                # Add delay between API calls to respect rate limits
                if processed_images + failed_images < total_images - skipped_images:
                    time.sleep(0.1)  # Small delay between requests

            except Exception as e:
                failed_images += 1
                total_api_calls += 1
                progress_bar.set_postfix(
                    {
                        "Processed": processed_images,
                        "Skipped": skipped_images,
                        "Failed": failed_images,
                        "API Calls": total_api_calls,
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

        # Get usage statistics
        usage_stats = self.captioner.get_usage_stats()

        # Create and return batch result
        batch_result = ApiCaptioningBatchResult(
            total_images=total_images,
            processed_images=processed_images,
            skipped_images=skipped_images,
            failed_images=failed_images,
            total_processing_time=total_processing_time,
            total_api_calls=total_api_calls,
            total_api_cost=None,
            results=results,
        )

        self._print_batch_summary(batch_result, usage_stats)
        return batch_result

    def process_batch_inline_requests(self, base_path: Path = None) -> ApiCaptioningBatchResult:
        """Process images using Gemini Batch API with inline requests."""
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

        # Filter images that need processing
        images_to_process = []
        skipped_images = 0

        for image_path in image_paths:
            if self.captioner.should_process_image(image_path):
                images_to_process.append(image_path)
            else:
                skipped_images += 1

        if not images_to_process:
            logging.info(f"All {len(images_to_process)} images already have captions, nothing to process")
            return self._create_empty_result()

        logging.info(f"Processing {len(images_to_process)} images using Gemini Batch API (inline requests)...")

        try:
            # Import Gemini client

            client = genai.Client()

            # Build inline requests for batch processing
            inline_requests = []
            for image_path in images_to_process:
                try:
                    request = self.captioner.request_builder.build_gemini_request(image_path)
                    inline_requests.append(request)
                except Exception as e:
                    logging.error(f"Failed to build request for {image_path}: {str(e)}")
                    continue

            if not inline_requests:
                logging.error("No valid requests could be built")
                return self._create_empty_result()

            # Create batch job with inline requests
            logging.info(f"Creating batch job with {len(inline_requests)} inline requests...")
            inline_batch_job = client.batches.create(
                model="models/gemini-2.5-flash",
                src=inline_requests,
                config={
                    "display_name": f"image-captioning-batch-{int(time.time())}",
                },
            )

            logging.info(f"Created batch job: {inline_batch_job.name}")

            # Monitor job status
            job_name = inline_batch_job.name
            completed_states = {
                "JOB_STATE_SUCCEEDED",
                "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED",
                "JOB_STATE_EXPIRED",
            }

            logging.info(f"Monitoring batch job status...")
            batch_job = client.batches.get(name=job_name)

            while batch_job.state.name not in completed_states:
                logging.info(f"Current state: {batch_job.state.name}. Waiting 30 seconds...")
                time.sleep(30)
                batch_job = client.batches.get(name=job_name)

            logging.info(f"Batch job finished with state: {batch_job.state.name}")

            # Process results
            results = []
            processed_images = 0
            failed_images = 0

            if batch_job.state.name == "JOB_STATE_SUCCEEDED":
                if batch_job.dest and batch_job.dest.inlined_responses:
                    for i, inline_response in enumerate(batch_job.dest.inlined_responses):
                        if i < len(images_to_process):
                            image_path = images_to_process[i]

                            if inline_response.response:
                                try:
                                    caption = inline_response.response.text.strip()
                                    caption_path = image_path.with_suffix(".txt")

                                    # Save caption
                                    success = self.captioner._save_caption(caption_path, caption)

                                    result = ApiCaptioningResult(
                                        image_path=image_path,
                                        caption=caption if success else None,
                                        caption_path=caption_path if success else None,
                                        processing_time=0.0,  # Batch processing time not individual
                                        success=success,
                                        retry_count=0,
                                    )
                                    results.append(result)

                                    if success:
                                        processed_images += 1
                                        logging.debug(f"Saved caption for {image_path.name}: {caption}")
                                    else:
                                        failed_images += 1

                                except Exception as e:
                                    failed_images += 1
                                    logging.error(f"Error processing response for {image_path}: {str(e)}")
                                    result = ApiCaptioningResult(
                                        image_path=image_path,
                                        processing_time=0.0,
                                        success=False,
                                        error_message=str(e),
                                    )
                                    results.append(result)
                            elif inline_response.error:
                                failed_images += 1
                                error_msg = str(inline_response.error)
                                logging.error(f"API error for {image_path}: {error_msg}")
                                result = ApiCaptioningResult(
                                    image_path=image_path,
                                    processing_time=0.0,
                                    success=False,
                                    error_message=error_msg,
                                )
                                results.append(result)
                else:
                    logging.error("No inline responses found in successful batch job")
                    failed_images = len(images_to_process)
            else:
                logging.error(f"Batch job failed with state: {batch_job.state.name}")
                if batch_job.error:
                    logging.error(f"Batch job error: {batch_job.error}")
                failed_images = len(images_to_process)

            total_processing_time = time.time() - start_time

            # Create batch result
            batch_result = ApiCaptioningBatchResult(
                total_images=len(image_paths),
                processed_images=processed_images,
                skipped_images=skipped_images,
                failed_images=failed_images,
                total_processing_time=total_processing_time,
                total_api_calls=1,  # One batch API call
                total_api_cost=None,
                results=results,
            )

            self._print_batch_summary(batch_result, self.captioner.get_usage_stats())
            return batch_result

        except Exception as e:
            logging.error(f"Batch processing error: {str(e)}")
            total_processing_time = time.time() - start_time
            return ApiCaptioningBatchResult(
                total_images=len(image_paths),
                processed_images=0,
                skipped_images=skipped_images,
                failed_images=len(images_to_process),
                total_processing_time=total_processing_time,
                total_api_calls=0,
                total_api_cost=None,
                results=[],
            )

    def process_batch_input_file(self, base_path: Path = None) -> ApiCaptioningBatchResult:
        """Process images using Gemini Batch API with input file method."""
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

        # Filter images that need processing
        images_to_process = []
        skipped_images = 0

        for image_path in image_paths:
            if self.captioner.should_process_image(image_path):
                images_to_process.append(image_path)
            else:
                skipped_images += 1

        if not images_to_process:
            logging.info("All images already have captions, nothing to process")
            return self._create_empty_result()

        logging.info(f"Processing {len(images_to_process)} images using Gemini Batch API (input file)...")

        try:
            client = genai.Client()

            # Create JSONL file with requests
            batch_requests = []
            for i, image_path in enumerate(images_to_process):
                try:
                    request = self.captioner.request_builder.build_gemini_request(image_path)
                    batch_requests.append({"key": f"request-{i}", "request": request})
                except Exception as e:
                    logging.error(f"Failed to build request for {image_path}: {str(e)}")
                    continue

            if not batch_requests:
                logging.error("No valid requests could be built")
                return self._create_empty_result()

            # Create temporary JSONL file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
                temp_file_path = f.name
                for req in batch_requests:
                    f.write(json.dumps(req) + "\n")

            # Upload the file to Gemini File API
            logging.info(f"Uploading batch requests file with {len(batch_requests)} requests...")
            uploaded_file = client.files.upload(
                file=temp_file_path,
                config=types.UploadFileConfig(
                    display_name=f"image-captioning-batch-{int(time.time())}",
                    mime_type="application/jsonl",
                ),
            )

            logging.info(f"Uploaded file: {uploaded_file.name}")

            # Create batch job with input file
            file_batch_job = client.batches.create(
                model="gemini-2.5-flash",
                src=uploaded_file.name,
                config={
                    "display_name": f"image-captioning-file-batch-{int(time.time())}",
                },
            )

            logging.info(f"Created batch job: {file_batch_job.name}")

            # Monitor job status
            job_name = file_batch_job.name
            completed_states = {
                "JOB_STATE_SUCCEEDED",
                "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED",
                "JOB_STATE_EXPIRED",
            }

            logging.info(f"Monitoring batch job status...")
            batch_job = client.batches.get(name=job_name)

            while batch_job.state.name not in completed_states:
                logging.info(f"Current state: {batch_job.state.name}. Waiting 30 seconds...")
                time.sleep(30)
                batch_job = client.batches.get(name=job_name)

            logging.info(f"Batch job finished with state: {batch_job.state.name}")

            # Process results
            results = []
            processed_images = 0
            failed_images = 0

            if batch_job.state.name == "JOB_STATE_SUCCEEDED":
                if batch_job.dest and batch_job.dest.file_name:
                    # Download and process result file
                    result_file_name = batch_job.dest.file_name
                    logging.info(f"Downloading results from file: {result_file_name}")

                    file_content = client.files.download(file=result_file_name)
                    result_lines = file_content.decode("utf-8").strip().split("\n")

                    # Create mapping from request keys to image paths
                    key_to_image = {}
                    for i, image_path in enumerate(images_to_process[: len(batch_requests)]):
                        key_to_image[f"request-{i}"] = image_path

                    for line in result_lines:
                        try:
                            result_data = json.loads(line)
                            key = result_data.get("key")

                            if key in key_to_image:
                                image_path = key_to_image[key]

                                if "response" in result_data:
                                    try:
                                        # Extract caption from response
                                        response = result_data["response"]
                                        if "candidates" in response and response["candidates"]:
                                            candidate = response["candidates"][0]
                                            if "content" in candidate and "parts" in candidate["content"]:
                                                parts = candidate["content"]["parts"]
                                                if parts and "text" in parts[0]:
                                                    caption = parts[0]["text"].strip()
                                                    caption_path = image_path.with_suffix(".txt")

                                                    # Save caption
                                                    success = self.captioner._save_caption(caption_path, caption)

                                                    result = ApiCaptioningResult(
                                                        image_path=image_path,
                                                        caption=(caption if success else None),
                                                        caption_path=(caption_path if success else None),
                                                        processing_time=0.0,
                                                        success=success,
                                                        retry_count=0,
                                                    )
                                                    results.append(result)

                                                    if success:
                                                        processed_images += 1
                                                        logging.debug(f"Saved caption for {image_path.name}: {caption}")
                                                    else:
                                                        failed_images += 1
                                                else:
                                                    raise ValueError("No text found in response parts")
                                            else:
                                                raise ValueError("No content parts found in candidate")
                                        else:
                                            raise ValueError("No candidates found in response")
                                    except Exception as e:
                                        failed_images += 1
                                        logging.error(f"Error processing response for {image_path}: {str(e)}")
                                        result = ApiCaptioningResult(
                                            image_path=image_path,
                                            processing_time=0.0,
                                            success=False,
                                            error_message=str(e),
                                        )
                                        results.append(result)
                                elif "error" in result_data:
                                    failed_images += 1
                                    error_msg = str(result_data["error"])
                                    logging.error(f"API error for {image_path}: {error_msg}")
                                    result = ApiCaptioningResult(
                                        image_path=image_path,
                                        processing_time=0.0,
                                        success=False,
                                        error_message=error_msg,
                                    )
                                    results.append(result)
                        except json.JSONDecodeError as e:
                            logging.error(f"Failed to parse result line: {line}, error: {str(e)}")
                            continue
                else:
                    logging.error("No result file found in successful batch job")
                    failed_images = len(images_to_process)
            else:
                logging.error(f"Batch job failed with state: {batch_job.state.name}")
                if batch_job.error:
                    logging.error(f"Batch job error: {batch_job.error}")
                failed_images = len(images_to_process)

            # Clean up temporary file
            try:
                import os

                os.unlink(temp_file_path)
            except:
                pass

            total_processing_time = time.time() - start_time

            # Create batch result
            batch_result = ApiCaptioningBatchResult(
                total_images=len(image_paths),
                processed_images=processed_images,
                skipped_images=skipped_images,
                failed_images=failed_images,
                total_processing_time=total_processing_time,
                total_api_calls=1,  # One batch API call
                total_api_cost=None,
                results=results,
            )

            self._print_batch_summary(batch_result, self.captioner.get_usage_stats())
            return batch_result

        except Exception as e:
            logging.error(f"Batch processing error: {str(e)}")
            total_processing_time = time.time() - start_time
            return ApiCaptioningBatchResult(
                total_images=len(image_paths),
                processed_images=0,
                skipped_images=skipped_images,
                failed_images=len(images_to_process),
                total_processing_time=total_processing_time,
                total_api_calls=0,
                total_api_cost=None,
                results=[],
            )

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

    def _print_batch_summary(self, batch_result: ApiCaptioningBatchResult, usage_stats) -> None:
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

        # API usage statistics
        logging.info(f"\nAPI Usage Statistics:")
        logging.info(f"Successful requests: {usage_stats.successful_requests}")
        logging.info(f"Failed requests: {usage_stats.failed_requests}")
        logging.info(f"Average response time: {usage_stats.average_response_time:.2f}s")

        logging.info(f"\nResults saved in: {self.config.paths.processed_data_dir}")
        logging.info(f"{'='*60}")

    def validate_api_setup(self) -> bool:
        """Validate that Gemini API is properly configured and accessible."""
        try:
            # Create a simple test to validate API setup
            import os

            api_key = os.getenv(self.config.api_captioning.api_key_env_var)
            if not api_key:
                logging.error(
                    f"API key not found in environment variable: {self.config.api_captioning.api_key_env_var}"
                )
                return False

            # Try to import and initialize Gemini client
            try:

                client = genai.Client(api_key=api_key)
                logging.info("Gemini API client initialized successfully")
            except ImportError:
                logging.error("Google GenAI library not found. Please install with: pip install google-generativeai")
                return False
            except Exception as e:
                logging.error(f"Failed to initialize Gemini client: {str(e)}")
                return False

            logging.info("API configuration appears valid")
            logging.info(f"Using model: {self.config.api_captioning.model_name}")
            logging.info(f"API endpoint: {self.config.api_captioning.api_endpoint}")
            return True

        except Exception as e:
            logging.error(f"API validation failed: {str(e)}")
            return False

    def run_async_batch(self, max_concurrent_requests: int = None) -> List[ApiCaptioningResult]:
        """
        Synchronous wrapper for async batch processing with semaphore control.

        Args:
            max_concurrent_requests: Maximum number of concurrent API requests (defaults to rate limit config)

        Returns:
            List of captioning results
        """
        if max_concurrent_requests is None:
            max_concurrent_requests = self.config.api_captioning.rate_limits.max_concurrent_requests

        return asyncio.run(self.process_batch_async(max_concurrent_requests))


def main():
    """Main entry point for the API-based image captioning system."""
    try:
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)

        # Initialize processor
        processor = ApiImageCaptioningProcessor()

        # Validate API setup
        if not processor.validate_api_setup():
            logging.error("API setup validation failed. Please check your configuration and API key.")
            return

        # Display rate limiting information
        rate_limits = processor.config.api_captioning.rate_limits
        print(f"\n=== Rate Limiting Configuration ===")
        print(f"Requests per minute (RPM): {rate_limits.rpm}")
        print(f"Tokens per minute (TPM): {rate_limits.tpm:,}")
        print(f"Requests per day (RPD): {rate_limits.rpd}")
        print(f"Max concurrent requests: {rate_limits.max_concurrent_requests}")

        # Demonstrate different processing methods:
        print("\n=== API Image Captioning Processing Options ===")
        print("1. Sync Batch Processing (Inline Requests)")
        print("2. Async Batch Processing (Concurrent with Semaphore)")
        print("3. Single Image Processing (Sync)")

        choice = input("\nSelect processing method (1/2/3) [default: 1]: ").strip() or "1"

        if choice == "1":
            print("\n--- Processing with Sync Batch (Inline Requests) ---")
            processor.process_batch_inline_requests()

        elif choice == "2":
            print("\n--- Processing with Async Batch (Concurrent) ---")
            rate_limit = processor.config.api_captioning.rate_limits.max_concurrent_requests
            max_concurrent = int(input(f"Max concurrent requests [default: {rate_limit}]: ").strip() or str(rate_limit))
            results = processor.run_async_batch(max_concurrent_requests=max_concurrent)
            print(f"\nAsync batch processing completed: {len(results)} images processed")

        elif choice == "3":
            print("\n--- Single Image Processing ---")
            # Get sample image for single processing
            sample_images = list(Path(processor.config.paths.input_dir).glob("**/*.jpg"))[:5]
            if sample_images:
                print(f"Processing first image: {sample_images[0]}")
                result = processor.process_single_image(sample_images[0])
                if result and result.success:
                    print(f"Caption: {result.caption}")
                else:
                    print("Processing failed")
            else:
                print("No images found for single processing")
        else:
            print("Invalid choice, using default sync batch processing")
            processor.process_batch_inline_requests()

    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
