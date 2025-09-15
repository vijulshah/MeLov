"""Utility functions for API-based PDF and document processing using Gemini API."""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import google.generativeai as genai
import yaml
from data_models.api_pdf_processing_data_models import (
    ApiPdfProcessingCompleteConfig,
    ApiPdfProcessingResult,
    DocumentFile,
    DocumentTask,
    DocumentType,
    ProcessingMode,
)
from dotenv import load_dotenv


class DocumentFileDetector:
    """Utility class for detecting and analyzing document files."""

    SUPPORTED_EXTENSIONS = {
        ".pdf": ("application/pdf", DocumentType.PDF),
        ".doc": ("application/msword", DocumentType.DOC),
        ".docx": (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            DocumentType.DOCX,
        ),
        ".txt": ("text/plain", DocumentType.TXT),
        ".rtf": ("application/rtf", DocumentType.RTF),
    }

    @classmethod
    def is_supported_document(cls, file_path: Path) -> bool:
        """Check if the file is a supported document type."""
        return file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def get_document_info(cls, file_path: Path) -> Optional[DocumentFile]:
        """Get document file information."""
        if not file_path.exists() or not cls.is_supported_document(file_path):
            return None

        suffix = file_path.suffix.lower()
        mime_type, doc_type = cls.SUPPORTED_EXTENSIONS[suffix]

        # Get file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Try to get page count for PDFs (would need PyPDF2 or similar)
        page_count = None
        if doc_type == DocumentType.PDF:
            page_count = cls._get_pdf_page_count(file_path)

        metadata = {
            "extension": suffix,
            "created_time": file_path.stat().st_ctime,
            "modified_time": file_path.stat().st_mtime,
        }

        return DocumentFile(
            file_path=file_path,
            document_type=doc_type,
            mime_type=mime_type,
            file_size_mb=file_size_mb,
            page_count=page_count,
            metadata=metadata,
        )

    @classmethod
    def find_documents(
        cls, base_path: Path, recursive: bool = True, max_files: Optional[int] = None
    ) -> List[DocumentFile]:
        """Find all supported documents in a directory."""
        documents = []

        if not base_path.exists():
            logging.warning(f"Path does not exist: {base_path}")
            return documents

        if base_path.is_file():
            doc_info = cls.get_document_info(base_path)
            if doc_info:
                documents.append(doc_info)
            return documents

        # Search pattern
        pattern = "**/*" if recursive else "*"

        for file_path in base_path.glob(pattern):
            if file_path.is_file() and cls.is_supported_document(file_path):
                doc_info = cls.get_document_info(file_path)
                if doc_info:
                    documents.append(doc_info)

                    if max_files and len(documents) >= max_files:
                        break

        return documents

    @staticmethod
    def _get_pdf_page_count(file_path: Path) -> Optional[int]:
        """Get page count for PDF files. Returns None if unable to determine."""
        try:
            # This is a simple approach - for production, use PyPDF2 or similar
            # For now, we'll estimate or return None
            return None
        except Exception:
            return None


class ApiPdfProcessorConfigManager:
    """Configuration manager for PDF processing."""

    @staticmethod
    def load_config(config_path: str) -> ApiPdfProcessingCompleteConfig:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            return ApiPdfProcessingCompleteConfig(**config_data)
        except Exception as e:
            logging.error(f"Failed to load config from {config_path}: {str(e)}")
            # Return default configuration
            return ApiPdfProcessingCompleteConfig(document_processing={}, paths={}, output_structure={}, prompts={})


class ApiDocumentProcessor:
    """Core API processor for document understanding using Gemini API."""

    def __init__(self, config: ApiPdfProcessingCompleteConfig):
        self.config = config
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            # Try to load from .env file in project root
            project_root = Path(__file__).parent.parent.parent  # Go up to project root
            env_file = project_root / ".env"

            if env_file.exists():
                load_dotenv(env_file)
                self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        # Configure the API
        genai.configure(api_key=self.api_key)

        # Initialize the model
        self.model = genai.GenerativeModel("gemini-2.5-flash")

        # Rate limiting state
        self._last_request_time = 0
        self._request_count = 0
        self._daily_request_count = 0
        self._last_reset_date = time.strftime("%Y-%m-%d")

    def _apply_rate_limiting(self) -> None:
        """Apply rate limiting for API requests."""
        current_time = time.time()
        current_date = time.strftime("%Y-%m-%d")

        # Reset daily counter if needed
        if current_date != self._last_reset_date:
            self._daily_request_count = 0
            self._last_reset_date = current_date

        # Check daily limit (using media understanding limits as base)
        if self._daily_request_count >= 1500:  # Gemini free tier daily limit
            raise Exception("Daily request limit reached (1500)")

        # Apply minimum delay between requests (4 seconds for 15 RPM)
        min_delay = 4.0  # 60/15 = 4 seconds
        time_since_last = current_time - self._last_request_time

        if time_since_last < min_delay:
            sleep_time = min_delay - time_since_last
            logging.info(f"Rate limiting: waiting {sleep_time:.2f} seconds")
            time.sleep(sleep_time)

        self._last_request_time = time.time()
        self._daily_request_count += 1

    def _estimate_token_usage(self, document_file: DocumentFile, content: str) -> int:
        """Estimate token usage based on document and response content."""
        # Base tokens for document processing (conservative estimate)
        base_tokens = 200

        # Add tokens for document content (estimate 1 token per 4 characters)
        if document_file.page_count:
            # For PDFs, use page count
            document_tokens = document_file.page_count * 258  # Gemini's page token rate
        else:
            # For other documents, estimate based on file size
            document_tokens = int(document_file.file_size_mb * 1000)  # Rough estimate

        # Add tokens for response content
        response_tokens = len(content) // 4 if content else 0

        return base_tokens + document_tokens + response_tokens

    def process_document_sync(
        self,
        document_file: DocumentFile,
        task: DocumentTask = DocumentTask.EXTRACT_TEXT,
        custom_prompt: Optional[str] = None,
    ) -> ApiPdfProcessingResult:
        """Process a document synchronously."""
        start_time = time.time()

        try:
            # Apply rate limiting
            self._apply_rate_limiting()

            # Prepare the prompt
            prompt = self._get_prompt(task, custom_prompt)

            # Prepare the document for API call
            if document_file.file_size_mb > 15:  # Use Files API for larger files
                result = self._process_with_files_api_sync(document_file, prompt)
            else:
                result = self._process_inline_sync(document_file, prompt)

            processing_time = time.time() - start_time

            return ApiPdfProcessingResult(
                file_path=document_file.file_path,
                document_type=document_file.document_type,
                task=task,
                content=result["content"],
                processing_time=processing_time,
                processing_mode=ProcessingMode.SYNC,
                success=result["success"],
                error_message=result.get("error_message"),
                model_used="gemini-2.5-flash",
                tokens_used=result.get("tokens_used"),
                file_size_mb=document_file.file_size_mb,
                page_count=document_file.page_count,
                text_length=len(result["content"]) if result["success"] else None,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logging.error(f"Error processing {document_file.file_path}: {str(e)}")

            return ApiPdfProcessingResult(
                file_path=document_file.file_path,
                document_type=document_file.document_type,
                task=task,
                content="",
                processing_time=processing_time,
                processing_mode=ProcessingMode.SYNC,
                success=False,
                error_message=str(e),
                model_used="gemini-2.5-flash",
                file_size_mb=document_file.file_size_mb,
                page_count=document_file.page_count,
            )

    async def process_document_async(
        self,
        document_file: DocumentFile,
        task: DocumentTask = DocumentTask.EXTRACT_TEXT,
        custom_prompt: Optional[str] = None,
    ) -> ApiPdfProcessingResult:
        """Process a document asynchronously."""
        start_time = time.time()

        try:
            # Apply rate limiting (in async context)
            await self._apply_rate_limiting_async()

            # Prepare the prompt
            prompt = self._get_prompt(task, custom_prompt)

            # Prepare the document for API call
            if document_file.file_size_mb > 15:  # Use Files API for larger files
                result = await self._process_with_files_api_async(document_file, prompt)
            else:
                result = await self._process_inline_async(document_file, prompt)

            processing_time = time.time() - start_time

            return ApiPdfProcessingResult(
                file_path=document_file.file_path,
                document_type=document_file.document_type,
                task=task,
                content=result["content"],
                processing_time=processing_time,
                processing_mode=ProcessingMode.ASYNC,
                success=result["success"],
                error_message=result.get("error_message"),
                model_used="gemini-2.5-flash",
                tokens_used=result.get("tokens_used"),
                file_size_mb=document_file.file_size_mb,
                page_count=document_file.page_count,
                text_length=len(result["content"]) if result["success"] else None,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logging.error(f"Error processing {document_file.file_path}: {str(e)}")

            return ApiPdfProcessingResult(
                file_path=document_file.file_path,
                document_type=document_file.document_type,
                task=task,
                content="",
                processing_time=processing_time,
                processing_mode=ProcessingMode.ASYNC,
                success=False,
                error_message=str(e),
                model_used="gemini-2.5-flash",
                file_size_mb=document_file.file_size_mb,
                page_count=document_file.page_count,
            )

    def _get_prompt(self, task: DocumentTask, custom_prompt: Optional[str] = None) -> str:
        """Get the appropriate prompt for the task."""
        if custom_prompt:
            return custom_prompt

        return self.config.prompts.default_prompts.get(task.value, "Extract and analyze the content of this document.")

    def _process_inline_sync(self, document_file: DocumentFile, prompt: str) -> Dict:
        """Process document inline (for smaller files) synchronously."""
        try:
            # Read and encode the file
            with open(document_file.file_path, "rb") as f:
                f.read()

            # For PDFs and other binary documents, we need to use the Files API
            # or convert to text first. For now, we'll use Files API approach
            return self._process_with_files_api_sync(document_file, prompt)

        except Exception as e:
            return {"success": False, "content": "", "error_message": str(e)}

    async def _process_inline_async(self, document_file: DocumentFile, prompt: str) -> Dict:
        """Process document inline (for smaller files) asynchronously."""
        try:
            # For async, we'll also use the Files API approach for consistency
            return await self._process_with_files_api_async(document_file, prompt)

        except Exception as e:
            return {"success": False, "content": "", "error_message": str(e)}

    def _process_with_files_api_sync(self, document_file: DocumentFile, prompt: str) -> Dict:
        """Process document using Files API synchronously."""
        try:
            # Upload file to Gemini Files API
            uploaded_file = genai.upload_file(path=str(document_file.file_path), mime_type=document_file.mime_type)

            # Wait for file to be processed
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(1)
                uploaded_file = genai.get_file(uploaded_file.name)

            if uploaded_file.state.name == "FAILED":
                raise Exception(f"File upload failed: {uploaded_file.state}")

            # Generate content
            response = self.model.generate_content([prompt, uploaded_file])

            # Clean up uploaded file
            try:
                genai.delete_file(uploaded_file.name)
            except Exception:
                pass  # Ignore cleanup errors

            # Calculate token usage (estimate based on content length)
            content = response.text if response.text else ""
            estimated_tokens = self._estimate_token_usage(document_file, content)

            return {
                "success": True,
                "content": content,
                "tokens_used": estimated_tokens,
                "uploaded_file_uri": uploaded_file.uri,
            }

        except Exception as e:
            return {"success": False, "content": "", "error_message": str(e)}

    async def _process_with_files_api_async(self, document_file: DocumentFile, prompt: str) -> Dict:
        """Process document using Files API asynchronously."""
        try:
            # Note: The current Google AI Python SDK doesn't have async support for Files API
            # For now, we'll run the sync version in an executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._process_with_files_api_sync, document_file, prompt)

        except Exception as e:
            return {"success": False, "content": "", "error_message": str(e)}

    async def _apply_rate_limiting_async(self) -> None:
        """Apply rate limiting for async API requests."""
        current_time = time.time()
        current_date = time.strftime("%Y-%m-%d")

        # Reset daily counter if needed
        if current_date != self._last_reset_date:
            self._daily_request_count = 0
            self._last_reset_date = current_date

        # Check daily limit
        if self._daily_request_count >= 1500:
            raise Exception("Daily request limit reached (1500)")

        # Apply minimum delay between requests
        min_delay = 4.0  # 60/15 = 4 seconds for 15 RPM
        time_since_last = current_time - self._last_request_time

        if time_since_last < min_delay:
            sleep_time = min_delay - time_since_last
            logging.info(f"Rate limiting: waiting {sleep_time:.2f} seconds")
            await asyncio.sleep(sleep_time)

        self._last_request_time = time.time()
        self._daily_request_count += 1

    def should_process_document(self, document_file: DocumentFile, task: DocumentTask, output_dir: Path) -> bool:
        """Check if document should be processed (not already processed)."""
        # Check if output file already exists
        output_path = self._get_output_path(document_file, task, output_dir)

        if output_path.exists() and not self.config.document_processing.overwrite_existing:
            return False

        return True

    def _get_output_path(self, document_file: DocumentFile, task: DocumentTask, output_dir: Path) -> Path:
        """Get the output path for processed document."""
        # Create subdirectory based on task
        task_dir = output_dir / task.value
        task_dir.mkdir(parents=True, exist_ok=True)

        # Create output filename
        base_name = document_file.file_path.stem
        output_filename = f"{base_name}_{task.value}.txt"

        return task_dir / output_filename

    def save_result(self, result: ApiPdfProcessingResult, output_dir: Path) -> Optional[Path]:
        """Save processing result to file."""
        if not result.success:
            return None

        try:
            # Create a simple output path based on file path and task
            task_dir = output_dir / result.task.value
            task_dir.mkdir(parents=True, exist_ok=True)

            base_name = result.file_path.stem
            output_filename = f"{base_name}_{result.task.value}.txt"
            output_path = task_dir / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.content)

            return output_path

        except Exception as e:
            logging.error(f"Failed to save result for {result.file_path}: {str(e)}")
            return None


class AsyncDocumentBatchProcessor:
    """Batch processor for handling multiple documents asynchronously."""

    def __init__(self, processor: ApiDocumentProcessor, max_concurrent: int = 3):
        self.processor = processor
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_batch(
        self,
        documents: List[DocumentFile],
        task: DocumentTask = DocumentTask.EXTRACT_TEXT,
        custom_prompt: Optional[str] = None,
    ) -> List[ApiPdfProcessingResult]:
        """Process a batch of documents asynchronously."""

        async def process_single_with_semaphore(
            doc: DocumentFile,
        ) -> ApiPdfProcessingResult:
            async with self.semaphore:
                return await self.processor.process_document_async(doc, task, custom_prompt)

        # Create tasks for all documents
        tasks = [process_single_with_semaphore(doc) for doc in documents]

        # Execute all tasks concurrently with limited concurrency
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                error_result = ApiPdfProcessingResult(
                    file_path=documents[i].file_path,
                    document_type=documents[i].document_type,
                    task=task,
                    content="",
                    processing_time=0.0,
                    processing_mode=ProcessingMode.ASYNC,
                    success=False,
                    error_message=str(result),
                    model_used="gemini-2.5-flash",
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)

        return processed_results
