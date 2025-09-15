"""API-based PDF and document processor for analyzing documents using Gemini API."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from data_models.api_pdf_processing_data_models import (
    ApiPdfProcessingBatchResult,
    ApiPdfProcessingResult,
    DocumentTask,
    DocumentType,
    ProcessingMode,
)
from tqdm import tqdm
from utils.api_pdf_processing_utils import (
    ApiDocumentProcessor,
    ApiPdfProcessorConfigManager,
    AsyncDocumentBatchProcessor,
    DocumentFileDetector,
)


class ApiPdfProcessingProcessor:
    """Main processor for API-based PDF and document processing using Gemini API."""

    def __init__(
        self,
        config_path: str = "./data_processing/configs/api_pdf_processing_config.yaml",
    ):
        # Load configuration
        self.config = ApiPdfProcessorConfigManager.load_config(config_path)

        # Initialize the document processor
        self.document_processor = ApiDocumentProcessor(self.config)

        # Set up logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up logging configuration for PDF processing."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/api_pdf_processing.log"),
                logging.StreamHandler(),
            ],
        )

    def process_single_document_sync(
        self,
        document_path: Path,
        task: DocumentTask = DocumentTask.EXTRACT_TEXT,
        custom_prompt: Optional[str] = None,
    ) -> ApiPdfProcessingResult:
        """Process a single document synchronously."""
        # Get document info
        document_file = DocumentFileDetector.get_document_info(document_path)
        if not document_file:
            return ApiPdfProcessingResult(
                file_path=document_path,
                document_type=DocumentType.PDF,  # Default fallback
                task=task,
                content="",
                processing_time=0.0,
                processing_mode=ProcessingMode.SYNC,
                success=False,
                error_message=f"Unsupported document type: {document_path}",
                model_used="gemini-2.5-flash",
            )

        # Process the document
        result = self.document_processor.process_document_sync(document_file, task, custom_prompt)

        # Save result if successful
        if result.success:
            output_dir = Path(self.config.paths.processed_data_dir) / self.config.paths.pdf_analysis_dir
            output_path = self.document_processor.save_result(result, output_dir)
            if output_path:
                result.output_file_path = output_path

        return result

    async def process_single_document_async(
        self,
        document_path: Path,
        task: DocumentTask = DocumentTask.EXTRACT_TEXT,
        custom_prompt: Optional[str] = None,
    ) -> ApiPdfProcessingResult:
        """Process a single document asynchronously."""
        # Get document info
        document_file = DocumentFileDetector.get_document_info(document_path)
        if not document_file:
            return ApiPdfProcessingResult(
                file_path=document_path,
                document_type=DocumentType.PDF,  # Default fallback
                task=task,
                content="",
                processing_time=0.0,
                processing_mode=ProcessingMode.ASYNC,
                success=False,
                error_message=f"Unsupported document type: {document_path}",
                model_used="gemini-2.5-flash",
            )

        # Process the document
        result = await self.document_processor.process_document_async(document_file, task, custom_prompt)

        # Save result if successful
        if result.success:
            output_dir = Path(self.config.paths.processed_data_dir) / self.config.paths.pdf_analysis_dir
            output_path = self.document_processor.save_result(result, output_dir)
            if output_path:
                result.output_file_path = output_path

        return result

    def process_batch_sync(
        self,
        base_path: Optional[Path] = None,
        task: DocumentTask = DocumentTask.EXTRACT_TEXT,
        custom_prompt: Optional[str] = None,
        max_files: Optional[int] = None,
    ) -> ApiPdfProcessingBatchResult:
        """Process all documents in a directory synchronously."""
        start_time = time.time()

        if base_path is None:
            base_path = Path(self.config.paths.raw_data_dir)

        if not base_path.exists():
            logging.error(f"Base path does not exist: {base_path}")
            return self._create_empty_result(ProcessingMode.SYNC)

        # Find all document files
        document_files = DocumentFileDetector.find_documents(base_path, recursive=True, max_files=max_files)

        if not document_files:
            logging.warning(f"No supported documents found in {base_path}")
            return self._create_empty_result(ProcessingMode.SYNC)

        # Filter documents that need processing
        output_dir = Path(self.config.paths.processed_data_dir) / self.config.paths.pdf_analysis_dir
        documents_to_process = []
        skipped_documents = 0

        for document_file in document_files:
            if self.document_processor.should_process_document(document_file, task, output_dir):
                documents_to_process.append(document_file)
            else:
                skipped_documents += 1

        if not documents_to_process:
            logging.info(f"All {len(document_files)} documents already processed, nothing to do")
            return self._create_empty_result(ProcessingMode.SYNC)

        total_documents = len(document_files)
        processed_documents = 0
        failed_documents = 0
        results = []

        logging.info(f"Starting synchronous processing of {len(documents_to_process)} documents...")
        logging.info(f"Task: {task.value}")
        logging.info(f"Model: gemini-2.5-flash")

        # Process each document with progress bar
        progress_bar = tqdm(
            documents_to_process,
            desc="Processing documents",
            unit="doc",
            total=len(documents_to_process),
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {desc}",
        )

        for document_file in progress_bar:
            try:
                # Update progress bar description with current document name
                progress_bar.set_description(f"Processing {document_file.file_path.name}")

                # Process the document
                result = self.document_processor.process_document_sync(document_file, task, custom_prompt)
                results.append(result)

                if result.success:
                    processed_documents += 1
                    # Save result
                    output_path = self.document_processor.save_result(result, output_dir)
                    if output_path:
                        result.output_file_path = output_path
                    logging.debug(f"Processed: {result.file_path.name}")
                else:
                    failed_documents += 1
                    logging.error(f"Failed: {document_file.file_path.name} - {result.error_message}")

                progress_bar.set_postfix(
                    {
                        "Processed": processed_documents,
                        "Skipped": skipped_documents,
                        "Failed": failed_documents,
                    }
                )

                # Small delay between requests to respect rate limits
                time.sleep(0.1)

            except Exception as e:
                failed_documents += 1
                progress_bar.set_postfix(
                    {
                        "Processed": processed_documents,
                        "Skipped": skipped_documents,
                        "Failed": failed_documents,
                    }
                )
                logging.error(f"Error processing {document_file.file_path}: {str(e)}")

                # Create error result
                error_result = ApiPdfProcessingResult(
                    file_path=document_file.file_path,
                    document_type=document_file.document_type,
                    task=task,
                    content="",
                    processing_time=0.0,
                    processing_mode=ProcessingMode.SYNC,
                    success=False,
                    error_message=str(e),
                    model_used="gemini-2.5-flash",
                )
                results.append(error_result)

        # Close the progress bar
        progress_bar.close()

        total_processing_time = time.time() - start_time

        # Create and return batch result
        batch_result = ApiPdfProcessingBatchResult(
            total_files=total_documents,
            processed_files=processed_documents,
            skipped_files=skipped_documents,
            failed_files=failed_documents,
            total_processing_time=total_processing_time,
            processing_mode=ProcessingMode.SYNC,
            total_api_calls=len(documents_to_process),
            results=results,
        )

        # Calculate statistics
        for result in results:
            if result.tokens_used:
                batch_result.total_tokens_used += result.tokens_used
            if result.api_cost:
                batch_result.total_api_cost += result.api_cost

        self._print_batch_summary(batch_result)
        return batch_result

    async def process_batch_async(
        self,
        base_path: Optional[Path] = None,
        task: DocumentTask = DocumentTask.EXTRACT_TEXT,
        custom_prompt: Optional[str] = None,
        max_files: Optional[int] = None,
        max_concurrent: int = 3,
    ) -> ApiPdfProcessingBatchResult:
        """Process all documents in a directory asynchronously."""
        start_time = time.time()

        if base_path is None:
            base_path = Path(self.config.paths.raw_data_dir)

        if not base_path.exists():
            logging.error(f"Base path does not exist: {base_path}")
            return self._create_empty_result(ProcessingMode.ASYNC)

        # Find all document files
        document_files = DocumentFileDetector.find_documents(base_path, recursive=True, max_files=max_files)

        if not document_files:
            logging.warning(f"No supported documents found in {base_path}")
            return self._create_empty_result(ProcessingMode.ASYNC)

        # Filter documents that need processing
        output_dir = Path(self.config.paths.processed_data_dir) / self.config.paths.pdf_analysis_dir
        documents_to_process = []
        skipped_documents = 0

        for document_file in document_files:
            if self.document_processor.should_process_document(document_file, task, output_dir):
                documents_to_process.append(document_file)
            else:
                skipped_documents += 1

        if not documents_to_process:
            logging.info(f"All {len(document_files)} documents already processed, nothing to do")
            return self._create_empty_result(ProcessingMode.ASYNC)

        total_documents = len(document_files)
        logging.info(f"Starting asynchronous processing of {len(documents_to_process)} documents...")
        logging.info(f"Task: {task.value}")
        logging.info(f"Max concurrent: {max_concurrent}")
        logging.info(f"Model: gemini-2.5-flash")

        # Use async batch processor
        batch_processor = AsyncDocumentBatchProcessor(self.document_processor, max_concurrent)

        # Process all documents asynchronously
        results = await batch_processor.process_batch(documents_to_process, task, custom_prompt)

        # Save successful results
        processed_documents = 0
        failed_documents = 0

        for result in results:
            if result.success:
                processed_documents += 1
                # Save result
                output_path = self.document_processor.save_result(result, output_dir)
                if output_path:
                    result.output_file_path = output_path
            else:
                failed_documents += 1

        total_processing_time = time.time() - start_time

        # Create and return batch result
        batch_result = ApiPdfProcessingBatchResult(
            total_files=total_documents,
            processed_files=processed_documents,
            skipped_files=skipped_documents,
            failed_files=failed_documents,
            total_processing_time=total_processing_time,
            processing_mode=ProcessingMode.ASYNC,
            total_api_calls=len(documents_to_process),
            results=results,
        )

        # Calculate statistics
        for result in results:
            if result.tokens_used:
                batch_result.total_tokens_used += result.tokens_used
            if result.api_cost:
                batch_result.total_api_cost += result.api_cost

        self._print_batch_summary(batch_result)
        return batch_result

    def process_documents_by_type(
        self,
        base_path: Optional[Path] = None,
        document_type: DocumentType = DocumentType.PDF,
        processing_mode: ProcessingMode = ProcessingMode.SYNC,
        task: DocumentTask = DocumentTask.EXTRACT_TEXT,
    ) -> ApiPdfProcessingBatchResult:
        """Process only documents of a specific type."""
        if base_path is None:
            base_path = Path(self.config.paths.raw_data_dir)

        # Find documents of specific type
        all_documents = DocumentFileDetector.find_documents(base_path, recursive=True)
        filtered_documents = [doc for doc in all_documents if doc.document_type == document_type]

        if not filtered_documents:
            logging.warning(f"No {document_type.value} documents found in {base_path}")
            return self._create_empty_result(processing_mode)

        logging.info(f"Found {len(filtered_documents)} {document_type.value} documents")

        # Process based on mode
        if processing_mode == ProcessingMode.SYNC:
            return self.process_batch_sync(base_path, task)
        else:
            return asyncio.run(self.process_batch_async(base_path, task))

    def _create_empty_result(self, processing_mode: ProcessingMode) -> ApiPdfProcessingBatchResult:
        """Create an empty batch result."""
        return ApiPdfProcessingBatchResult(
            total_files=0,
            processed_files=0,
            skipped_files=0,
            failed_files=0,
            total_processing_time=0.0,
            processing_mode=processing_mode,
            total_api_calls=0,
            results=[],
        )

    def _print_batch_summary(self, batch_result: ApiPdfProcessingBatchResult) -> None:
        """Print batch processing summary."""
        logging.info(f"\n{'='*60}")
        logging.info("API PDF/DOCUMENT PROCESSING COMPLETED")
        logging.info(f"{'='*60}")
        logging.info(f"Processing mode: {batch_result.processing_mode.value}")
        logging.info(f"Total files found: {batch_result.total_files}")
        logging.info(f"Successfully processed: {batch_result.processed_files}")
        logging.info(f"Skipped (already exist): {batch_result.skipped_files}")
        logging.info(f"Failed: {batch_result.failed_files}")
        logging.info(f"Total processing time: {batch_result.total_processing_time:.2f}s")
        logging.info(f"Total API calls: {batch_result.total_api_calls}")

        if batch_result.total_tokens_used > 0:
            logging.info(f"Total tokens used: {batch_result.total_tokens_used:,}")

        if batch_result.total_api_cost > 0:
            logging.info(f"Estimated API cost: ${batch_result.total_api_cost:.4f}")

        if batch_result.total_files > 0:
            avg_time = batch_result.total_processing_time / batch_result.total_files
            logging.info(f"Average time per document: {avg_time:.2f}s")

        # Print statistics by document type
        if batch_result.files_by_type:
            logging.info("\nFiles processed by type:")
            for doc_type, count in batch_result.files_by_type.items():
                logging.info(f"  {doc_type}: {count}")

        logging.info(f"\nResults saved in: {self.config.paths.processed_data_dir}/{self.config.paths.pdf_analysis_dir}")
        logging.info(f"{'='*60}")


def main():
    """Main entry point for the API-based PDF/document processing system."""
    try:
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)

        # Initialize processor
        processor = ApiPdfProcessingProcessor()

        print("\n=== API PDF/Document Processing ===")
        print("Processing PDF and document files using Gemini API...")

        print("\n=== Processing Options ===")
        print("1. Process All Documents (Sync)")
        print("2. Process All Documents (Async)")
        print("3. Process PDFs Only (Sync)")
        print("4. Process PDFs Only (Async)")
        print("5. Process Single Document (Sync)")
        print("6. Process Single Document (Async)")
        print("7. Extract Text from All Documents")
        print("8. Analyze Content of All Documents")
        print("9. Summarize All Documents")

        choice = input("\nSelect processing method (1-9) [default: 1]: ").strip() or "1"

        if choice == "1":
            print("\n--- Processing All Documents (Sync) ---")
            result = processor.process_batch_sync()
        elif choice == "2":
            print("\n--- Processing All Documents (Async) ---")
            result = asyncio.run(processor.process_batch_async())
        elif choice == "3":
            print("\n--- Processing PDFs Only (Sync) ---")
            result = processor.process_documents_by_type(
                document_type=DocumentType.PDF, processing_mode=ProcessingMode.SYNC
            )
        elif choice == "4":
            print("\n--- Processing PDFs Only (Async) ---")
            result = processor.process_documents_by_type(
                document_type=DocumentType.PDF, processing_mode=ProcessingMode.ASYNC
            )
        elif choice == "5":
            print("\n--- Single Document Processing (Sync) ---")
            file_path = input("Enter path to document file: ").strip()
            if file_path and Path(file_path).exists():
                result = processor.process_single_document_sync(Path(file_path))
                if result.success:
                    print(f"Content extracted: {result.content[:200]}...")
                else:
                    print(f"Processing failed: {result.error_message}")
            else:
                print("Invalid file path")
                return
        elif choice == "6":
            print("\n--- Single Document Processing (Async) ---")
            file_path = input("Enter path to document file: ").strip()
            if file_path and Path(file_path).exists():
                result = asyncio.run(processor.process_single_document_async(Path(file_path)))
                if result.success:
                    print(f"Content extracted: {result.content[:200]}...")
                else:
                    print(f"Processing failed: {result.error_message}")
            else:
                print("Invalid file path")
                return
        elif choice == "7":
            print("\n--- Extract Text from All Documents ---")
            result = processor.process_batch_sync(task=DocumentTask.EXTRACT_TEXT)
        elif choice == "8":
            print("\n--- Analyze Content of All Documents ---")
            result = processor.process_batch_sync(task=DocumentTask.ANALYZE_CONTENT)
        elif choice == "9":
            print("\n--- Summarize All Documents ---")
            result = processor.process_batch_sync(task=DocumentTask.SUMMARIZE)
        else:
            print("Invalid choice, processing all documents synchronously")
            result = processor.process_batch_sync()

        if hasattr(result, "processed_files") and result.processed_files > 0:
            print(f"\nDocument processing completed: {result.processed_files} documents processed successfully")
        else:
            print("\nNo documents were processed")

    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
