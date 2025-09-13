"""Document processor for handling PDFs and images with object detection."""

import time
from pathlib import Path
from typing import List

import fitz
from models import BatchProcessingResult, Detection, FileType, ProcessingResult
from utils import (
    ConfigManager,
    DetectionVisualizer,
    DirectoryManager,
    FileDiscovery,
    ImageProcessor,
    ModelManager,
)


class DocumentProcessor:
    """Main processor for documents with object detection capabilities."""

    def __init__(self, config_path: str = "./data_processing/config.yaml"):
        self.config = ConfigManager.load_config(config_path)
        self.model_manager = ModelManager(self.config)
        self.visualizer = DetectionVisualizer(self.config)

    def process_image(self, image_path: Path) -> ProcessingResult:
        """Process a single image file."""
        start_time = time.time()

        try:
            # Load image
            image = ImageProcessor.load_image(image_path)
            if image is None:
                return self._create_error_result(image_path, FileType.IMAGE, "Failed to load image")

            # Set up output directories
            base_output_dir = DirectoryManager.get_output_base_path(image_path, self.config)
            detections_dir, cropped_dir, original_dir = DirectoryManager.create_output_structure(
                base_output_dir, self.config
            )

            # Perform object detection
            detections = self.model_manager.detect_objects(image)
            high_conf_persons = self.model_manager.filter_high_confidence_persons(detections)

            # Process detections
            self._process_detections(
                image,
                detections,
                high_conf_persons,
                detections_dir,
                cropped_dir,
                original_dir,
                image_identifier="image",
            )

            processing_time = time.time() - start_time

            return ProcessingResult(
                file_path=image_path,
                file_type=FileType.IMAGE,
                total_detections=len(detections),
                person_detections=len(
                    [d for d in detections if d.label.lower() == self.config.detection.target_label.lower()]
                ),
                high_confidence_persons=len(high_conf_persons),
                processing_time=processing_time,
                success=True,
            )

        except Exception as e:
            return self._create_error_result(image_path, FileType.IMAGE, str(e))

    def process_pdf(self, pdf_path: Path) -> ProcessingResult:
        """Process a single PDF file."""
        start_time = time.time()

        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)

            # Set up output directories
            base_output_dir = DirectoryManager.get_output_base_path(pdf_path, self.config)
            detections_dir, cropped_dir, original_dir = DirectoryManager.create_output_structure(
                base_output_dir, self.config
            )

            total_detections = 0
            total_person_detections = 0
            total_high_conf_persons = 0

            # Convert all pages to images first for batch processing
            page_images = []
            page_data = []

            print(f"  Converting {len(pdf_document)} pages to images...")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                page_image = ImageProcessor.pdf_page_to_image(page, self.config.detection.dpi_scale)
                page_images.append(page_image)
                page_data.append({"page_num": page_num, "image": page_image})

            # Process pages in batches for better GPU utilization
            batch_size = self.config.detection.batch_size
            print(f"  Processing pages in batches of {batch_size}...")

            for batch_start in range(0, len(page_images), batch_size):
                batch_end = min(batch_start + batch_size, len(page_images))
                batch_images = page_images[batch_start:batch_end]
                batch_page_data = page_data[batch_start:batch_end]

                # Perform batch object detection
                batch_detections = self.model_manager.detect_objects_batch(batch_images)

                # Process each page in the batch
                for i, (page_info, detections) in enumerate(zip(batch_page_data, batch_detections)):
                    page_num = page_info["page_num"]
                    page_image = page_info["image"]

                    high_conf_persons = self.model_manager.filter_high_confidence_persons(detections)

                    # Process detections
                    self._process_detections(
                        page_image,
                        detections,
                        high_conf_persons,
                        detections_dir,
                        cropped_dir,
                        original_dir,
                        image_identifier=f"page_{page_num + 1}",
                    )

                    # Update counters
                    total_detections += len(detections)
                    total_person_detections += len(
                        [d for d in detections if d.label.lower() == self.config.detection.target_label.lower()]
                    )
                    total_high_conf_persons += len(high_conf_persons)

                    print(
                        f"    Processed page {page_num + 1}/{len(pdf_document)} - "
                        f"{len(detections)} detections, {len(high_conf_persons)} high-conf persons"
                    )

            pdf_document.close()
            processing_time = time.time() - start_time

            return ProcessingResult(
                file_path=pdf_path,
                file_type=FileType.PDF,
                total_detections=total_detections,
                person_detections=total_person_detections,
                high_confidence_persons=total_high_conf_persons,
                processing_time=processing_time,
                success=True,
            )

        except Exception as e:
            return self._create_error_result(pdf_path, FileType.PDF, str(e))

    def _process_detections(
        self,
        image,
        detections: List[Detection],
        high_conf_persons: List[Detection],
        detections_dir: Path,
        cropped_dir: Path,
        original_dir: Path,
        image_identifier: str,
    ) -> None:
        """Process detections for a single image/page."""

        # Save original image
        original_path = original_dir / f"{image_identifier}_original.png"
        image.save(original_path)

        if detections:
            # Create and save annotated image
            annotated_image = self.visualizer.annotate_image(image, detections)
            annotated_path = detections_dir / f"{image_identifier}_with_detections.png"
            annotated_image.save(annotated_path)

            # Process high-confidence person detections
            for i, person_detection in enumerate(high_conf_persons, 1):
                cropped_person = ImageProcessor.crop_person(
                    image, person_detection, self.config.visualization.padding_ratio
                )

                crop_filename = f"{image_identifier}_person_{i}_score_{person_detection.score:.2f}.png"
                crop_path = cropped_dir / crop_filename
                cropped_person.save(crop_path)

    def _create_error_result(self, file_path: Path, file_type: FileType, error_message: str) -> ProcessingResult:
        """Create a ProcessingResult for failed processing."""
        return ProcessingResult(
            file_path=file_path,
            file_type=file_type,
            total_detections=0,
            person_detections=0,
            high_confidence_persons=0,
            processing_time=0.0,
            success=False,
            error_message=error_message,
        )

    def process_batch(self) -> BatchProcessingResult:
        """Process all files in the configured directories."""
        start_time = time.time()

        # Discover files
        pdf_files, image_files = FileDiscovery.discover_all_files(self.config)
        total_files = len(pdf_files) + len(image_files)

        print(f"Found {len(pdf_files)} PDF files and {len(image_files)} image files")
        print(f"Total files to process: {total_files}")

        if total_files == 0:
            print("No files found to process")
            return BatchProcessingResult(
                total_files=0,
                successful_files=0,
                failed_files=0,
                total_processing_time=0.0,
                results=[],
            )

        results = []
        current_file = 0

        # Process PDF files
        for pdf_path in pdf_files:
            current_file += 1
            print(f"\nProcessing PDF {current_file}/{total_files}: {pdf_path}")
            result = self.process_pdf(pdf_path)
            results.append(result)
            self._print_result(result)

        # Process image files
        for image_path in image_files:
            current_file += 1
            print(f"\nProcessing Image {current_file}/{total_files}: {image_path}")
            result = self.process_image(image_path)
            results.append(result)
            self._print_result(result)

        # Calculate final statistics
        successful_files = sum(1 for r in results if r.success)
        failed_files = total_files - successful_files
        total_processing_time = time.time() - start_time

        batch_result = BatchProcessingResult(
            total_files=total_files,
            successful_files=successful_files,
            failed_files=failed_files,
            total_processing_time=total_processing_time,
            results=results,
        )

        self._print_batch_summary(batch_result)
        return batch_result

    def _print_result(self, result: ProcessingResult) -> None:
        """Print processing result for a single file."""
        if result.success:
            print(
                f"  ✓ Success: {result.total_detections} detections, "
                f"{result.high_confidence_persons} high-conf persons, "
                f"{result.processing_time:.2f}s"
            )
        else:
            print(f"  ✗ Failed: {result.error_message}")

    def _print_batch_summary(self, batch_result: BatchProcessingResult) -> None:
        """Print batch processing summary."""
        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETED")
        print(f"{'='*60}")
        print(f"Total files: {batch_result.total_files}")
        print(f"Successful: {batch_result.successful_files}")
        print(f"Failed: {batch_result.failed_files}")
        print(f"Total processing time: {batch_result.total_processing_time:.2f}s")
        print(f"Results saved in: {self.config.paths.processed_data_dir}")
        print(f"{'='*60}")


def main():
    """Main entry point for the document processing system."""
    processor = DocumentProcessor()
    processor.process_batch()


if __name__ == "__main__":
    main()
