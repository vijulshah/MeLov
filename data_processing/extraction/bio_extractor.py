"""
Bio-data extraction from multiple file formats using docling and image processing.
"""

import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, List, Literal, Optional

try:
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter
except ImportError:
    raise ImportError("docling is required. Install with: pip install docling")

# Handle imports for both script and module execution
try:
    # When run as module: python -m data_processing.extraction.main
    from ..models.bio_models import (
        BioData,
        BioDataType,
        ExtractionMetadata,
        ExtractionResult,
        FileType,
        ImageInfo,
        PersonalInfo,
    )
    from ..utils.config_manager import ConfigManager
    from ..utils.image_processor import ImageProcessor
except ImportError:
    # When run as script or imported from script
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.bio_models import (
        BioData,
        BioDataType,
        ExtractionMetadata,
        ExtractionResult,
        FileType,
        ImageInfo,
        PersonalInfo,
    )
    from utils.config_manager import ConfigManager
    from utils.image_processor import ImageProcessor


class BioDataExtractor:
    """Extract bio-data from multiple file formats using docling and process images."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize bio-data extractor.

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or ConfigManager()
        self.logger = self._setup_logging()
        self.converter = self._setup_converter()

        # Initialize image processor
        try:
            self.image_processor = ImageProcessor(self.config)
        except ImportError as e:
            self.logger.warning(f"Image processing not available: {e}")
            self.image_processor = None

        # File format mappings
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
        self.document_extensions = {".pdf", ".docx", ".doc"}
        self.supported_extensions = self.image_extensions | self.document_extensions

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_config = self.config.get_logging_config()

        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, log_config.get("level", "INFO")))

        # Create formatter
        formatter = logging.Formatter(log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if specified
        log_file = log_config.get("file")
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _setup_converter(self) -> DocumentConverter:
        """Setup docling document converter."""
        extraction_settings = self.config.get_extraction_settings()

        # Configure PDF pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True

        # Set table extraction if available
        if hasattr(pipeline_options, "do_table_structure"):
            pipeline_options.do_table_structure = extraction_settings.get("extract_tables", True)

        # Set image extraction if available
        if hasattr(pipeline_options, "generate_page_images"):
            pipeline_options.generate_page_images = extraction_settings.get("extract_images", True)

        # Create converter with support for multiple formats
        converter = DocumentConverter(
            # format_options={
            #     InputFormat.PDF: pipeline_options,
            #     InputFormat.DOCX: {},  # Default options for DOCX
            #     InputFormat.IMAGE: {},  # Default options for images
            # }
        )

        return converter

    def _get_file_type(self, file_path: Path) -> FileType:
        """
        Determine file type from extension.

        Args:
            file_path: Path to file

        Returns:
            FileType enum value
        """
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return FileType.PDF
        elif ext in self.image_extensions:
            return FileType.IMAGE
        elif ext in {".docx", ".doc"}:
            return FileType.DOCX
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _extract_images_from_pdf(
        self, pdf_path: Path, bio_type: Literal["my_biodata", "ppl_biodata"]
    ) -> List[ImageInfo]:
        """
        Extract images from PDF document.

        Args:
            pdf_path: Path to PDF file
            bio_type: Type of bio data being processed

        Returns:
            List of extracted image information
        """
        extracted_images = []

        try:
            # Convert document to extract images
            result = self.converter.convert(pdf_path)

            # Check if images were extracted
            if hasattr(result.document, "pictures") and result.document.pictures:
                images_output_dir = Path(self.config.get_images_output_path(bio_type))
                images_output_dir.mkdir(parents=True, exist_ok=True)

                # Create subdirectory for this PDF
                pdf_name = pdf_path.stem
                pdf_images_dir = images_output_dir / pdf_name
                pdf_images_dir.mkdir(exist_ok=True)

                for i, picture in enumerate(result.document.pictures):
                    try:
                        # Save extracted image
                        image_filename = f"{pdf_name}_page_{picture.page}_img_{i}.png"
                        image_path = pdf_images_dir / image_filename

                        # Convert and save image
                        if hasattr(picture, "image") and picture.image:
                            picture.image.save(image_path)

                            # Process with image processor if available
                            if self.image_processor:
                                image_info = self.image_processor.process_image(
                                    str(image_path),
                                    extracted_from_pdf=True,
                                    page_number=picture.page,
                                    original_filename=pdf_path.name,
                                )
                                extracted_images.append(image_info)
                            else:
                                # Create basic ImageInfo without AI processing
                                image_info = ImageInfo(
                                    file_path=str(image_path),
                                    original_filename=pdf_path.name,
                                    extracted_from_pdf=True,
                                    page_number=picture.page,
                                )
                                extracted_images.append(image_info)

                            self.logger.info(f"Extracted image: {image_filename}")

                    except Exception as e:
                        self.logger.error(f"Error saving extracted image {i}: {e}")

        except Exception as e:
            self.logger.error(f"Error extracting images from PDF: {e}")

        return extracted_images

    def extract_from_file(
        self,
        file_path: str,
        bio_type: Literal["my_biodata", "ppl_biodata"] = "my_biodata",
    ) -> ExtractionResult:
        """
        Extract bio-data from file (PDF, image, or document).

        Args:
            file_path: Path to file
            bio_type: Type of bio data being processed

        Returns:
            ExtractionResult with extracted data
        """
        start_time = time.time()
        file_path = Path(file_path)

        try:
            if not file_path.exists():
                return ExtractionResult(success=False, errors=[f"File not found: {file_path}"])

            # Determine file type
            file_type = self._get_file_type(file_path)

            self.logger.info(f"Starting {bio_type} extraction from {file_type.value}: {file_path}")

            extracted_images = []
            raw_text = ""
            markdown_content = ""

            if file_type == FileType.IMAGE:
                # Handle standalone image
                extracted_images = self._process_standalone_image(file_path, bio_type)
                raw_text = f"Standalone image file: {file_path.name}"

            else:
                # Handle documents (PDF, DOCX)
                try:
                    # Convert document
                    result = self.converter.convert(file_path)

                    # Extract text content
                    raw_text = result.document.export_to_text()
                    markdown_content = result.document.export_to_markdown()

                    self.logger.info(f"Extracted {len(raw_text)} characters from {file_type.value}")

                    # Extract images if this is a PDF and image extraction is enabled
                    if file_type == FileType.PDF and self.config.get_extraction_settings().get("extract_images", True):
                        extracted_images = self._extract_images_from_pdf(file_path, bio_type)

                except Exception as e:
                    self.logger.error(f"Error converting document: {e}")
                    return ExtractionResult(success=False, errors=[f"Error converting document: {e}"])

            # Parse bio-data from text
            bio_data = self._parse_bio_data(raw_text, file_path, bio_type, file_type, extracted_images)

            processing_time = time.time() - start_time

            # Update metadata with processing time and image stats
            if bio_data and bio_data.metadata:
                bio_data.metadata.processing_time_seconds = processing_time
                bio_data.metadata.images_extracted = len(extracted_images)
                bio_data.metadata.images_with_persons = sum(1 for img in extracted_images if img.contains_person)

            return ExtractionResult(
                success=True,
                bio_data=bio_data,
                raw_text=raw_text,
                markdown_content=markdown_content,
                extracted_images=extracted_images,
            )

        except Exception as e:
            self.logger.error(f"Error extracting from file: {e}")
            return ExtractionResult(success=False, errors=[str(e)])

    def _process_standalone_image(
        self, image_path: Path, bio_type: Literal["my_biodata", "ppl_biodata"]
    ) -> List[ImageInfo]:
        """
        Process a standalone image file.

        Args:
            image_path: Path to image file
            bio_type: Type of bio data being processed

        Returns:
            List containing the processed image info
        """
        try:
            # Copy image to processed location
            images_output_dir = Path(self.config.get_images_output_path(bio_type))
            images_output_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with timestamp to avoid conflicts
            timestamp = int(time.time())
            output_filename = f"{image_path.stem}_{timestamp}{image_path.suffix}"
            output_path = images_output_dir / output_filename

            shutil.copy2(image_path, output_path)

            # Process with image processor if available
            if self.image_processor:
                image_info = self.image_processor.process_image(
                    str(output_path),
                    extracted_from_pdf=False,
                    original_filename=image_path.name,
                )
            else:
                # Create basic ImageInfo without AI processing
                image_info = ImageInfo(
                    file_path=str(output_path),
                    original_filename=image_path.name,
                    extracted_from_pdf=False,
                )

            return [image_info]

        except Exception as e:
            self.logger.error(f"Error processing standalone image: {e}")
            return []

    def _parse_bio_data(
        self,
        text: str,
        source_file: Path,
        bio_type: Literal["my_biodata", "ppl_biodata"],
        file_type: FileType,
        extracted_images: List[ImageInfo] = None,
    ) -> BioData:
        """
        Parse bio-data from extracted text.

        Args:
            text: Extracted text from file
            source_file: Source file path
            bio_type: Type of bio data being processed
            file_type: Type of source file
            extracted_images: List of extracted/processed images

        Returns:
            Structured bio-data
        """
        # Create metadata
        metadata = ExtractionMetadata(
            source_file=str(source_file),
            bio_data_type=BioDataType(bio_type),
            file_type=file_type,
            processing_time_seconds=0.0,  # Will be updated later
            images_extracted=len(extracted_images) if extracted_images else 0,
            images_with_persons=sum(1 for img in (extracted_images or []) if img.contains_person),
        )

        # Extract personal information
        personal_info = self._extract_personal_info(text)

        # Create bio-data object
        bio_data = BioData(
            personal_info=personal_info,
            education=self._extract_education(text),
            professional=self._extract_professional(text),
            interests=self._extract_interests(text),
            lifestyle=self._extract_lifestyle(text),
            relationship=self._extract_relationship(text),
            images=extracted_images or [],
            metadata=metadata,
        )

        return bio_data

    def _extract_personal_info(self, text: str) -> PersonalInfo:
        """Extract personal information from text."""
        # Basic patterns for extraction
        name_pattern = r"(?:name|Name|NAME)[:\s]+([A-Za-z\s]+)"
        age_pattern = r"(?:age|Age|AGE)[:\s]+(\d+)"
        email_pattern = r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
        phone_pattern = r"(?:phone|Phone|PHONE|mobile|Mobile)[:\s]*([\+\d\s\-\(\)]+)"

        # Extract basic information
        name_match = re.search(name_pattern, text)
        age_match = re.search(age_pattern, text)
        email_match = re.search(email_pattern, text)
        phone_match = re.search(phone_pattern, text)

        # Build personal info
        name = name_match.group(1).strip() if name_match else "Unknown"
        age = int(age_match.group(1)) if age_match else None

        contact_info = None
        if email_match or phone_match:
            try:
                # When run as module
                from ..models.bio_models import ContactInfo
            except ImportError:
                # When run as script
                from models.bio_models import ContactInfo

            contact_info = ContactInfo(
                email=email_match.group(1) if email_match else None,
                phone=phone_match.group(1).strip() if phone_match else None,
            )

        return PersonalInfo(name=name, age=age, contact_info=contact_info)

    def _extract_education(self, text: str) -> Optional[Any]:
        """Extract education information from text."""
        try:
            # When run as module
            from ..models.bio_models import Education
        except ImportError:
            # When run as script
            from models.bio_models import Education

        # Pattern matching for education
        degree_patterns = [
            r"(?:degree|Degree|DEGREE)[:\s]+([A-Za-z\s]+)",
            r"(?:bachelor|master|phd|doctorate|BA|BS|MS|MA|PhD)[\s\w]*",
        ]

        institution_patterns = [
            r"(?:university|college|school|University|College|School)[:\s]+([A-Za-z\s]+)",
            r"([A-Za-z\s]+(?:University|College|Institute))",
        ]

        # Extract education data
        degree = None
        institution = None

        for pattern in degree_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                degree = match.group(1).strip()
                break

        for pattern in institution_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                institution = match.group(1).strip()
                break

        if degree or institution:
            return Education(degree=degree, institution=institution)

        return None

    def _extract_professional(self, text: str) -> Optional[Any]:
        """Extract professional information from text."""
        try:
            # When run as module
            from ..models.bio_models import Professional
        except ImportError:
            # When run as script
            from models.bio_models import Professional

        job_patterns = [
            r"(?:job|position|role|title)[:\s]+([A-Za-z\s]+)",
            r"(?:work as|working as)[:\s]+([A-Za-z\s]+)",
        ]

        company_patterns = [
            r"(?:company|employer|organization)[:\s]+([A-Za-z\s]+)",
            r"(?:at|@)\s+([A-Za-z\s]+(?:Inc|Corp|Ltd|LLC|Company))",
        ]

        # Extract professional data
        job = None
        company = None

        for pattern in job_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                job = match.group(1).strip()
                break

        for pattern in company_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                company = match.group(1).strip()
                break

        if job or company:
            return Professional(current_job=job, company=company)

        return None

    def _extract_interests(self, text: str) -> Optional[Any]:
        """Extract interests from text."""
        try:
            # When run as module
            from ..models.bio_models import Interests
        except ImportError:
            # When run as script
            from models.bio_models import Interests

        # Look for hobby/interest sections
        interest_keywords = [
            "hobby",
            "hobbies",
            "interest",
            "interests",
            "like",
            "enjoy",
        ]

        for keyword in interest_keywords:
            pattern = rf"(?:{keyword})[:\s]+([A-Za-z\s,]+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                hobbies = [h.strip() for h in match.group(1).split(",")]
                return Interests(hobbies=hobbies)

        return None

    def _extract_lifestyle(self, text: str) -> Optional[Any]:
        """Extract lifestyle information from text."""
        try:
            # When run as module
            from ..models.bio_models import Lifestyle
        except ImportError:
            # When run as script
            from models.bio_models import Lifestyle

        # Simple extraction - can be enhanced
        if any(word in text.lower() for word in ["diet", "exercise", "smoking", "drinking"]):
            return Lifestyle()

        return None

    def _extract_relationship(self, text: str) -> Optional[Any]:
        """Extract relationship preferences from text."""
        try:
            # When run as module
            from ..models.bio_models import RelationshipPreferences
        except ImportError:
            # When run as script
            from models.bio_models import RelationshipPreferences

        # Simple extraction - can be enhanced
        if any(word in text.lower() for word in ["single", "married", "relationship", "dating"]):
            return RelationshipPreferences()

        return None

    def save_to_json(
        self,
        bio_data: BioData,
        output_path: str,
        bio_type: Literal["my_biodata", "ppl_biodata"] = "my_biodata",
    ) -> bool:
        """
        Save bio-data to JSON file.

        Args:
            bio_data: Bio-data to save
            output_path: Output file path
            bio_type: Type of bio data being saved

        Returns:
            True if successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(bio_data.dict(), f, indent=2, default=str)

            self.logger.info(f"{bio_type.replace('_', ' ').title()} bio-data saved to: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving {bio_type} bio-data: {e}")
            return False

    def process_file(
        self,
        file_path: str,
        bio_type: Literal["my_biodata", "ppl_biodata"] = "my_biodata",
        output_name: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Complete processing pipeline for any supported file.

        Args:
            file_path: Path to file
            bio_type: Type of bio data being processed
            output_name: Custom output filename

        Returns:
            Extraction result
        """
        # Extract bio-data
        result = self.extract_from_file(file_path, bio_type)

        if result.success and result.bio_data:
            # Generate output filename
            if output_name is None:
                file_name = Path(file_path).stem
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_name = f"{file_name}_{bio_type}_{timestamp}.json"

            output_path = Path(self.config.get_output_path(bio_type)) / output_name

            # Save to JSON
            if self.save_to_json(result.bio_data, output_path, bio_type):
                self.logger.info(f"Processing completed successfully: {output_path}")
                # Add output path to result for reference
                result.output_path = str(output_path)
            else:
                result.errors = result.errors or []
                result.errors.append("Failed to save bio-data to JSON")

        return result

    def process_pdf_file(
        self,
        pdf_path: str,
        bio_type: Literal["my_biodata", "ppl_biodata"] = "my_biodata",
        output_name: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Complete processing pipeline for a PDF file (backward compatibility).

        Args:
            pdf_path: Path to PDF file
            bio_type: Type of bio data being processed
            output_name: Custom output filename

        Returns:
            Extraction result
        """
        return self.process_file(pdf_path, bio_type, output_name)

    def process_my_biodata(self, file_path: str, output_name: Optional[str] = None) -> ExtractionResult:
        """
        Process user's own bio-data from any supported file.

        Args:
            file_path: Path to file containing user's bio-data
            output_name: Custom output filename

        Returns:
            Extraction result
        """
        return self.process_file(file_path, BioDataType.MY_BIODATA, output_name)

    def process_people_biodata(self, file_path: str, output_name: Optional[str] = None) -> ExtractionResult:
        """
        Process other people's bio-data from any supported file.

        Args:
            file_path: Path to file containing other person's bio-data
            output_name: Custom output filename

        Returns:
            Extraction result
        """
        return self.process_file(file_path, BioDataType.PPL_BIODATA, output_name)

    def batch_process_directory(
        self,
        bio_type: Literal[BioDataType.MY_BIODATA, BioDataType.PPL_BIODATA] = BioDataType.PPL_BIODATA,
        batch_size: int = 1,
    ) -> List[ExtractionResult]:
        """
        Process all supported files in a directory.

        Args:
            bio_type: Type of bio data being processed
            input_directory: Custom input directory, uses config default if None

        Returns:
            List of extraction results
        """
        input_directory = self.config.get_file_input_path(bio_type)

        input_path = Path(input_directory)
        if not input_path.exists():
            self.logger.error(f"Input directory does not exist: {input_path}")
            return []

        # Find all supported files
        supported_files = []
        for ext in self.supported_extensions:
            pattern = f"*{ext}"
            supported_files.extend(input_path.glob(pattern))
            # Also check subdirectories
            supported_files.extend(input_path.rglob(pattern))

        # Remove duplicates and sort
        supported_files = sorted(set(supported_files))

        if not supported_files:
            self.logger.warning(f"No supported files found in: {input_path}")
            return []

        results = []
        self.logger.info(f"Processing {len(supported_files)} files from {input_path}")

        for file_path in supported_files:
            self.logger.info(f"Processing: {file_path.name}")
            result = self.process_file(str(file_path), bio_type)
            results.append(result)

            if result.success:
                self.logger.info(f"Successfully processed: {file_path.name}")
                if result.extracted_images:
                    self.logger.info(f"  - Extracted {len(result.extracted_images)} images")
                    person_count = sum(1 for img in result.extracted_images if img.contains_person)
                    if person_count > 0:
                        self.logger.info(f"  - Found {person_count} images with persons")
            else:
                self.logger.error(f"Failed to process {file_path.name}: {result.errors}")

        return results

    def is_supported_file(self, file_path: str) -> bool:
        """
        Check if file format is supported.

        Args:
            file_path: Path to file

        Returns:
            True if file format is supported
        """
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return sorted(list(self.supported_extensions))

    def cleanup(self):
        """Cleanup resources."""
        if self.image_processor:
            self.image_processor.cleanup()
