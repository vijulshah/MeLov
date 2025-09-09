"""
Bio-data extraction from PDF using docling.
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
import re

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
except ImportError:
    raise ImportError("docling is required. Install with: pip install docling")

from ..models.bio_models import (
    BioData, ExtractionResult, ExtractionMetadata, PersonalInfo, 
    BioDataType
)
from ..utils.config_manager import ConfigManager


class BioDataExtractor:
    """Extract bio-data from PDF documents using docling."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize bio-data extractor.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or ConfigManager()
        self.logger = self._setup_logging()
        self.converter = self._setup_converter()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_config = self.config.get_logging_config()
        
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))
        
        # Create formatter
        formatter = logging.Formatter(
            log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        log_file = log_config.get('file')
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
        pipeline_options.do_table_structure = extraction_settings.get('extract_tables', True)
        
        # Create converter
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: pipeline_options,
            }
        )
        
        return converter
    
    def extract_from_pdf(
        self, 
        pdf_path: str, 
        bio_type: Literal["my_biodata", "ppl_biodata"] = "my_biodata"
    ) -> ExtractionResult:
        """
        Extract bio-data from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            bio_type: Type of bio data being processed
            
        Returns:
            ExtractionResult with extracted data
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        try:
            if not pdf_path.exists():
                return ExtractionResult(
                    success=False,
                    errors=[f"PDF file not found: {pdf_path}"]
                )
            
            self.logger.info(f"Starting {bio_type} extraction from: {pdf_path}")
            
            # Convert document
            result = self.converter.convert(pdf_path)
            
            # Extract text content
            raw_text = result.document.export_to_text()
            markdown_content = result.document.export_to_markdown()
            
            self.logger.info(f"Extracted {len(raw_text)} characters from PDF")
            
            # Parse bio-data from text
            bio_data = self._parse_bio_data(raw_text, pdf_path, bio_type)
            
            processing_time = time.time() - start_time
            
            # Update metadata with processing time
            if bio_data and bio_data.metadata:
                bio_data.metadata.processing_time_seconds = processing_time
            
            return ExtractionResult(
                success=True,
                bio_data=bio_data,
                raw_text=raw_text,
                markdown_content=markdown_content
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting from PDF: {e}")
            return ExtractionResult(
                success=False,
                errors=[str(e)]
            )
    
    def _parse_bio_data(
        self, 
        text: str, 
        source_file: Path, 
        bio_type: Literal["my_biodata", "ppl_biodata"]
    ) -> BioData:
        """
        Parse bio-data from extracted text.
        
        Args:
            text: Extracted text from PDF
            source_file: Source PDF file path
            bio_type: Type of bio data being processed
            
        Returns:
            Structured bio-data
        """
        # Create metadata
        metadata = ExtractionMetadata(
            source_file=str(source_file),
            bio_data_type=BioDataType(bio_type),
            processing_time_seconds=0.0  # Will be updated later
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
            metadata=metadata
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
            from ..models.bio_models import ContactInfo
            contact_info = ContactInfo(
                email=email_match.group(1) if email_match else None,
                phone=phone_match.group(1).strip() if phone_match else None
            )
        
        return PersonalInfo(
            name=name,
            age=age,
            contact_info=contact_info
        )
    
    def _extract_education(self, text: str) -> Optional[Any]:
        """Extract education information from text."""
        from ..models.bio_models import Education
        
        # Pattern matching for education
        degree_patterns = [
            r"(?:degree|Degree|DEGREE)[:\s]+([A-Za-z\s]+)",
            r"(?:bachelor|master|phd|doctorate|BA|BS|MS|MA|PhD)[\s\w]*"
        ]
        
        institution_patterns = [
            r"(?:university|college|school|University|College|School)[:\s]+([A-Za-z\s]+)",
            r"([A-Za-z\s]+(?:University|College|Institute))"
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
            return Education(
                degree=degree,
                institution=institution
            )
        
        return None
    
    def _extract_professional(self, text: str) -> Optional[Any]:
        """Extract professional information from text."""
        from ..models.bio_models import Professional
        
        job_patterns = [
            r"(?:job|position|role|title)[:\s]+([A-Za-z\s]+)",
            r"(?:work as|working as)[:\s]+([A-Za-z\s]+)"
        ]
        
        company_patterns = [
            r"(?:company|employer|organization)[:\s]+([A-Za-z\s]+)",
            r"(?:at|@)\s+([A-Za-z\s]+(?:Inc|Corp|Ltd|LLC|Company))"
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
            return Professional(
                current_job=job,
                company=company
            )
        
        return None
    
    def _extract_interests(self, text: str) -> Optional[Any]:
        """Extract interests from text."""
        from ..models.bio_models import Interests
        
        # Look for hobby/interest sections
        interest_keywords = ['hobby', 'hobbies', 'interest', 'interests', 'like', 'enjoy']
        
        for keyword in interest_keywords:
            pattern = rf"(?:{keyword})[:\s]+([A-Za-z\s,]+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                hobbies = [h.strip() for h in match.group(1).split(',')]
                return Interests(hobbies=hobbies)
        
        return None
    
    def _extract_lifestyle(self, text: str) -> Optional[Any]:
        """Extract lifestyle information from text."""
        from ..models.bio_models import Lifestyle
        
        # Simple extraction - can be enhanced
        if any(word in text.lower() for word in ['diet', 'exercise', 'smoking', 'drinking']):
            return Lifestyle()
        
        return None
    
    def _extract_relationship(self, text: str) -> Optional[Any]:
        """Extract relationship preferences from text."""
        from ..models.bio_models import RelationshipPreferences
        
        # Simple extraction - can be enhanced
        if any(word in text.lower() for word in ['single', 'married', 'relationship', 'dating']):
            return RelationshipPreferences()
        
        return None
    
    def save_to_json(
        self, 
        bio_data: BioData, 
        output_path: str, 
        bio_type: Literal["my_biodata", "ppl_biodata"] = "my_biodata"
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
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(bio_data.dict(), f, indent=2, default=str)
            
            self.logger.info(f"{bio_type.replace('_', ' ').title()} bio-data saved to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving {bio_type} bio-data: {e}")
            return False
    
    def process_pdf_file(
        self, 
        pdf_path: str, 
        bio_type: Literal["my_biodata", "ppl_biodata"] = "my_biodata",
        output_name: Optional[str] = None
    ) -> ExtractionResult:
        """
        Complete processing pipeline for a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            bio_type: Type of bio data being processed
            output_name: Custom output filename
            
        Returns:
            Extraction result
        """
        # Extract bio-data
        result = self.extract_from_pdf(pdf_path, bio_type)
        
        if result.success and result.bio_data:
            # Generate output filename
            if output_name is None:
                pdf_name = Path(pdf_path).stem
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_name = f"{pdf_name}_{bio_type}_{timestamp}.json"
            
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
    
    def process_my_biodata(self, pdf_path: str, output_name: Optional[str] = None) -> ExtractionResult:
        """
        Process user's own bio-data from PDF.
        
        Args:
            pdf_path: Path to PDF file containing user's bio-data
            output_name: Custom output filename
            
        Returns:
            Extraction result
        """
        return self.process_pdf_file(pdf_path, "my_biodata", output_name)
    
    def process_people_biodata(self, pdf_path: str, output_name: Optional[str] = None) -> ExtractionResult:
        """
        Process other people's bio-data from PDF.
        
        Args:
            pdf_path: Path to PDF file containing other person's bio-data
            output_name: Custom output filename
            
        Returns:
            Extraction result
        """
        return self.process_pdf_file(pdf_path, "ppl_biodata", output_name)
    
    def batch_process_directory(
        self, 
        bio_type: Literal["my_biodata", "ppl_biodata"],
        input_directory: Optional[str] = None
    ) -> List[ExtractionResult]:
        """
        Process all PDF files in a directory.
        
        Args:
            bio_type: Type of bio data being processed
            input_directory: Custom input directory, uses config default if None
            
        Returns:
            List of extraction results
        """
        if input_directory is None:
            input_directory = self.config.get_pdf_input_path(bio_type)
        
        input_path = Path(input_directory)
        if not input_path.exists():
            self.logger.error(f"Input directory does not exist: {input_path}")
            return []
        
        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            self.logger.warning(f"No PDF files found in: {input_path}")
            return []
        
        results = []
        self.logger.info(f"Processing {len(pdf_files)} PDF files from {input_path}")
        
        for pdf_file in pdf_files:
            self.logger.info(f"Processing: {pdf_file.name}")
            result = self.process_pdf_file(str(pdf_file), bio_type)
            results.append(result)
            
            if result.success:
                self.logger.info(f"Successfully processed: {pdf_file.name}")
            else:
                self.logger.error(f"Failed to process {pdf_file.name}: {result.errors}")
        
        return results
