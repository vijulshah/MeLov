"""Pydantic models for API-based PDF and document processing using Gemini API."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types for processing."""

    PDF = "pdf"
    DOC = "doc"
    DOCX = "docx"
    TXT = "txt"
    RTF = "rtf"


class ProcessingMode(str, Enum):
    """Processing mode for API calls."""

    SYNC = "sync"
    ASYNC = "async"


class DocumentTask(str, Enum):
    """Available document processing tasks."""

    EXTRACT_TEXT = "extract_text"
    ANALYZE_CONTENT = "analyze_content"
    SUMMARIZE = "summarize"
    EXTRACT_STRUCTURED_DATA = "extract_structured_data"
    CLASSIFY = "classify"
    EXTRACT_METADATA = "extract_metadata"


class DocumentFile(BaseModel):
    """Represents a document file for processing."""

    file_path: Path = Field(description="Path to the document file")
    document_type: DocumentType = Field(description="Type of document")
    mime_type: str = Field(description="MIME type of the file")
    file_size_mb: float = Field(description="File size in megabytes")
    page_count: Optional[int] = Field(default=None, description="Number of pages in document")
    uploaded_file_uri: Optional[str] = Field(default=None, description="URI if uploaded via Files API")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional document metadata")


class ApiPdfProcessingResult(BaseModel):
    """Result from API-based PDF/document processing."""

    file_path: Path = Field(description="Path to the processed document file")
    document_type: DocumentType = Field(description="Type of document processed")
    task: DocumentTask = Field(description="Processing task performed")
    content: str = Field(description="Extracted/analyzed content")

    # Processing metadata
    processing_time: float = Field(description="Time taken for processing in seconds")
    processing_mode: ProcessingMode = Field(description="Mode used for processing (sync/async)")
    success: bool = Field(description="Whether processing was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")

    # API metadata
    model_used: str = Field(description="Model name used for processing")
    tokens_used: Optional[int] = Field(default=None, description="Number of tokens used")
    api_cost: Optional[float] = Field(default=None, description="Estimated API cost")

    # Document metadata
    file_size_mb: Optional[float] = Field(default=None, description="File size in megabytes")
    page_count: Optional[int] = Field(default=None, description="Number of pages processed")
    uploaded_file_uri: Optional[str] = Field(default=None, description="URI if uploaded via Files API")

    # Content-specific metadata
    extracted_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Structured extracted data")
    confidence_score: Optional[float] = Field(default=None, description="Confidence score if available")
    text_length: Optional[int] = Field(default=None, description="Length of extracted text")

    # Output file path
    output_file_path: Optional[Path] = Field(default=None, description="Path where processed content was saved")


class ApiPdfProcessingBatchResult(BaseModel):
    """Result from batch processing multiple PDF/document files."""

    total_files: int = Field(description="Total number of files found")
    processed_files: int = Field(description="Number of successfully processed files")
    failed_files: int = Field(description="Number of failed files")
    skipped_files: int = Field(description="Number of skipped files (already processed)")

    results: List[ApiPdfProcessingResult] = Field(default_factory=list, description="Individual processing results")

    # Batch metadata
    total_processing_time: float = Field(description="Total time for batch processing")
    processing_mode: ProcessingMode = Field(description="Mode used for batch processing")
    total_tokens_used: int = Field(default=0, description="Total tokens used in batch")
    total_api_cost: float = Field(default=0.0, description="Total estimated API cost")
    total_api_calls: int = Field(default=0, description="Total number of API calls made")

    # Statistics by document type
    files_by_type: Dict[str, int] = Field(default_factory=dict, description="Count of files processed by document type")

    def add_result(self, result: ApiPdfProcessingResult) -> None:
        """Add a result to the batch results."""
        self.results.append(result)

        if result.success:
            self.processed_files += 1
        else:
            self.failed_files += 1

        if result.tokens_used:
            self.total_tokens_used += result.tokens_used

        if result.api_cost:
            self.total_api_cost += result.api_cost

        # Update type statistics
        doc_type_str = result.document_type.value
        self.files_by_type[doc_type_str] = self.files_by_type.get(doc_type_str, 0) + 1


class DocumentProcessingConfig(BaseModel):
    """Configuration for document processing."""

    # Processing settings
    processing_mode: ProcessingMode = Field(default=ProcessingMode.SYNC, description="Default processing mode")
    default_task: DocumentTask = Field(default=DocumentTask.EXTRACT_TEXT, description="Default processing task")
    batch_size: int = Field(default=5, ge=1, le=20, description="Batch size for async processing")
    max_concurrent_requests: int = Field(default=3, ge=1, le=10, description="Max concurrent async requests")

    # File handling
    supported_extensions: List[str] = Field(
        default=[".pdf", ".doc", ".docx", ".txt", ".rtf"],
        description="Supported file extensions",
    )
    max_file_size_mb: float = Field(default=50.0, description="Maximum file size in MB")
    overwrite_existing: bool = Field(default=False, description="Whether to overwrite existing processed files")

    # Output settings
    save_extracted_text: bool = Field(default=True, description="Save extracted text to files")
    save_structured_data: bool = Field(default=True, description="Save structured data as JSON")
    output_format: str = Field(default="txt", description="Output format for extracted content")


class ApiPdfProcessingPathConfig(BaseModel):
    """Configuration for PDF processing paths."""

    raw_data_dir: str = Field(default="./data/raw", description="Directory containing source documents")
    processed_data_dir: str = Field(default="./data/processed", description="Directory for processed output")
    pdf_analysis_dir: str = Field(default="pdf_analysis", description="Directory for analysis results")


class ApiPdfProcessingOutputStructure(BaseModel):
    """Configuration for PDF processing output directory structure."""

    extracted_text_dir: str = Field(default="extracted_text", description="Directory name for extracted text files")
    analysis_results_dir: str = Field(default="analysis_results", description="Directory name for analysis results")
    structured_data_dir: str = Field(
        default="structured_data",
        description="Directory name for extracted structured data",
    )
    summaries_dir: str = Field(default="summaries", description="Directory name for document summaries")
    metadata_dir: str = Field(default="metadata", description="Directory name for extracted metadata")


class DocumentPromptConfig(BaseModel):
    """Configuration for document processing prompts."""

    default_prompts: Dict[str, str] = Field(
        default_factory=lambda: {
            "extract_text": "Extract all text content from this document. Preserve formatting and structure where possible.",
            "analyze_content": "Analyze this document and provide insights about its content, structure, purpose, and key information.",
            "summarize": "Provide a comprehensive summary of this document, highlighting the main points and key information.",
            "extract_structured_data": "Extract structured data from this document including tables, lists, key-value pairs, and other formatted information. Return as JSON where possible.",
            "classify": "Classify this document by type, topic, and purpose. Identify the main categories and themes.",
            "extract_metadata": "Extract metadata from this document including title, author, creation date, and other document properties.",
        },
        description="Default prompts for different processing tasks",
    )

    custom_prompts: Dict[str, str] = Field(default_factory=dict, description="Custom prompts for specific use cases")


class ApiPdfProcessingCompleteConfig(BaseModel):
    """Complete configuration combining all PDF processing config sections."""

    document_processing: DocumentProcessingConfig
    paths: ApiPdfProcessingPathConfig
    output_structure: ApiPdfProcessingOutputStructure
    prompts: DocumentPromptConfig


# Token cost calculators for document processing
class DocumentTokenCalculator:
    """Calculate token costs for document processing based on Gemini API specifications."""

    @staticmethod
    def calculate_pdf_tokens(page_count: int) -> int:
        """Calculate tokens for a PDF based on page count."""
        return page_count * 258  # 258 tokens per page for Gemini

    @staticmethod
    def calculate_text_tokens(text_length: int) -> int:
        """Calculate tokens for text content (rough estimate)."""
        # Rough estimate: 1 token ≈ 4 characters for English text
        return max(1, text_length // 4)

    @staticmethod
    def estimate_processing_cost(tokens_used: int, cost_per_1k_tokens: float = 0.01) -> float:
        """Estimate processing cost based on tokens used."""
        return (tokens_used / 1000) * cost_per_1k_tokens
