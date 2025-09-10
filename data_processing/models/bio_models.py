"""
Pydantic models for bio-data structure and validation.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class BioDataType(str, Enum):
    """Bio data type enumeration."""

    MY_BIODATA = "my_biodata"
    PPL_BIODATA = "ppl_biodata"


class Gender(str, Enum):
    """Gender enumeration."""

    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non_binary"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


class RelationshipStatus(str, Enum):
    """Relationship status enumeration."""

    SINGLE = "single"
    DATING = "dating"
    IN_RELATIONSHIP = "in_relationship"
    MARRIED = "married"
    DIVORCED = "divorced"
    WIDOWED = "widowed"


class DietPreference(str, Enum):
    """Diet preference enumeration."""

    OMNIVORE = "omnivore"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    PESCATARIAN = "pescatarian"
    KETO = "keto"
    PALEO = "paleo"


class FileType(str, Enum):
    """Supported file types for bio data extraction."""

    PDF = "pdf"
    IMAGE = "image"
    DOCX = "docx"
    DOC = "doc"


class ImageInfo(BaseModel):
    """Information about extracted or processed images."""

    file_path: str = Field(..., description="Path to the image file")
    original_filename: Optional[str] = None
    extracted_from_pdf: bool = Field(False, description="Whether image was extracted from PDF")
    page_number: Optional[int] = Field(None, description="PDF page number if extracted from PDF")
    contains_person: Optional[bool] = Field(None, description="Whether image contains a person")
    person_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    description: Optional[str] = Field(None, description="AI-generated description of the image")
    description_model: Optional[str] = Field(None, description="Model used for image description")
    width: Optional[int] = None
    height: Optional[int] = None
    file_size_bytes: Optional[int] = None


class ContactInfo(BaseModel):
    """Contact information model."""

    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    social_media: Optional[Dict[str, str]] = None


class PersonalInfo(BaseModel):
    """Personal information model."""

    name: str = Field(..., min_length=1, max_length=100)
    age: Optional[int] = Field(None, ge=18, le=100)
    date_of_birth: Optional[date] = None
    gender: Optional[Gender] = None
    location: Optional[str] = None
    nationality: Optional[str] = None
    contact_info: Optional[ContactInfo] = None

    @field_validator("age")
    @classmethod
    def validate_age_consistency(cls, v, info):
        """Validate age and date of birth consistency."""
        if info.data.get("date_of_birth") and v:
            calculated_age = (datetime.now().date() - info.data["date_of_birth"]).days // 365
            if abs(calculated_age - v) > 1:  # Allow 1 year difference
                raise ValueError("Age and date of birth are inconsistent")
        return v


class Education(BaseModel):
    """Education model."""

    degree: Optional[str] = None
    institution: Optional[str] = None
    graduation_year: Optional[int] = Field(None, ge=1950, le=2030)
    gpa: Optional[float] = Field(None, ge=0.0, le=4.0)
    major: Optional[str] = None
    certifications: Optional[List[str]] = None


class Professional(BaseModel):
    """Professional information model."""

    current_job: Optional[str] = None
    company: Optional[str] = None
    experience_years: Optional[int] = Field(None, ge=0, le=50)
    skills: Optional[List[str]] = None
    industry: Optional[str] = None
    salary_range: Optional[str] = None


class Interests(BaseModel):
    """Interests and hobbies model."""

    hobbies: Optional[List[str]] = None
    sports: Optional[List[str]] = None
    music: Optional[List[str]] = None
    travel: Optional[List[str]] = None
    books: Optional[List[str]] = None
    movies: Optional[List[str]] = None


class Lifestyle(BaseModel):
    """Lifestyle preferences model."""

    diet_preferences: Optional[List[DietPreference]] = None
    exercise_habits: Optional[str] = None
    smoking: Optional[bool] = None
    drinking: Optional[str] = None
    pets: Optional[List[str]] = None


class RelationshipPreferences(BaseModel):
    """Relationship preferences model."""

    relationship_status: Optional[RelationshipStatus] = None
    looking_for: Optional[List[str]] = None
    preferences: Optional[Dict[str, Any]] = None
    deal_breakers: Optional[List[str]] = None


class ExtractionMetadata(BaseModel):
    """Metadata for extraction process."""

    source_file: str
    bio_data_type: BioDataType = Field(..., description="Type of bio data being processed")
    file_type: FileType = Field(..., description="Type of source file")
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
    docling_version: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    extraction_method: str = "docling"
    images_extracted: int = Field(0, description="Number of images extracted/processed")
    images_with_persons: int = Field(0, description="Number of images containing persons")


class BioData(BaseModel):
    """Complete bio-data model."""

    personal_info: PersonalInfo
    education: Optional[Education] = None
    professional: Optional[Professional] = None
    interests: Optional[Interests] = None
    lifestyle: Optional[Lifestyle] = None
    relationship: Optional[RelationshipPreferences] = None
    images: Optional[List[ImageInfo]] = Field(default_factory=list, description="Associated images")
    metadata: ExtractionMetadata


class ExtractionResult(BaseModel):
    """Result of bio-data extraction."""

    success: bool
    bio_data: Optional[BioData] = None
    raw_text: Optional[str] = None
    markdown_content: Optional[str] = None
    output_path: Optional[str] = None
    extracted_images: Optional[List[ImageInfo]] = Field(default_factory=list)
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


class FileProcessingConfig(BaseModel):
    """File processing configuration for a specific bio data type."""

    input_path: str
    output_path: str
    images_output_path: str


class FileProcessingSettings(BaseModel):
    """File processing settings."""

    my_biodata: FileProcessingConfig
    ppl_biodata: FileProcessingConfig
    supported_formats: List[str] = Field(default=["pdf", "jpg", "jpeg", "png", "bmp", "tiff", "gif", "docx", "doc"])


class ExtractionSettings(BaseModel):
    """Extraction settings for document processing."""

    extract_images: bool = True
    extract_tables: bool = True
    preserve_formatting: bool = True
    save_extracted_images: bool = True
    image_formats: List[str] = Field(default=["png", "jpg", "jpeg"])
    max_image_size_mb: int = Field(10, ge=1, le=100)


class PersonDetectionConfig(BaseModel):
    """Person detection model configuration."""

    model: str = "facebook/detr-resnet-50"
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    device: str = "auto"


class ImageDescriptionConfig(BaseModel):
    """Image description model configuration."""

    model: str = "Salesforce/blip-image-captioning-large"
    max_length: int = Field(150, ge=50, le=500)
    device: str = "auto"


class ImageProcessingConfig(BaseModel):
    """Image processing configuration."""

    person_detection: PersonDetectionConfig
    image_description: ImageDescriptionConfig
    cache_dir: str = "local/hf_cache"
    use_cache: bool = True


class BioDataFields(BaseModel):
    """Bio data fields configuration."""

    personal_info: List[str]
    education: List[str]
    professional: List[str]
    interests: List[str]
    lifestyle: List[str]
    relationship: List[str]


class OutputConfig(BaseModel):
    """Output configuration."""

    format: str = "json"
    include_metadata: bool = True
    structured_output: bool = True
    validate_schema: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    file: str = "logs/bio_extraction.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class DataProcessingConfig(BaseModel):
    """Configuration model for data processing."""

    batch_processing: bool = True
    batch_size: int = Field(10, ge=1, le=100)
    processing_bio_types: List[BioDataType] = Field(default=[BioDataType.MY_BIODATA, BioDataType.PPL_BIODATA])
    file_processing: FileProcessingSettings
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    image_processing: ImageProcessingConfig
    bio_data_fields: BioDataFields
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
