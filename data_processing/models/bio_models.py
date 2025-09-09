"""
Pydantic models for bio-data structure and validation.
"""
from datetime import datetime, date
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, EmailStr


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

    @validator('age', 'date_of_birth')
    def validate_age_consistency(cls, v, values):
        """Validate age and date of birth consistency."""
        if 'date_of_birth' in values and values['date_of_birth'] and v:
            calculated_age = (datetime.now().date() - values['date_of_birth']).days // 365
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
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
    docling_version: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    extraction_method: str = "docling"


class BioData(BaseModel):
    """Complete bio-data model."""
    personal_info: PersonalInfo
    education: Optional[Education] = None
    professional: Optional[Professional] = None
    interests: Optional[Interests] = None
    lifestyle: Optional[Lifestyle] = None
    relationship: Optional[RelationshipPreferences] = None
    metadata: ExtractionMetadata

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"
        schema_extra = {
            "example": {
                "personal_info": {
                    "name": "John Doe",
                    "age": 28,
                    "gender": "male",
                    "location": "New York, USA",
                    "contact_info": {
                        "email": "john.doe@example.com",
                        "phone": "+1-555-0123"
                    }
                },
                "education": {
                    "degree": "Bachelor of Science",
                    "institution": "MIT",
                    "graduation_year": 2018,
                    "major": "Computer Science"
                },
                "professional": {
                    "current_job": "Software Engineer",
                    "company": "Tech Corp",
                    "experience_years": 5,
                    "skills": ["Python", "JavaScript", "React"]
                }
            }
        }


class ExtractionResult(BaseModel):
    """Result of bio-data extraction."""
    success: bool
    bio_data: Optional[BioData] = None
    raw_text: Optional[str] = None
    markdown_content: Optional[str] = None
    output_path: Optional[str] = None
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None

    class Config:
        """Pydantic configuration."""
        validate_assignment = True
