"""
Text processing utilities for bio data embeddings.
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from ..models.vector_models import VectorStoreConfig
from ...data_processing.models.bio_models import BioData


class BioDataTextProcessor:
    """Process bio data for embedding generation."""
    
    def __init__(self, config: VectorStoreConfig):
        """
        Initialize text processor.
        
        Args:
            config: Vector store configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Field mapping for text extraction
        self.field_processors = {
            'personal_info': self._process_personal_info,
            'education': self._process_education,
            'professional': self._process_professional,
            'interests': self._process_interests,
            'lifestyle': self._process_lifestyle,
            'relationship': self._process_relationship
        }
    
    def process_bio_data(self, bio_data: BioData, embedding_fields: Optional[Dict[str, List[str]]] = None) -> str:
        """
        Process bio data into text suitable for embeddings.
        
        Args:
            bio_data: Bio data to process
            embedding_fields: Fields to include in text
            
        Returns:
            Processed text string
        """
        if embedding_fields is None:
            # Use default fields from configuration
            embedding_fields = {
                'personal_info': ['name', 'location', 'nationality'],
                'education': ['degree', 'institution', 'major', 'certifications'],
                'professional': ['current_job', 'company', 'skills', 'industry'],
                'interests': ['hobbies', 'sports', 'music', 'travel', 'books', 'movies'],
                'lifestyle': ['diet_preferences', 'exercise_habits', 'pets'],
                'relationship': ['relationship_status', 'looking_for', 'preferences']
            }
        
        text_parts = []
        
        # Process each section
        for section_name, fields in embedding_fields.items():
            if hasattr(bio_data, section_name):
                section_data = getattr(bio_data, section_name)
                if section_data:
                    section_text = self._process_section(section_name, section_data, fields)
                    if section_text:
                        text_parts.append(section_text)
        
        # Combine all text parts
        full_text = ' '.join(text_parts)
        
        # Clean and normalize text
        processed_text = self._clean_text(full_text)
        
        # Validate text length
        if len(processed_text) < self.config.min_text_length:
            self.logger.warning(f"Processed text too short: {len(processed_text)} characters")
        elif len(processed_text) > self.config.max_text_length:
            self.logger.warning(f"Processed text too long: {len(processed_text)} characters, truncating")
            processed_text = processed_text[:self.config.max_text_length]
        
        return processed_text
    
    def _process_section(self, section_name: str, section_data: Any, fields: List[str]) -> str:
        """
        Process a specific section of bio data.
        
        Args:
            section_name: Name of the section
            section_data: Section data object
            fields: Fields to extract
            
        Returns:
            Processed section text
        """
        processor = self.field_processors.get(section_name)
        if processor:
            return processor(section_data, fields)
        else:
            return self._generic_section_processor(section_data, fields)
    
    def _process_personal_info(self, personal_info: Any, fields: List[str]) -> str:
        """Process personal information section."""
        text_parts = []
        
        if 'name' in fields and personal_info.name:
            text_parts.append(f"Name: {personal_info.name}")
        
        if 'age' in fields and personal_info.age:
            text_parts.append(f"Age: {personal_info.age}")
        
        if 'gender' in fields and personal_info.gender:
            text_parts.append(f"Gender: {personal_info.gender}")
        
        if 'location' in fields and personal_info.location:
            text_parts.append(f"Location: {personal_info.location}")
        
        if 'nationality' in fields and personal_info.nationality:
            text_parts.append(f"Nationality: {personal_info.nationality}")
        
        # Contact information
        if personal_info.contact_info:
            if 'contact_info' in fields:
                if personal_info.contact_info.email:
                    text_parts.append(f"Email: {personal_info.contact_info.email}")
                if personal_info.contact_info.linkedin:
                    text_parts.append(f"LinkedIn: {personal_info.contact_info.linkedin}")
        
        return ' '.join(text_parts)
    
    def _process_education(self, education: Any, fields: List[str]) -> str:
        """Process education section."""
        if not education:
            return ""
        
        text_parts = []
        
        if 'degree' in fields and education.degree:
            text_parts.append(f"Degree: {education.degree}")
        
        if 'institution' in fields and education.institution:
            text_parts.append(f"Institution: {education.institution}")
        
        if 'major' in fields and education.major:
            text_parts.append(f"Major: {education.major}")
        
        if 'graduation_year' in fields and education.graduation_year:
            text_parts.append(f"Graduation Year: {education.graduation_year}")
        
        if 'gpa' in fields and education.gpa:
            text_parts.append(f"GPA: {education.gpa}")
        
        if 'certifications' in fields and education.certifications:
            certs_text = ', '.join(education.certifications)
            text_parts.append(f"Certifications: {certs_text}")
        
        return ' '.join(text_parts)
    
    def _process_professional(self, professional: Any, fields: List[str]) -> str:
        """Process professional section."""
        if not professional:
            return ""
        
        text_parts = []
        
        if 'current_job' in fields and professional.current_job:
            text_parts.append(f"Job: {professional.current_job}")
        
        if 'company' in fields and professional.company:
            text_parts.append(f"Company: {professional.company}")
        
        if 'industry' in fields and professional.industry:
            text_parts.append(f"Industry: {professional.industry}")
        
        if 'experience_years' in fields and professional.experience_years:
            text_parts.append(f"Experience: {professional.experience_years} years")
        
        if 'skills' in fields and professional.skills:
            skills_text = ', '.join(professional.skills)
            text_parts.append(f"Skills: {skills_text}")
        
        if 'salary_range' in fields and professional.salary_range:
            text_parts.append(f"Salary Range: {professional.salary_range}")
        
        return ' '.join(text_parts)
    
    def _process_interests(self, interests: Any, fields: List[str]) -> str:
        """Process interests section."""
        if not interests:
            return ""
        
        text_parts = []
        
        interest_fields = {
            'hobbies': interests.hobbies,
            'sports': interests.sports,
            'music': interests.music,
            'travel': interests.travel,
            'books': interests.books,
            'movies': interests.movies
        }
        
        for field_name, field_value in interest_fields.items():
            if field_name in fields and field_value:
                items_text = ', '.join(field_value)
                text_parts.append(f"{field_name.title()}: {items_text}")
        
        return ' '.join(text_parts)
    
    def _process_lifestyle(self, lifestyle: Any, fields: List[str]) -> str:
        """Process lifestyle section."""
        if not lifestyle:
            return ""
        
        text_parts = []
        
        if 'diet_preferences' in fields and lifestyle.diet_preferences:
            diet_text = ', '.join([str(d) for d in lifestyle.diet_preferences])
            text_parts.append(f"Diet: {diet_text}")
        
        if 'exercise_habits' in fields and lifestyle.exercise_habits:
            text_parts.append(f"Exercise: {lifestyle.exercise_habits}")
        
        if 'smoking' in fields and lifestyle.smoking is not None:
            smoking_text = "Yes" if lifestyle.smoking else "No"
            text_parts.append(f"Smoking: {smoking_text}")
        
        if 'drinking' in fields and lifestyle.drinking:
            text_parts.append(f"Drinking: {lifestyle.drinking}")
        
        if 'pets' in fields and lifestyle.pets:
            pets_text = ', '.join(lifestyle.pets)
            text_parts.append(f"Pets: {pets_text}")
        
        return ' '.join(text_parts)
    
    def _process_relationship(self, relationship: Any, fields: List[str]) -> str:
        """Process relationship section."""
        if not relationship:
            return ""
        
        text_parts = []
        
        if 'relationship_status' in fields and relationship.relationship_status:
            text_parts.append(f"Status: {relationship.relationship_status}")
        
        if 'looking_for' in fields and relationship.looking_for:
            looking_text = ', '.join(relationship.looking_for)
            text_parts.append(f"Looking for: {looking_text}")
        
        if 'preferences' in fields and relationship.preferences:
            # Process preferences dict
            pref_items = []
            for key, value in relationship.preferences.items():
                if isinstance(value, list):
                    value = ', '.join(str(v) for v in value)
                pref_items.append(f"{key}: {value}")
            if pref_items:
                text_parts.append(f"Preferences: {'; '.join(pref_items)}")
        
        if 'deal_breakers' in fields and relationship.deal_breakers:
            breakers_text = ', '.join(relationship.deal_breakers)
            text_parts.append(f"Deal breakers: {breakers_text}")
        
        return ' '.join(text_parts)
    
    def _generic_section_processor(self, section_data: Any, fields: List[str]) -> str:
        """Generic processor for unknown sections."""
        text_parts = []
        
        for field in fields:
            if hasattr(section_data, field):
                value = getattr(section_data, field)
                if value:
                    if isinstance(value, list):
                        value = ', '.join(str(v) for v in value)
                    text_parts.append(f"{field.replace('_', ' ').title()}: {value}")
        
        return ' '.join(text_parts)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,;:!?()-]', '', text)
        
        # Normalize case for certain patterns
        text = re.sub(r'\b(email|phone|linkedin)\b', lambda m: m.group().title(), text, flags=re.IGNORECASE)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks for embedding.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.config.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.chunk_size
            
            # If this is not the last chunk, try to break at a word boundary
            if end < len(text):
                # Find the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position considering overlap
            start = end - self.config.chunk_overlap
            if start <= 0:
                start = end
        
        return chunks
    
    def extract_metadata_text(self, bio_data: BioData) -> Dict[str, str]:
        """
        Extract metadata text for each section.
        
        Args:
            bio_data: Bio data to process
            
        Returns:
            Dictionary mapping section names to text
        """
        metadata_text = {}
        
        sections = {
            'personal_info': bio_data.personal_info,
            'education': bio_data.education,
            'professional': bio_data.professional,
            'interests': bio_data.interests,
            'lifestyle': bio_data.lifestyle,
            'relationship': bio_data.relationship
        }
        
        for section_name, section_data in sections.items():
            if section_data:
                # Get all available fields for this section
                if section_name == 'personal_info':
                    fields = ['name', 'age', 'gender', 'location', 'nationality', 'contact_info']
                elif section_name == 'education':
                    fields = ['degree', 'institution', 'major', 'graduation_year', 'gpa', 'certifications']
                elif section_name == 'professional':
                    fields = ['current_job', 'company', 'industry', 'experience_years', 'skills', 'salary_range']
                elif section_name == 'interests':
                    fields = ['hobbies', 'sports', 'music', 'travel', 'books', 'movies']
                elif section_name == 'lifestyle':
                    fields = ['diet_preferences', 'exercise_habits', 'smoking', 'drinking', 'pets']
                elif section_name == 'relationship':
                    fields = ['relationship_status', 'looking_for', 'preferences', 'deal_breakers']
                else:
                    fields = []
                
                section_text = self._process_section(section_name, section_data, fields)
                if section_text:
                    metadata_text[section_name] = section_text
        
        return metadata_text
