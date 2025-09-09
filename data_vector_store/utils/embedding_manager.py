"""
Embedding generation and management for bio data.
"""
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    raise ImportError("sentence-transformers and torch are required. Install with: pip install sentence-transformers torch")

from ..models.vector_models import VectorStoreConfig, BioDataEmbedding
from ..utils.text_processor import BioDataTextProcessor
from ...data_processing.models.bio_models import BioData


class EmbeddingManager:
    """Manage embedding generation and caching for bio data."""
    
    def __init__(self, config: VectorStoreConfig):
        """
        Initialize embedding manager.
        
        Args:
            config: Vector store configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.text_processor = BioDataTextProcessor(config)
        
        # Initialize embedding model
        self.model = self._load_embedding_model()
        self.model_dimension = self._get_model_dimension()
        
        # Embedding cache
        self._embedding_cache = {} if config.enable_embedding_cache else None
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _load_embedding_model(self) -> SentenceTransformer:
        """Load sentence transformer model."""
        try:
            self.logger.info(f"Loading embedding model: {self.config.model_name}")
            
            # Configure device
            device = self.config.device
            if device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
            
            model = SentenceTransformer(self.config.model_name, device=device)
            
            # Set max sequence length
            model.max_seq_length = self.config.max_sequence_length
            
            self.logger.info(f"Model loaded successfully on {device}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _get_model_dimension(self) -> int:
        """Get the dimension of the embedding model."""
        try:
            # Test with dummy text to get dimension
            test_embedding = self.model.encode(["test"], show_progress_bar=False)
            dimension = test_embedding.shape[1]
            
            if dimension != self.config.dimension:
                self.logger.warning(f"Model dimension ({dimension}) differs from config ({self.config.dimension})")
            
            return dimension
            
        except Exception as e:
            self.logger.error(f"Failed to determine model dimension: {e}")
            return self.config.dimension
    
    def generate_bio_data_embedding(
        self, 
        bio_data: BioData, 
        embedding_fields: Optional[Dict[str, List[str]]] = None
    ) -> BioDataEmbedding:
        """
        Generate embedding for bio data.
        
        Args:
            bio_data: Bio data to embed
            embedding_fields: Fields to include in embedding
            
        Returns:
            Bio data embedding object
        """
        start_time = time.time()
        
        # Process bio data to text
        embedding_text = self.text_processor.process_bio_data(bio_data, embedding_fields)
        
        if not embedding_text or len(embedding_text.strip()) < self.config.min_text_length:
            raise ValueError(f"Insufficient text for embedding: {len(embedding_text)} characters")
        
        # Generate embedding
        embedding_vector = self._generate_embedding(embedding_text)
        
        # Create bio data ID
        bio_data_id = self._create_bio_data_id(bio_data)
        
        # Extract section metadata
        section_metadata = self._extract_section_metadata(bio_data)
        
        # Create embedding object
        bio_embedding = BioDataEmbedding(
            bio_data_id=bio_data_id,
            source_file=bio_data.metadata.source_file,
            bio_data_type=bio_data.metadata.bio_data_type,
            embedding_vector=embedding_vector.tolist(),
            embedding_text=embedding_text,
            model_name=self.config.model_name,
            vector_dimension=len(embedding_vector),
            **section_metadata
        )
        
        processing_time = time.time() - start_time
        self.logger.info(f"Generated embedding for {bio_data_id} in {processing_time:.3f}s")
        
        return bio_embedding
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Check cache first
        if self._embedding_cache is not None:
            cache_key = hash(text)
            if cache_key in self._embedding_cache:
                self._cache_hits += 1
                return self._embedding_cache[cache_key]
            self._cache_misses += 1
        
        try:
            # Generate embedding
            embedding = self.model.encode(
                [text], 
                batch_size=1,
                show_progress_bar=False,
                convert_to_tensor=False,
                normalize_embeddings=self.config.normalize_vectors
            )[0]
            
            # Ensure numpy array
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Cache the embedding
            if self._embedding_cache is not None:
                # Simple cache size management
                if len(self._embedding_cache) > 1000:  # Simple LRU-like behavior
                    # Remove oldest entries (simplified)
                    oldest_keys = list(self._embedding_cache.keys())[:100]
                    for key in oldest_keys:
                        del self._embedding_cache[key]
                
                self._embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def _create_bio_data_id(self, bio_data: BioData) -> str:
        """
        Create unique ID for bio data.
        
        Args:
            bio_data: Bio data object
            
        Returns:
            Unique identifier
        """
        # Use combination of name, source file, and timestamp
        name = bio_data.personal_info.name or "unknown"
        source_file = Path(bio_data.metadata.source_file).stem
        timestamp = bio_data.metadata.extraction_timestamp.strftime("%Y%m%d_%H%M%S")
        bio_type = bio_data.metadata.bio_data_type
        
        bio_id = f"{bio_type}_{name.replace(' ', '_')}_{source_file}_{timestamp}"
        
        # Clean ID to remove special characters
        bio_id = "".join(c for c in bio_id if c.isalnum() or c in ['_', '-'])
        
        return bio_id
    
    def _extract_section_metadata(self, bio_data: BioData) -> Dict[str, Dict[str, Any]]:
        """
        Extract metadata from each section for storage.
        
        Args:
            bio_data: Bio data object
            
        Returns:
            Section metadata dictionary
        """
        metadata = {}
        
        # Personal info (always present)
        metadata['personal_info'] = {
            'name': bio_data.personal_info.name,
            'age': bio_data.personal_info.age,
            'gender': bio_data.personal_info.gender,
            'location': bio_data.personal_info.location,
            'nationality': bio_data.personal_info.nationality
        }
        
        # Education
        if bio_data.education:
            metadata['education'] = {
                'degree': bio_data.education.degree,
                'institution': bio_data.education.institution,
                'major': bio_data.education.major,
                'graduation_year': bio_data.education.graduation_year,
                'gpa': bio_data.education.gpa,
                'certifications': bio_data.education.certifications
            }
        
        # Professional
        if bio_data.professional:
            metadata['professional'] = {
                'current_job': bio_data.professional.current_job,
                'company': bio_data.professional.company,
                'industry': bio_data.professional.industry,
                'experience_years': bio_data.professional.experience_years,
                'skills': bio_data.professional.skills,
                'salary_range': bio_data.professional.salary_range
            }
        
        # Interests
        if bio_data.interests:
            metadata['interests'] = {
                'hobbies': bio_data.interests.hobbies,
                'sports': bio_data.interests.sports,
                'music': bio_data.interests.music,
                'travel': bio_data.interests.travel,
                'books': bio_data.interests.books,
                'movies': bio_data.interests.movies
            }
        
        # Lifestyle
        if bio_data.lifestyle:
            metadata['lifestyle'] = {
                'diet_preferences': [str(d) for d in bio_data.lifestyle.diet_preferences] if bio_data.lifestyle.diet_preferences else None,
                'exercise_habits': bio_data.lifestyle.exercise_habits,
                'smoking': bio_data.lifestyle.smoking,
                'drinking': bio_data.lifestyle.drinking,
                'pets': bio_data.lifestyle.pets
            }
        
        # Relationship
        if bio_data.relationship:
            metadata['relationship'] = {
                'relationship_status': bio_data.relationship.relationship_status,
                'looking_for': bio_data.relationship.looking_for,
                'preferences': bio_data.relationship.preferences,
                'deal_breakers': bio_data.relationship.deal_breakers
            }
        
        return metadata
    
    def batch_generate_embeddings(
        self, 
        bio_data_list: List[BioData],
        embedding_fields: Optional[Dict[str, List[str]]] = None
    ) -> List[BioDataEmbedding]:
        """
        Generate embeddings for multiple bio data objects.
        
        Args:
            bio_data_list: List of bio data objects
            embedding_fields: Fields to include in embeddings
            
        Returns:
            List of bio data embeddings
        """
        self.logger.info(f"Generating embeddings for {len(bio_data_list)} bio data objects")
        
        embeddings = []
        start_time = time.time()
        
        for i, bio_data in enumerate(bio_data_list):
            try:
                embedding = self.generate_bio_data_embedding(bio_data, embedding_fields)
                embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(bio_data_list)} embeddings")
                    
            except Exception as e:
                self.logger.error(f"Failed to generate embedding for bio data {i}: {e}")
                continue
        
        total_time = time.time() - start_time
        self.logger.info(f"Generated {len(embeddings)} embeddings in {total_time:.3f}s")
        
        return embeddings
    
    def update_embedding(
        self, 
        bio_data: BioData, 
        existing_embedding: BioDataEmbedding,
        embedding_fields: Optional[Dict[str, List[str]]] = None
    ) -> BioDataEmbedding:
        """
        Update existing embedding with new bio data.
        
        Args:
            bio_data: Updated bio data
            existing_embedding: Existing embedding to update
            embedding_fields: Fields to include in embedding
            
        Returns:
            Updated bio data embedding
        """
        self.logger.info(f"Updating embedding for {existing_embedding.bio_data_id}")
        
        # Generate new embedding
        new_embedding = self.generate_bio_data_embedding(bio_data, embedding_fields)
        
        # Preserve the original ID and creation time if desired
        new_embedding.bio_data_id = existing_embedding.bio_data_id
        
        return new_embedding
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        if self._embedding_cache is None:
            return {"cache_enabled": False}
        
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_enabled": True,
            "cache_size": len(self._embedding_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate
        }
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        if self._embedding_cache is not None:
            self._embedding_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            self.logger.info("Embedding cache cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "model_name": self.config.model_name,
            "dimension": self.model_dimension,
            "max_sequence_length": self.config.max_sequence_length,
            "device": str(self.model.device),
            "normalize_vectors": self.config.normalize_vectors
        }
