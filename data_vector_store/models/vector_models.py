"""
Pydantic models for vector store operations and bio data similarity matching.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Union, Tuple
from enum import Enum
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, validator


class IndexType(str, Enum):
    """FAISS index types."""
    FLAT_IP = "IndexFlatIP"  # Inner Product (cosine similarity)
    FLAT_L2 = "IndexFlatL2"  # L2 distance (Euclidean)
    IVF_FLAT = "IndexIVFFlat"  # Inverted file with flat quantizer
    IVF_PQ = "IndexIVFPQ"  # Inverted file with product quantizer


class EmbeddingModel(str, Enum):
    """Supported embedding models."""
    MINI_LM_L6_V2 = "all-MiniLM-L6-v2"
    MPNET_BASE_V2 = "all-mpnet-base-v2"
    DISTIL_ROBERTA_V1 = "all-distilroberta-v1"
    PARAPHRASE_MINI_LM = "paraphrase-MiniLM-L6-v2"


class VectorStoreConfig(BaseModel):
    """Configuration for vector store operations."""
    
    # FAISS configuration
    index_type: IndexType = IndexType.FLAT_IP
    dimension: int = Field(384, ge=128, le=1536)
    normalize_vectors: bool = True
    
    # Storage paths
    base_path: str = "data_vector_store/indices"
    my_biodata_index: str = "my_biodata.faiss"
    ppl_biodata_index: str = "ppl_biodata.faiss"
    metadata_extension: str = ".json"
    
    # Embedding settings
    model_name: EmbeddingModel = EmbeddingModel.MINI_LM_L6_V2
    max_sequence_length: int = Field(512, ge=128, le=1024)
    batch_size: int = Field(32, ge=1, le=128)
    device: str = "cpu"
    
    # Text processing
    chunk_size: int = Field(500, ge=100, le=2000)
    chunk_overlap: int = Field(50, ge=0, le=200)
    include_metadata_in_text: bool = True
    
    # Search configuration
    default_k: int = Field(10, ge=1, le=100)
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)
    enable_reranking: bool = True
    rerank_top_k: int = Field(50, ge=10, le=200)
    
    # Performance settings
    max_memory_usage_gb: float = Field(4.0, ge=1.0, le=32.0)
    enable_memory_mapping: bool = True
    n_jobs: int = -1
    enable_multiprocessing: bool = False
    
    # Validation settings
    min_text_length: int = Field(50, ge=10, le=500)
    max_text_length: int = Field(10000, ge=1000, le=50000)
    validate_embeddings: bool = True
    enable_duplicate_detection: bool = True
    duplicate_threshold: float = Field(0.95, ge=0.8, le=1.0)

    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        """Ensure chunk overlap is less than chunk size."""
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class BioDataEmbedding(BaseModel):
    """Embedding representation of bio data."""
    
    # Identifiers
    bio_data_id: str = Field(..., description="Unique identifier for bio data")
    source_file: str = Field(..., description="Original source file")
    bio_data_type: str = Field(..., description="Type: my_biodata or ppl_biodata")
    
    # Embedding data
    embedding_vector: List[float] = Field(..., description="Dense vector representation")
    embedding_text: str = Field(..., description="Text used to generate embedding")
    
    # Metadata for matching
    personal_info: Dict[str, Any] = Field(default_factory=dict)
    education: Optional[Dict[str, Any]] = None
    professional: Optional[Dict[str, Any]] = None
    interests: Optional[Dict[str, Any]] = None
    lifestyle: Optional[Dict[str, Any]] = None
    relationship: Optional[Dict[str, Any]] = None
    
    # Processing metadata
    created_at: datetime = Field(default_factory=datetime.now)
    model_name: str = Field(..., description="Embedding model used")
    vector_dimension: int = Field(..., description="Dimension of embedding vector")
    
    @validator('embedding_vector')
    def validate_embedding_vector(cls, v):
        """Validate embedding vector."""
        if not v:
            raise ValueError("Embedding vector cannot be empty")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Embedding vector must contain only numbers")
        return v

    @validator('vector_dimension')
    def validate_dimension_consistency(cls, v, values):
        """Ensure vector dimension matches actual vector length."""
        if 'embedding_vector' in values and len(values['embedding_vector']) != v:
            raise ValueError("Vector dimension must match embedding vector length")
        return v

    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class SimilarityResult(BaseModel):
    """Result of similarity search."""
    
    # Match information
    bio_data_id: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    distance: float = Field(..., description="Distance metric from FAISS")
    
    # Bio data information
    personal_info: Dict[str, Any]
    education: Optional[Dict[str, Any]] = None
    professional: Optional[Dict[str, Any]] = None
    interests: Optional[Dict[str, Any]] = None
    lifestyle: Optional[Dict[str, Any]] = None
    relationship: Optional[Dict[str, Any]] = None
    
    # Metadata
    source_file: str
    bio_data_type: str
    created_at: datetime
    
    # Matching details
    matching_reasons: List[str] = Field(default_factory=list)
    section_scores: Dict[str, float] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class SearchQuery(BaseModel):
    """Search query configuration."""
    
    # Query parameters
    query_bio_data_id: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    query_text: Optional[str] = None
    
    # Search configuration
    k: int = Field(10, ge=1, le=100, description="Number of results to return")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)
    target_bio_type: str = Field("ppl_biodata", description="Type to search in")
    
    # Filtering options
    filter_criteria: Optional[Dict[str, Any]] = None
    exclude_ids: Optional[List[str]] = None
    
    # Advanced options
    enable_reranking: bool = True
    section_weights: Optional[Dict[str, float]] = None

    @validator('query_embedding', 'query_text', 'query_bio_data_id')
    def validate_query_inputs(cls, v, values, field):
        """Ensure at least one query input is provided."""
        query_fields = ['query_bio_data_id', 'query_embedding', 'query_text']
        provided_fields = [f for f in query_fields if values.get(f) is not None or (field.name == f and v is not None)]
        
        if not provided_fields:
            raise ValueError("At least one of query_bio_data_id, query_embedding, or query_text must be provided")
        
        return v


class SearchResults(BaseModel):
    """Complete search results."""
    
    # Query information
    query: SearchQuery
    total_results: int
    search_time_ms: float
    
    # Results
    results: List[SimilarityResult]
    
    # Metadata
    index_info: Dict[str, Any] = Field(default_factory=dict)
    performance_stats: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class VectorStoreStats(BaseModel):
    """Statistics about vector store."""
    
    # Index statistics
    total_vectors: int = 0
    my_biodata_count: int = 0
    ppl_biodata_count: int = 0
    
    # Storage information
    index_size_mb: float = 0.0
    metadata_size_mb: float = 0.0
    total_size_mb: float = 0.0
    
    # Configuration
    vector_dimension: int
    index_type: str
    model_name: str
    
    # Performance metrics
    last_search_time_ms: Optional[float] = None
    average_search_time_ms: Optional[float] = None
    total_searches: int = 0
    
    # Timestamps
    created_at: datetime
    last_updated: datetime = Field(default_factory=datetime.now)
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class IndexMetadata(BaseModel):
    """Metadata for FAISS index."""
    
    # Index configuration
    index_type: IndexType
    dimension: int
    total_vectors: int
    
    # Bio data mapping
    id_to_metadata: Dict[str, BioDataEmbedding] = Field(default_factory=dict)
    
    # Statistics
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    
    # Configuration hash for consistency checks
    config_hash: Optional[str] = None
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        
    def add_bio_data(self, bio_data_embedding: BioDataEmbedding) -> None:
        """Add bio data embedding to metadata."""
        self.id_to_metadata[bio_data_embedding.bio_data_id] = bio_data_embedding
        self.total_vectors = len(self.id_to_metadata)
        self.last_updated = datetime.now()
    
    def remove_bio_data(self, bio_data_id: str) -> bool:
        """Remove bio data from metadata."""
        if bio_data_id in self.id_to_metadata:
            del self.id_to_metadata[bio_data_id]
            self.total_vectors = len(self.id_to_metadata)
            self.last_updated = datetime.now()
            return True
        return False
    
    def get_bio_data_ids(self) -> List[str]:
        """Get all bio data IDs."""
        return list(self.id_to_metadata.keys())
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about stored bio data."""
        my_count = sum(1 for bd in self.id_to_metadata.values() 
                      if bd.bio_data_type == "my_biodata")
        ppl_count = sum(1 for bd in self.id_to_metadata.values() 
                       if bd.bio_data_type == "ppl_biodata")
        
        return {
            "total": self.total_vectors,
            "my_biodata": my_count,
            "ppl_biodata": ppl_count
        }
