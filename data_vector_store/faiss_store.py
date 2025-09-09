"""
FAISS-based vector store for bio data similarity matching.
"""
import json
import time
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu or faiss-gpu is required. Install with: pip install faiss-cpu")

from ..models.vector_models import (
    VectorStoreConfig, BioDataEmbedding, SimilarityResult, SearchQuery,
    SearchResults, VectorStoreStats, IndexMetadata, IndexType
)
from ..utils.config_manager import VectorStoreConfigManager
from ..utils.embedding_manager import EmbeddingManager
from ...data_processing.models.bio_models import BioData


class FAISSVectorStore:
    """FAISS-based vector store for bio data matching."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize FAISS vector store.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_manager = VectorStoreConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.logger = self._setup_logging()
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(self.config)
        
        # FAISS indices for different bio data types
        self.indices: Dict[str, faiss.Index] = {}
        self.metadata: Dict[str, IndexMetadata] = {}
        
        # Performance tracking
        self.search_times = []
        self.total_searches = 0
        
        # Initialize or load existing indices
        self._initialize_indices()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_indices(self) -> None:
        """Initialize FAISS indices for bio data types."""
        bio_types = ["my_biodata", "ppl_biodata"]
        
        for bio_type in bio_types:
            index_path = self.config_manager.get_index_path(bio_type)
            metadata_path = self.config_manager.get_metadata_path(bio_type)
            
            if index_path.exists() and metadata_path.exists():
                # Load existing index
                self._load_index(bio_type)
            else:
                # Create new index
                self._create_index(bio_type)
    
    def _create_index(self, bio_type: str) -> None:
        """
        Create new FAISS index.
        
        Args:
            bio_type: Type of bio data (my_biodata or ppl_biodata)
        """
        self.logger.info(f"Creating new FAISS index for {bio_type}")
        
        # Create FAISS index based on configuration
        dimension = self.embedding_manager.model_dimension
        
        if self.config.index_type == IndexType.FLAT_IP:
            index = faiss.IndexFlatIP(dimension)
        elif self.config.index_type == IndexType.FLAT_L2:
            index = faiss.IndexFlatL2(dimension)
        elif self.config.index_type == IndexType.IVF_FLAT:
            # For larger datasets, use inverted file index
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 centroids
        else:
            # Default to flat IP
            index = faiss.IndexFlatIP(dimension)
        
        self.indices[bio_type] = index
        
        # Create metadata
        self.metadata[bio_type] = IndexMetadata(
            index_type=self.config.index_type,
            dimension=dimension,
            total_vectors=0,
            config_hash=self.config_manager.get_config_hash()
        )
        
        # Ensure directory exists
        index_path = self.config_manager.get_index_path(bio_type)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created {self.config.index_type} index for {bio_type} with dimension {dimension}")
    
    def _load_index(self, bio_type: str) -> None:
        """
        Load existing FAISS index and metadata.
        
        Args:
            bio_type: Type of bio data
        """
        try:
            index_path = self.config_manager.get_index_path(bio_type)
            metadata_path = self.config_manager.get_metadata_path(bio_type)
            
            # Load FAISS index
            index = faiss.read_index(str(index_path))
            self.indices[bio_type] = index
            
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            self.metadata[bio_type] = IndexMetadata(**metadata_dict)
            
            self.logger.info(f"Loaded {bio_type} index with {self.metadata[bio_type].total_vectors} vectors")
            
        except Exception as e:
            self.logger.error(f"Failed to load {bio_type} index: {e}")
            # Create new index if loading fails
            self._create_index(bio_type)
    
    def _save_index(self, bio_type: str) -> None:
        """
        Save FAISS index and metadata to disk.
        
        Args:
            bio_type: Type of bio data
        """
        try:
            index_path = self.config_manager.get_index_path(bio_type)
            metadata_path = self.config_manager.get_metadata_path(bio_type)
            
            # Ensure directory exists
            index_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.indices[bio_type], str(index_path))
            
            # Save metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata[bio_type].dict(), f, indent=2, default=str)
            
            self.logger.debug(f"Saved {bio_type} index and metadata")
            
        except Exception as e:
            self.logger.error(f"Failed to save {bio_type} index: {e}")
            raise
    
    def add_bio_data(self, bio_data: BioData) -> str:
        """
        Add bio data to the vector store.
        
        Args:
            bio_data: Bio data to add
            
        Returns:
            Bio data ID
        """
        start_time = time.time()
        
        # Generate embedding
        bio_embedding = self.embedding_manager.generate_bio_data_embedding(bio_data)
        bio_type = bio_embedding.bio_data_type
        
        # Check for duplicates if enabled
        if self.config.enable_duplicate_detection:
            duplicate_id = self._check_for_duplicates(bio_embedding)
            if duplicate_id:
                self.logger.warning(f"Duplicate bio data detected: {duplicate_id}")
                return duplicate_id
        
        # Add to FAISS index
        embedding_vector = np.array([bio_embedding.embedding_vector], dtype=np.float32)
        
        # Normalize vectors if configured
        if self.config.normalize_vectors:
            faiss.normalize_L2(embedding_vector)
        
        self.indices[bio_type].add(embedding_vector)
        
        # Update metadata
        self.metadata[bio_type].add_bio_data(bio_embedding)
        
        # Save to disk
        self._save_index(bio_type)
        
        processing_time = time.time() - start_time
        self.logger.info(f"Added bio data {bio_embedding.bio_data_id} to {bio_type} in {processing_time:.3f}s")
        
        return bio_embedding.bio_data_id
    
    def _check_for_duplicates(self, bio_embedding: BioDataEmbedding) -> Optional[str]:
        """
        Check for duplicate bio data based on similarity threshold.
        
        Args:
            bio_embedding: Bio embedding to check
            
        Returns:
            ID of duplicate if found, None otherwise
        """
        bio_type = bio_embedding.bio_data_type
        
        if bio_type not in self.indices or self.indices[bio_type].ntotal == 0:
            return None
        
        # Search for similar embeddings
        embedding_vector = np.array([bio_embedding.embedding_vector], dtype=np.float32)
        
        if self.config.normalize_vectors:
            faiss.normalize_L2(embedding_vector)
        
        # Search for top 5 most similar
        scores, indices = self.indices[bio_type].search(embedding_vector, min(5, self.indices[bio_type].ntotal))
        
        # Check if any score exceeds duplicate threshold
        for score, idx in zip(scores[0], indices[0]):
            if score >= self.config.duplicate_threshold:
                # Find bio data ID for this index
                for bio_id, stored_embedding in self.metadata[bio_type].id_to_metadata.items():
                    # Simple approach - would need better index mapping in production
                    if np.allclose(stored_embedding.embedding_vector, bio_embedding.embedding_vector, rtol=1e-5):
                        return bio_id
        
        return None
    
    def remove_bio_data(self, bio_data_id: str, bio_type: str) -> bool:
        """
        Remove bio data from the vector store.
        
        Args:
            bio_data_id: ID of bio data to remove
            bio_type: Type of bio data
            
        Returns:
            True if removed successfully
        """
        if bio_type not in self.metadata:
            return False
        
        if bio_data_id not in self.metadata[bio_type].id_to_metadata:
            self.logger.warning(f"Bio data {bio_data_id} not found in {bio_type}")
            return False
        
        # For simplicity, we'll rebuild the index without this item
        # In production, you might want to use a more efficient approach
        remaining_embeddings = []
        for bid, embedding in self.metadata[bio_type].id_to_metadata.items():
            if bid != bio_data_id:
                remaining_embeddings.append(embedding)
        
        # Recreate index
        self._create_index(bio_type)
        
        # Re-add remaining embeddings
        if remaining_embeddings:
            vectors = np.array([emb.embedding_vector for emb in remaining_embeddings], dtype=np.float32)
            if self.config.normalize_vectors:
                faiss.normalize_L2(vectors)
            
            self.indices[bio_type].add(vectors)
            
            # Update metadata
            for embedding in remaining_embeddings:
                self.metadata[bio_type].add_bio_data(embedding)
        
        # Save updated index
        self._save_index(bio_type)
        
        self.logger.info(f"Removed bio data {bio_data_id} from {bio_type}")
        return True
    
    def search_similar(self, query: SearchQuery) -> SearchResults:
        """
        Search for similar bio data.
        
        Args:
            query: Search query configuration
            
        Returns:
            Search results
        """
        start_time = time.time()
        
        # Get query embedding
        if query.query_bio_data_id:
            query_embedding = self._get_bio_data_embedding(query.query_bio_data_id)
        elif query.query_embedding:
            query_embedding = np.array([query.query_embedding], dtype=np.float32)
        elif query.query_text:
            # Generate embedding from text
            embedding_vector = self.embedding_manager._generate_embedding(query.query_text)
            query_embedding = np.array([embedding_vector], dtype=np.float32)
        else:
            raise ValueError("No query provided")
        
        # Normalize if configured
        if self.config.normalize_vectors:
            faiss.normalize_L2(query_embedding)
        
        # Search in target bio type
        target_type = query.target_bio_type
        if target_type not in self.indices:
            raise ValueError(f"Unknown target bio type: {target_type}")
        
        index = self.indices[target_type]
        if index.ntotal == 0:
            return SearchResults(
                query=query,
                total_results=0,
                search_time_ms=0,
                results=[]
            )
        
        # Perform search
        k = min(query.k, index.ntotal)
        if query.enable_reranking:
            # Retrieve more results for reranking
            search_k = min(query.rerank_top_k if hasattr(query, 'rerank_top_k') else self.config.rerank_top_k, index.ntotal)
        else:
            search_k = k
        
        scores, indices = index.search(query_embedding, search_k)
        
        # Convert results to SimilarityResult objects
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
            
            # Find bio data for this index
            bio_data_embedding = self._get_bio_data_by_index(target_type, idx)
            if bio_data_embedding and score >= query.similarity_threshold:
                similarity_result = SimilarityResult(
                    bio_data_id=bio_data_embedding.bio_data_id,
                    similarity_score=float(score),
                    distance=float(1.0 - score) if self.config.index_type == IndexType.FLAT_IP else float(score),
                    personal_info=bio_data_embedding.personal_info,
                    education=bio_data_embedding.education,
                    professional=bio_data_embedding.professional,
                    interests=bio_data_embedding.interests,
                    lifestyle=bio_data_embedding.lifestyle,
                    relationship=bio_data_embedding.relationship,
                    source_file=bio_data_embedding.source_file,
                    bio_data_type=bio_data_embedding.bio_data_type,
                    created_at=bio_data_embedding.created_at
                )
                results.append(similarity_result)
        
        # Apply reranking if enabled
        if query.enable_reranking and len(results) > k:
            results = self._rerank_results(results, query)[:k]
        else:
            results = results[:k]
        
        search_time = time.time() - start_time
        search_time_ms = search_time * 1000
        
        # Update performance tracking
        self.search_times.append(search_time_ms)
        self.total_searches += 1
        
        # Keep only last 100 search times for average calculation
        if len(self.search_times) > 100:
            self.search_times = self.search_times[-100:]
        
        self.logger.info(f"Found {len(results)} similar bio data in {search_time_ms:.2f}ms")
        
        return SearchResults(
            query=query,
            total_results=len(results),
            search_time_ms=search_time_ms,
            results=results,
            index_info={
                "total_vectors": index.ntotal,
                "dimension": index.d if hasattr(index, 'd') else self.embedding_manager.model_dimension
            },
            performance_stats={
                "average_search_time_ms": np.mean(self.search_times) if self.search_times else 0
            }
        )
    
    def _get_bio_data_embedding(self, bio_data_id: str) -> np.ndarray:
        """Get embedding vector for bio data ID."""
        for bio_type in self.metadata:
            if bio_data_id in self.metadata[bio_type].id_to_metadata:
                embedding = self.metadata[bio_type].id_to_metadata[bio_data_id]
                return np.array([embedding.embedding_vector], dtype=np.float32)
        
        raise ValueError(f"Bio data {bio_data_id} not found")
    
    def _get_bio_data_by_index(self, bio_type: str, index: int) -> Optional[BioDataEmbedding]:
        """Get bio data embedding by FAISS index."""
        # Simple approach - in production you'd want better index mapping
        metadata_items = list(self.metadata[bio_type].id_to_metadata.values())
        if 0 <= index < len(metadata_items):
            return metadata_items[index]
        return None
    
    def _rerank_results(self, results: List[SimilarityResult], query: SearchQuery) -> List[SimilarityResult]:
        """
        Rerank results based on section weights and additional criteria.
        
        Args:
            results: Initial search results
            query: Search query with reranking configuration
            
        Returns:
            Reranked results
        """
        # Get section weights
        section_weights = query.section_weights or self.config_manager.get_section_weights()
        
        for result in results:
            # Calculate weighted score based on different sections
            weighted_score = 0.0
            total_weight = 0.0
            
            sections = {
                'personal_info': result.personal_info,
                'education': result.education,
                'professional': result.professional,
                'interests': result.interests,
                'lifestyle': result.lifestyle,
                'relationship': result.relationship
            }
            
            for section_name, section_data in sections.items():
                weight = section_weights.get(section_name, 0.0)
                if weight > 0 and section_data:
                    # Simple scoring - in production you'd want more sophisticated scoring
                    section_score = result.similarity_score
                    weighted_score += section_score * weight
                    total_weight += weight
            
            # Update similarity score with weighted score
            if total_weight > 0:
                result.similarity_score = weighted_score / total_weight
        
        # Sort by updated similarity score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results
    
    def get_stats(self) -> VectorStoreStats:
        """Get vector store statistics."""
        total_vectors = sum(idx.ntotal for idx in self.indices.values())
        my_biodata_count = self.indices.get("my_biodata", faiss.IndexFlatIP(1)).ntotal
        ppl_biodata_count = self.indices.get("ppl_biodata", faiss.IndexFlatIP(1)).ntotal
        
        # Calculate storage sizes (approximate)
        index_size_mb = 0.0
        metadata_size_mb = 0.0
        
        for bio_type in ["my_biodata", "ppl_biodata"]:
            index_path = self.config_manager.get_index_path(bio_type)
            metadata_path = self.config_manager.get_metadata_path(bio_type)
            
            if index_path.exists():
                index_size_mb += index_path.stat().st_size / (1024 * 1024)
            if metadata_path.exists():
                metadata_size_mb += metadata_path.stat().st_size / (1024 * 1024)
        
        return VectorStoreStats(
            total_vectors=total_vectors,
            my_biodata_count=my_biodata_count,
            ppl_biodata_count=ppl_biodata_count,
            index_size_mb=index_size_mb,
            metadata_size_mb=metadata_size_mb,
            total_size_mb=index_size_mb + metadata_size_mb,
            vector_dimension=self.embedding_manager.model_dimension,
            index_type=self.config.index_type,
            model_name=self.config.model_name,
            last_search_time_ms=self.search_times[-1] if self.search_times else None,
            average_search_time_ms=np.mean(self.search_times) if self.search_times else None,
            total_searches=self.total_searches,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
    
    def backup_indices(self, backup_path: Optional[str] = None) -> str:
        """
        Create backup of all indices and metadata.
        
        Args:
            backup_path: Custom backup path
            
        Returns:
            Path to backup directory
        """
        if backup_path is None:
            backup_path = self.config_manager.get_backup_path("all")
        
        backup_path = Path(backup_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = backup_path / f"backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup each bio type
        for bio_type in ["my_biodata", "ppl_biodata"]:
            bio_backup_dir = backup_dir / bio_type
            bio_backup_dir.mkdir(exist_ok=True)
            
            # Copy index and metadata files
            index_path = self.config_manager.get_index_path(bio_type)
            metadata_path = self.config_manager.get_metadata_path(bio_type)
            
            if index_path.exists():
                shutil.copy2(index_path, bio_backup_dir / index_path.name)
            if metadata_path.exists():
                shutil.copy2(metadata_path, bio_backup_dir / metadata_path.name)
        
        # Save configuration
        config_backup_path = backup_dir / "vector_store_config.yaml"
        self.config_manager.save_config(config_backup_path)
        
        self.logger.info(f"Created backup at: {backup_dir}")
        return str(backup_dir)
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """
        Restore indices from backup.
        
        Args:
            backup_path: Path to backup directory
            
        Returns:
            True if restore successful
        """
        try:
            backup_path = Path(backup_path)
            
            for bio_type in ["my_biodata", "ppl_biodata"]:
                bio_backup_dir = backup_path / bio_type
                
                if bio_backup_dir.exists():
                    # Restore index and metadata
                    for file_path in bio_backup_dir.iterdir():
                        if file_path.suffix == '.faiss':
                            target_path = self.config_manager.get_index_path(bio_type)
                        elif file_path.suffix == '.json':
                            target_path = self.config_manager.get_metadata_path(bio_type)
                        else:
                            continue
                        
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file_path, target_path)
            
            # Reload indices
            self._initialize_indices()
            
            self.logger.info(f"Restored from backup: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore from backup: {e}")
            return False
    
    def optimize_indices(self) -> None:
        """Optimize FAISS indices for better performance."""
        for bio_type, index in self.indices.items():
            if index.ntotal > 1000:  # Only optimize if we have enough vectors
                self.logger.info(f"Optimizing {bio_type} index with {index.ntotal} vectors")
                
                # For IVF indices, train if not already trained
                if hasattr(index, 'is_trained') and not index.is_trained:
                    # Get all vectors for training
                    all_vectors = []
                    for embedding in self.metadata[bio_type].id_to_metadata.values():
                        all_vectors.append(embedding.embedding_vector)
                    
                    if all_vectors:
                        training_vectors = np.array(all_vectors, dtype=np.float32)
                        if self.config.normalize_vectors:
                            faiss.normalize_L2(training_vectors)
                        
                        index.train(training_vectors)
                        self.logger.info(f"Trained {bio_type} index")
                
                # Save optimized index
                self._save_index(bio_type)
    
    def close(self) -> None:
        """Close vector store and save all indices."""
        for bio_type in self.indices:
            self._save_index(bio_type)
        
        self.logger.info("Vector store closed and saved")
