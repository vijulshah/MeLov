"""
Configuration manager for vector store operations.
"""
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib

from ..models.vector_models import VectorStoreConfig


class VectorStoreConfigManager:
    """Configuration manager for vector store operations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self.config = self._load_config()
        self.logger = self._setup_logging()
    
    def _get_default_config_path(self) -> Path:
        """Get default configuration file path."""
        return Path("data_vector_store/config/vector_store_config.yaml")
    
    def _load_config(self) -> VectorStoreConfig:
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Extract vector store specific configuration
            vector_config = config_data.get('vector_store', {})
            embedding_config = config_data.get('vector_store', {}).get('embeddings', {})
            search_config = config_data.get('search', {})
            performance_config = config_data.get('performance', {})
            validation_config = config_data.get('validation', {})
            
            # Merge configurations
            merged_config = {
                **vector_config.get('faiss', {}),
                **vector_config.get('storage', {}),
                **embedding_config,
                **vector_config.get('text_processing', {}),
                **search_config,
                **performance_config,
                **validation_config
            }
            
            return VectorStoreConfig(**merged_config)
            
        except Exception as e:
            # Return default configuration if loading fails
            logging.warning(f"Failed to load configuration: {e}. Using defaults.")
            return VectorStoreConfig()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            log_config = config_data.get('logging', {})
        except:
            log_config = {'level': 'INFO'}
        
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))
        
        # Create formatter
        formatter = logging.Formatter(
            log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Console handler
        if not logger.handlers:
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
    
    def get_config(self) -> VectorStoreConfig:
        """Get current configuration."""
        return self.config
    
    def get_index_path(self, bio_type: str) -> Path:
        """Get path for FAISS index file."""
        base_path = Path(self.config.base_path)
        
        if bio_type == "my_biodata":
            return base_path / self.config.my_biodata_index
        elif bio_type == "ppl_biodata":
            return base_path / self.config.ppl_biodata_index
        else:
            raise ValueError(f"Unknown bio type: {bio_type}")
    
    def get_metadata_path(self, bio_type: str) -> Path:
        """Get path for metadata file."""
        index_path = self.get_index_path(bio_type)
        return index_path.with_suffix(self.config.metadata_extension)
    
    def get_backup_path(self, bio_type: str) -> Path:
        """Get backup directory path."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            backup_path = config_data.get('maintenance', {}).get('backup_path', 'data_vector_store/backups')
        except:
            backup_path = 'data_vector_store/backups'
        
        return Path(backup_path) / bio_type
    
    def get_embedding_fields(self) -> Dict[str, list]:
        """Get fields to include in embeddings."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            return config_data.get('vector_store', {}).get('text_processing', {}).get('embedding_fields', {})
        except:
            return {}
    
    def get_section_weights(self) -> Dict[str, float]:
        """Get section weights for similarity calculation."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            return config_data.get('search', {}).get('section_weights', {})
        except:
            return {
                'personal_info': 0.2,
                'education': 0.15,
                'professional': 0.25,
                'interests': 0.25,
                'lifestyle': 0.1,
                'relationship': 0.05
            }
    
    def get_maintenance_config(self) -> Dict[str, Any]:
        """Get maintenance configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            return config_data.get('maintenance', {})
        except:
            return {}
    
    def get_config_hash(self) -> str:
        """Get hash of current configuration for consistency checks."""
        config_str = yaml.dump(self.config.dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def validate_config(self) -> bool:
        """Validate current configuration."""
        try:
            # Check if all required paths exist or can be created
            base_path = Path(self.config.base_path)
            base_path.mkdir(parents=True, exist_ok=True)
            
            # Validate model name
            if self.config.model_name not in ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 
                                             'all-distilroberta-v1', 'paraphrase-MiniLM-L6-v2']:
                self.logger.warning(f"Unknown model name: {self.config.model_name}")
            
            # Validate dimension consistency
            model_dimensions = {
                'all-MiniLM-L6-v2': 384,
                'all-mpnet-base-v2': 768,
                'all-distilroberta-v1': 768,
                'paraphrase-MiniLM-L6-v2': 384
            }
            
            expected_dim = model_dimensions.get(self.config.model_name, self.config.dimension)
            if self.config.dimension != expected_dim:
                self.logger.warning(f"Dimension mismatch: configured {self.config.dimension}, "
                                  f"expected {expected_dim} for model {self.config.model_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()
        self.logger.info("Configuration reloaded")
    
    def update_config(self, **kwargs) -> None:
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated configuration: {key} = {value}")
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        output_path = Path(output_path) if output_path else self.config_path
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_dict = {
                'vector_store': {
                    'faiss': {
                        'index_type': self.config.index_type,
                        'dimension': self.config.dimension,
                        'normalize_vectors': self.config.normalize_vectors
                    },
                    'storage': {
                        'base_path': self.config.base_path,
                        'my_biodata_index': self.config.my_biodata_index,
                        'ppl_biodata_index': self.config.ppl_biodata_index,
                        'metadata_extension': self.config.metadata_extension
                    },
                    'embeddings': {
                        'model_name': self.config.model_name,
                        'max_sequence_length': self.config.max_sequence_length,
                        'batch_size': self.config.batch_size,
                        'device': self.config.device
                    }
                },
                'search': {
                    'default_k': self.config.default_k,
                    'similarity_threshold': self.config.similarity_threshold,
                    'enable_reranking': self.config.enable_reranking
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
