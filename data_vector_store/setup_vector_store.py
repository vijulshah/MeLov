"""
Main setup and management for bio data vector store using FAISS.
This module provides high-level functions to setup, manage, and use the vector store.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import error handling for optional dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

# Local imports
from data_processing.models.bio_models import BioData, ExtractionResult
from data_processing.extraction.bio_extractor import BioDataExtractor
from data_processing.utils.config_manager import ConfigManager


class VectorStoreSetup:
    """Setup and management class for bio data vector store."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize vector store setup.
        
        Args:
            config_path: Path to vector store configuration file
        """
        self.config_path = config_path or "data_vector_store/config/vector_store_config.yaml"
        self.logger = self._setup_logging()
        
        # Check dependencies
        self._check_dependencies()
        
        # Initialize components only if dependencies are available
        if FAISS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
            from data_vector_store.faiss_store import FAISSVectorStore
            self.vector_store = FAISSVectorStore(config_path)
        else:
            self.vector_store = None
            self.logger.warning("Vector store not available due to missing dependencies")
        
        # Initialize bio data extractor for processing
        self.bio_extractor = BioDataExtractor(ConfigManager())
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        missing_deps = []
        
        if not FAISS_AVAILABLE:
            missing_deps.append("faiss-cpu")
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            missing_deps.append("sentence-transformers")
        
        if missing_deps:
            self.logger.error(f"Missing required dependencies: {', '.join(missing_deps)}")
            self.logger.info("Install with: pip install " + " ".join(missing_deps))
    
    def setup_vector_store(self) -> bool:
        """
        Setup the vector store with initial configuration.
        
        Returns:
            True if setup successful
        """
        if not self.vector_store:
            self.logger.error("Vector store not available - missing dependencies")
            return False
        
        try:
            # Validate configuration
            if not self.vector_store.config_manager.validate_config():
                self.logger.error("Invalid configuration")
                return False
            
            # Create necessary directories
            base_path = Path(self.vector_store.config.base_path)
            base_path.mkdir(parents=True, exist_ok=True)
            
            # Create backup directory
            backup_path = self.vector_store.config_manager.get_backup_path("all")
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Create logs directory
            logs_path = Path("logs")
            logs_path.mkdir(exist_ok=True)
            
            self.logger.info("Vector store setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Vector store setup failed: {e}")
            return False
    
    def add_bio_data_from_json(self, json_path: str) -> Optional[str]:
        """
        Add bio data from JSON file to vector store.
        
        Args:
            json_path: Path to JSON file containing bio data
            
        Returns:
            Bio data ID if successful, None otherwise
        """
        if not self.vector_store:
            self.logger.error("Vector store not available")
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                bio_data_dict = json.load(f)
            
            # Convert dict to BioData object
            bio_data = BioData(**bio_data_dict)
            
            # Add to vector store
            bio_data_id = self.vector_store.add_bio_data(bio_data)
            
            self.logger.info(f"Added bio data from {json_path} with ID: {bio_data_id}")
            return bio_data_id
            
        except Exception as e:
            self.logger.error(f"Failed to add bio data from {json_path}: {e}")
            return None
    
    def process_and_add_pdf(self, pdf_path: str, bio_type: str = "ppl_biodata") -> Optional[str]:
        """
        Process PDF file and add extracted bio data to vector store.
        
        Args:
            pdf_path: Path to PDF file
            bio_type: Type of bio data (my_biodata or ppl_biodata)
            
        Returns:
            Bio data ID if successful, None otherwise
        """
        if not self.vector_store:
            self.logger.error("Vector store not available")
            return None
        
        try:
            # Extract bio data from PDF
            result = self.bio_extractor.extract_from_pdf(pdf_path, bio_type)
            
            if not result.success or not result.bio_data:
                self.logger.error(f"Failed to extract bio data from {pdf_path}: {result.errors}")
                return None
            
            # Add to vector store
            bio_data_id = self.vector_store.add_bio_data(result.bio_data)
            
            self.logger.info(f"Processed PDF {pdf_path} and added bio data with ID: {bio_data_id}")
            return bio_data_id
            
        except Exception as e:
            self.logger.error(f"Failed to process PDF {pdf_path}: {e}")
            return None
    
    def batch_process_directory(self, directory_path: str, bio_type: str = "ppl_biodata") -> List[str]:
        """
        Process all PDF files in a directory and add to vector store.
        
        Args:
            directory_path: Path to directory containing PDF files
            bio_type: Type of bio data
            
        Returns:
            List of bio data IDs that were successfully added
        """
        if not self.vector_store:
            self.logger.error("Vector store not available")
            return []
        
        directory = Path(directory_path)
        if not directory.exists():
            self.logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        pdf_files = list(directory.glob("*.pdf"))
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {directory_path}")
            return []
        
        self.logger.info(f"Processing {len(pdf_files)} PDF files from {directory_path}")
        
        successful_ids = []
        for pdf_file in pdf_files:
            bio_data_id = self.process_and_add_pdf(str(pdf_file), bio_type)
            if bio_data_id:
                successful_ids.append(bio_data_id)
        
        self.logger.info(f"Successfully processed {len(successful_ids)}/{len(pdf_files)} PDF files")
        return successful_ids
    
    def search_similar_bio_data(
        self, 
        query_bio_data_id: Optional[str] = None,
        query_text: Optional[str] = None,
        target_bio_type: str = "ppl_biodata",
        k: int = 10,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Search for similar bio data.
        
        Args:
            query_bio_data_id: ID of bio data to use as query
            query_text: Text to use as query
            target_bio_type: Type to search in (my_biodata or ppl_biodata)
            k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            Search results dictionary
        """
        if not self.vector_store:
            self.logger.error("Vector store not available")
            return {"error": "Vector store not available"}
        
        try:
            from data_vector_store.models.vector_models import SearchQuery
            
            # Create search query
            query = SearchQuery(
                query_bio_data_id=query_bio_data_id,
                query_text=query_text,
                target_bio_type=target_bio_type,
                k=k,
                similarity_threshold=similarity_threshold
            )
            
            # Perform search
            results = self.vector_store.search_similar(query)
            
            # Convert to dict for easier handling
            results_dict = {
                "total_results": results.total_results,
                "search_time_ms": results.search_time_ms,
                "results": []
            }
            
            for result in results.results:
                result_dict = {
                    "bio_data_id": result.bio_data_id,
                    "similarity_score": result.similarity_score,
                    "personal_info": result.personal_info,
                    "education": result.education,
                    "professional": result.professional,
                    "interests": result.interests,
                    "lifestyle": result.lifestyle,
                    "relationship": result.relationship,
                    "source_file": result.source_file
                }
                results_dict["results"].append(result_dict)
            
            return results_dict
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return {"error": str(e)}
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Statistics dictionary
        """
        if not self.vector_store:
            return {"error": "Vector store not available"}
        
        try:
            stats = self.vector_store.get_stats()
            return stats.dict()
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def backup_vector_store(self, backup_path: Optional[str] = None) -> str:
        """
        Create backup of vector store.
        
        Args:
            backup_path: Custom backup path
            
        Returns:
            Path to backup directory
        """
        if not self.vector_store:
            raise ValueError("Vector store not available")
        
        return self.vector_store.backup_indices(backup_path)
    
    def restore_vector_store(self, backup_path: str) -> bool:
        """
        Restore vector store from backup.
        
        Args:
            backup_path: Path to backup directory
            
        Returns:
            True if restore successful
        """
        if not self.vector_store:
            return False
        
        return self.vector_store.restore_from_backup(backup_path)
    
    def optimize_vector_store(self) -> None:
        """Optimize vector store indices for better performance."""
        if not self.vector_store:
            self.logger.error("Vector store not available")
            return
        
        self.vector_store.optimize_indices()
    
    def list_bio_data(self, bio_type: str = "ppl_biodata") -> List[Dict[str, Any]]:
        """
        List all bio data in the vector store.
        
        Args:
            bio_type: Type of bio data to list
            
        Returns:
            List of bio data information
        """
        if not self.vector_store:
            return []
        
        try:
            metadata = self.vector_store.metadata.get(bio_type, None)
            if not metadata:
                return []
            
            bio_data_list = []
            for bio_id, embedding in metadata.id_to_metadata.items():
                bio_info = {
                    "bio_data_id": bio_id,
                    "source_file": embedding.source_file,
                    "created_at": embedding.created_at.isoformat(),
                    "personal_info": embedding.personal_info
                }
                bio_data_list.append(bio_info)
            
            return bio_data_list
            
        except Exception as e:
            self.logger.error(f"Failed to list bio data: {e}")
            return []
    
    def remove_bio_data(self, bio_data_id: str, bio_type: str) -> bool:
        """
        Remove bio data from vector store.
        
        Args:
            bio_data_id: ID of bio data to remove
            bio_type: Type of bio data
            
        Returns:
            True if removed successfully
        """
        if not self.vector_store:
            return False
        
        return self.vector_store.remove_bio_data(bio_data_id, bio_type)
    
    def close(self) -> None:
        """Close vector store and save all data."""
        if self.vector_store:
            self.vector_store.close()


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bio Data Vector Store Setup")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--setup", action="store_true", help="Setup vector store")
    parser.add_argument("--add-pdf", help="Add PDF file to vector store")
    parser.add_argument("--add-json", help="Add JSON file to vector store")
    parser.add_argument("--bio-type", default="ppl_biodata", choices=["my_biodata", "ppl_biodata"])
    parser.add_argument("--search-text", help="Search for similar bio data using text")
    parser.add_argument("--search-id", help="Search for similar bio data using bio data ID")
    parser.add_argument("--stats", action="store_true", help="Show vector store statistics")
    parser.add_argument("--list", action="store_true", help="List all bio data")
    parser.add_argument("--backup", help="Create backup at specified path")
    parser.add_argument("--restore", help="Restore from backup at specified path")
    parser.add_argument("--optimize", action="store_true", help="Optimize vector store")
    
    args = parser.parse_args()
    
    # Initialize vector store setup
    setup = VectorStoreSetup(args.config)
    
    try:
        if args.setup:
            if setup.setup_vector_store():
                print("Vector store setup completed successfully")
            else:
                print("Vector store setup failed")
                return 1
        
        if args.add_pdf:
            bio_data_id = setup.process_and_add_pdf(args.add_pdf, args.bio_type)
            if bio_data_id:
                print(f"Added PDF with bio data ID: {bio_data_id}")
            else:
                print("Failed to add PDF")
                return 1
        
        if args.add_json:
            bio_data_id = setup.add_bio_data_from_json(args.add_json)
            if bio_data_id:
                print(f"Added JSON with bio data ID: {bio_data_id}")
            else:
                print("Failed to add JSON")
                return 1
        
        if args.search_text:
            results = setup.search_similar_bio_data(query_text=args.search_text)
            print(f"Search results: {json.dumps(results, indent=2, default=str)}")
        
        if args.search_id:
            results = setup.search_similar_bio_data(query_bio_data_id=args.search_id)
            print(f"Search results: {json.dumps(results, indent=2, default=str)}")
        
        if args.stats:
            stats = setup.get_vector_store_stats()
            print(f"Vector store statistics: {json.dumps(stats, indent=2, default=str)}")
        
        if args.list:
            bio_data = setup.list_bio_data(args.bio_type)
            print(f"Bio data in {args.bio_type}: {json.dumps(bio_data, indent=2, default=str)}")
        
        if args.backup:
            backup_path = setup.backup_vector_store(args.backup)
            print(f"Created backup at: {backup_path}")
        
        if args.restore:
            if setup.restore_vector_store(args.restore):
                print(f"Restored from backup: {args.restore}")
            else:
                print("Failed to restore from backup")
                return 1
        
        if args.optimize:
            setup.optimize_vector_store()
            print("Vector store optimization completed")
        
    finally:
        setup.close()
    
    return 0


if __name__ == "__main__":
    exit(main())