# Bio Data Vector Store

A comprehensive FAISS-based vector database solution for storing and matching bio data profiles using semantic embeddings.

## Overview

This module provides a complete vector database implementation for bio data matching applications. It transforms structured bio data (extracted from PDFs using docling) into dense vector embeddings using sentence transformers, then stores them in FAISS indices for fast similarity search.

## Features

- **FAISS Integration**: High-performance vector similarity search using Facebook AI Similarity Search
- **Sentence Transformers**: Semantic embeddings using pre-trained transformer models
- **Configurable**: YAML-based configuration for all aspects of the system
- **Pydantic Models**: Type-safe data validation and serialization
- **Bio Data Processing**: Intelligent text processing optimized for bio data content
- **Similarity Search**: Advanced similarity matching with reranking and section weighting
- **Backup & Recovery**: Built-in backup and restore functionality
- **Performance Optimization**: Memory management, caching, and index optimization

## Architecture

```
data_vector_store/
├── config/
│   └── vector_store_config.yaml    # Configuration file
├── models/
│   └── vector_models.py            # Pydantic models for vector operations
├── utils/
│   ├── config_manager.py           # Configuration management
│   ├── text_processor.py           # Bio data text processing
│   └── embedding_manager.py        # Embedding generation and management
├── faiss_store.py                  # Main FAISS vector store implementation
└── setup_vector_store.py           # High-level setup and management API
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `faiss-cpu` or `faiss-gpu`: Vector similarity search engine
- `sentence-transformers`: Pre-trained transformer models for embeddings
- `torch`: Deep learning framework
- `numpy`: Numerical computing
- `pydantic`: Data validation
- `PyYAML`: Configuration file parsing

## Configuration

The system is configured via `data_vector_store/config/vector_store_config.yaml`:

### Vector Store Configuration
```yaml
vector_store:
  faiss:
    index_type: "IndexFlatIP"      # FAISS index type
    dimension: 384                 # Embedding dimension
    normalize_vectors: true        # Normalize for cosine similarity
  
  storage:
    base_path: "data_vector_store/indices"
    my_biodata_index: "my_biodata.faiss"
    ppl_biodata_index: "ppl_biodata.faiss"
  
  embeddings:
    model_name: "all-MiniLM-L6-v2"  # Sentence transformer model
    max_sequence_length: 512
    batch_size: 32
    device: "cpu"                   # or "cuda" for GPU
```

### Search Configuration
```yaml
search:
  default_k: 10                    # Default number of results
  similarity_threshold: 0.7        # Minimum similarity score
  enable_reranking: true           # Enable result reranking
  
  section_weights:                 # Weights for different bio sections
    personal_info: 0.2
    education: 0.15
    professional: 0.25
    interests: 0.25
    lifestyle: 0.1
    relationship: 0.05
```

## Usage

### Basic Setup

```python
from data_vector_store.setup_vector_store import VectorStoreSetup

# Initialize vector store
setup = VectorStoreSetup()

# Setup the vector store (creates directories, validates config)
setup.setup_vector_store()
```

### Adding Bio Data

From extracted JSON files:
```python
# Add bio data from JSON file
bio_data_id = setup.add_bio_data_from_json("data/processed/my_biodata/person1.json")
```

Directly from PDF files:
```python
# Process PDF and add to vector store
bio_data_id = setup.process_and_add_pdf("data/raw/ppl_biodata/person1.pdf", "ppl_biodata")
```

Batch processing:
```python
# Process all PDFs in a directory
bio_data_ids = setup.batch_process_directory("data/raw/ppl_biodata/", "ppl_biodata")
```

### Searching for Similar Bio Data

Text-based search:
```python
# Search using natural language text
results = setup.search_similar_bio_data(
    query_text="Software engineer interested in hiking and photography",
    target_bio_type="ppl_biodata",
    k=10,
    similarity_threshold=0.7
)
```

Bio data ID-based search:
```python
# Search using existing bio data as query
results = setup.search_similar_bio_data(
    query_bio_data_id="my_biodata_john_doe_20240909_143022",
    target_bio_type="ppl_biodata",
    k=5
)
```

### Managing the Vector Store

Get statistics:
```python
stats = setup.get_vector_store_stats()
print(f"Total vectors: {stats['total_vectors']}")
print(f"Index size: {stats['total_size_mb']:.2f} MB")
```

List stored bio data:
```python
bio_data_list = setup.list_bio_data("ppl_biodata")
for bio_data in bio_data_list:
    print(f"ID: {bio_data['bio_data_id']}, Name: {bio_data['personal_info']['name']}")
```

Backup and restore:
```python
# Create backup
backup_path = setup.backup_vector_store()

# Restore from backup
setup.restore_vector_store(backup_path)
```

Optimize performance:
```python
# Optimize indices for better search performance
setup.optimize_vector_store()
```

### Command Line Interface

The module provides a command-line interface for common operations:

```bash
# Setup vector store
python -m data_vector_store.setup_vector_store --setup

# Add PDF file
python -m data_vector_store.setup_vector_store --add-pdf "data/raw/ppl_biodata/person1.pdf" --bio-type ppl_biodata

# Search for similar bio data
python -m data_vector_store.setup_vector_store --search-text "Software engineer with hiking interests"

# Show statistics
python -m data_vector_store.setup_vector_store --stats

# Create backup
python -m data_vector_store.setup_vector_store --backup "backups/manual_backup"
```

## Advanced Usage

### Custom Embedding Fields

Configure which bio data fields to include in embeddings:

```python
from data_vector_store.utils.config_manager import VectorStoreConfigManager

config_manager = VectorStoreConfigManager()
embedding_fields = {
    'personal_info': ['name', 'location', 'age'],
    'professional': ['current_job', 'skills', 'industry'],
    'interests': ['hobbies', 'sports']
}

# This would typically be set in the YAML configuration
```

### Custom Similarity Scoring

Implement custom reranking logic:

```python
from data_vector_store.models.vector_models import SearchQuery

query = SearchQuery(
    query_text="Looking for outdoor enthusiasts",
    target_bio_type="ppl_biodata",
    k=20,
    enable_reranking=True,
    section_weights={
        'interests': 0.5,
        'lifestyle': 0.3,
        'personal_info': 0.2
    }
)

results = setup.vector_store.search_similar(query)
```

### Performance Tuning

For large datasets, consider:

1. **GPU Acceleration**: Set `device: "cuda"` in configuration
2. **Index Type**: Use `IndexIVFFlat` for datasets > 10,000 vectors
3. **Memory Mapping**: Enable for large indices
4. **Batch Processing**: Process multiple bio data items at once

```yaml
performance:
  max_memory_usage_gb: 8
  enable_memory_mapping: true
  enable_multiprocessing: true
  n_jobs: 4
```

## Data Flow

1. **Bio Data Extraction**: PDFs processed by docling → structured JSON
2. **Text Processing**: JSON bio data → processed text suitable for embeddings
3. **Embedding Generation**: Text → dense vectors using sentence transformers
4. **Index Storage**: Vectors stored in FAISS indices with metadata
5. **Similarity Search**: Query vectors → similar bio data retrieval
6. **Result Reranking**: Raw similarity scores → weighted and reranked results

## File Structure

The vector store creates the following file structure:

```
data_vector_store/
├── indices/
│   ├── my_biodata.faiss        # FAISS index for user's bio data
│   ├── my_biodata.json         # Metadata for user's bio data
│   ├── ppl_biodata.faiss       # FAISS index for other people's bio data
│   └── ppl_biodata.json        # Metadata for other people's bio data
├── backups/
│   └── backup_YYYYMMDD_HHMMSS/ # Timestamped backups
└── config/
    └── vector_store_config.yaml # Configuration file
```

## Error Handling

The system includes comprehensive error handling:

- **Missing Dependencies**: Graceful degradation if FAISS or sentence-transformers not available
- **Configuration Validation**: Validates all configuration parameters
- **File Operations**: Handles missing files, permissions, disk space
- **Memory Management**: Monitors memory usage and provides warnings
- **Index Corruption**: Automatic index rebuilding if corruption detected

## Performance Considerations

### Memory Usage
- Embeddings: ~1.5KB per bio data entry (384-dimensional vectors)
- Metadata: ~2-5KB per bio data entry (depending on content)
- Model: ~90MB for all-MiniLM-L6-v2 (smaller models available)

### Search Performance
- **Flat Index**: O(n) search time, suitable for < 10,000 vectors
- **IVF Index**: O(log n) search time, suitable for > 10,000 vectors
- **Typical Performance**: < 10ms for 1,000 vectors, < 100ms for 100,000 vectors

### Scaling Recommendations
- **< 1,000 bio data**: Use IndexFlatIP, single-threaded
- **1,000 - 10,000**: Use IndexFlatIP, enable multiprocessing
- **10,000 - 100,000**: Use IndexIVFFlat, GPU acceleration
- **> 100,000**: Consider distributed solutions, data sharding

## Troubleshooting

### Common Issues

1. **Import Errors**: Install missing dependencies
   ```bash
   pip install faiss-cpu sentence-transformers torch
   ```

2. **CUDA Errors**: Ensure CUDA-compatible PyTorch installation
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Memory Errors**: Reduce batch size or enable memory mapping
   ```yaml
   embeddings:
     batch_size: 16  # Reduce from 32
   performance:
     enable_memory_mapping: true
   ```

4. **Slow Search**: Optimize indices and consider GPU acceleration
   ```python
   setup.optimize_vector_store()
   ```

### Logging

Enable detailed logging for debugging:

```yaml
logging:
  level: "DEBUG"
  file: "logs/vector_store.log"
  enable_performance_logging: true
```

## Contributing

When contributing to the vector store module:

1. Follow PEP 8 coding standards
2. Add type hints to all functions
3. Include comprehensive docstrings
4. Write unit tests for new functionality
5. Update configuration schema if adding new parameters
6. Test with different embedding models and index types

## License

This project is part of the MeLov bio data matching application.
