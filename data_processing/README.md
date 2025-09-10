# Bio-Data Extraction Application

This application extracts bio-data from multiple file formats (PDF, images, DOCX) using docling and structures it into JSON format with proper validation using Pydantic models. It includes advanced image processing capabilities using Hugging Face models for person detection and image description.

## Features

- 🔍 **Multi-Format Support**: Extracts from PDF, images (JPG, PNG, etc.), and DOCX files
- 🖼️ **Image Processing**: Extracts images from PDFs and processes standalone images
- 🤖 **AI-Powered Analysis**: Uses Hugging Face models for person detection and image description
- 📊 **Structured Data**: Converts extracted text into structured JSON format
- ✅ **Data Validation**: Uses Pydantic models for robust data validation
- ⚙️ **Configurable**: YAML-based configuration for easy customization
- 📝 **Logging**: Comprehensive logging for debugging and monitoring
- 🎯 **Flexible Parsing**: Supports various bio-data formats and layouts
- 👤 **Dual Processing**: Separate handling for your bio-data vs. other people's bio-data
- 📦 **Batch Processing**: Process multiple files in a directory at once

## Supported File Formats

- **Documents**: PDF, DOCX, DOC
- **Images**: JPG, JPEG, PNG, BMP, TIFF, GIF

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create the necessary directories:
```bash
mkdir -p data/raw/my_biodata data/raw/ppl_biodata
mkdir -p data/processed/my_biodata data/processed/ppl_biodata
mkdir -p data/processed/my_biodata/images data/processed/ppl_biodata/images
mkdir -p logs local/hf_cache
```

## Configuration

The application uses a YAML configuration file located at `data_processing/config/bio_extraction_config.yaml`. You can customize:

- Input/output paths for both bio-data types
- Image processing settings
- Hugging Face model configurations
- Bio-data fields to extract
- Extraction settings
- Logging configuration

### Configuration Structure

```yaml
file_processing:
  my_biodata:
    input_path: "data/raw/my_biodata"
    output_path: "data/processed/my_biodata"
    images_output_path: "data/processed/my_biodata/images"
  ppl_biodata:
    input_path: "data/raw/ppl_biodata"
    output_path: "data/processed/ppl_biodata"
    images_output_path: "data/processed/ppl_biodata/images"
  supported_formats: ["pdf", "jpg", "jpeg", "png", "bmp", "tiff", "gif", "docx", "doc"]

image_processing:
  person_detection:
    model: "facebook/detr-resnet-50"
    confidence_threshold: 0.7
    device: "auto"
  image_description:
    model: "Salesforce/blip-image-captioning-large"
    max_length: 150
    device: "auto"
```

## Usage

### Command Line Interface

#### Process Your Bio-Data
```bash
# PDF file
python -m data_processing.extraction.main path/to/your/resume.pdf --bio-type my_biodata

# Image file
python -m data_processing.extraction.main path/to/your/photo.jpg --bio-type my_biodata

# DOCX file
python -m data_processing.extraction.main path/to/your/resume.docx --bio-type my_biodata
```

#### Process Other People's Bio-Data
```bash
python -m data_processing.extraction.main path/to/other/file.pdf --bio-type ppl_biodata
```

#### Batch Processing
```bash
# Process all supported files in my_biodata directory
python -m data_processing.extraction.main --batch --bio-type my_biodata

# Process all supported files in ppl_biodata directory
python -m data_processing.extraction.main --batch --bio-type ppl_biodata
```

#### Show Supported Formats
```bash
python -m data_processing.extraction.main --supported-formats
```

### Simple CLI Utility

Use the convenient CLI for common operations:

```bash
# Extract from any supported file
python bio_cli.py extract-my path/to/your/file.pdf
python bio_cli.py extract-people path/to/other/file.jpg

# Batch processing
python bio_cli.py batch-my
python bio_cli.py batch-people

# System status
python bio_cli.py status

# Show supported formats
python bio_cli.py formats
```

### Programmatic Usage

```python
from data_processing.extraction.bio_extractor import BioDataExtractor
from data_processing.utils.config_manager import ConfigManager

# Initialize
config = ConfigManager()
extractor = BioDataExtractor(config)

# Process any supported file
result = extractor.process_file("path/to/file.pdf", "my_biodata")

# Process with specific methods
result = extractor.process_my_biodata("path/to/your/file.jpg")
result = extractor.process_people_biodata("path/to/other/file.docx")

# Batch processing
my_results = extractor.batch_process_directory("my_biodata")
people_results = extractor.batch_process_directory("ppl_biodata")

if result.success:
    print(f"Extracted data for: {result.bio_data.personal_info.name}")

    # Access extracted images
    if result.extracted_images:
        print(f"Found {len(result.extracted_images)} images")
        for img in result.extracted_images:
            if img.contains_person:
                print(f"  - {img.file_path}: {img.description}")

# Clean up resources
extractor.cleanup()
```

## Data Structure

The extracted bio-data includes comprehensive information about images and processing:

```json
{
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
  },
  "interests": {
    "hobbies": ["reading", "hiking", "photography"]
  },
  "lifestyle": {
    "diet_preferences": ["vegetarian"],
    "exercise_habits": "regular"
  },
  "relationship": {
    "relationship_status": "single",
    "looking_for": ["long-term relationship"]
  },
  "images": [
    {
      "file_path": "data/processed/my_biodata/images/john_doe/profile_img_0.png",
      "original_filename": "john_doe_resume.pdf",
      "extracted_from_pdf": true,
      "page_number": 1,
      "contains_person": true,
      "person_confidence": 0.95,
      "description": "A professional portrait of a young man in business attire smiling at the camera",
      "description_model": "Salesforce/blip-image-captioning-large",
      "width": 300,
      "height": 400,
      "file_size_bytes": 45231
    }
  ],
  "metadata": {
    "source_file": "path/to/file.pdf",
    "bio_data_type": "my_biodata",
    "file_type": "pdf",
    "extraction_timestamp": "2025-09-09T12:00:00",
    "processing_time_seconds": 5.2,
    "extraction_method": "docling",
    "images_extracted": 1,
    "images_with_persons": 1
  }
}
```

## Image Processing Features

### Person Detection
- Uses Facebook's DETR model for object detection
- Identifies images containing people
- Provides confidence scores for detections
- Configurable confidence thresholds

### Image Description
- Uses Salesforce's BLIP model for image captioning
- Generates natural language descriptions
- Only processes images with detected persons (for efficiency)
- Configurable description length

### Image Extraction from PDFs
- Automatically extracts images embedded in PDF documents
- Saves images in organized directory structure
- Tracks page numbers and source information
- Supports multiple image formats

### Standalone Image Processing
- Processes standalone image files directly
- Copies images to organized storage structure
- Applies same AI analysis as extracted images

## Directory Structure

```
MeLov/
├── data/
│   ├── raw/
│   │   ├── my_biodata/           # Your bio-data files (any format)
│   │   └── ppl_biodata/          # Other people's bio-data files
│   └── processed/
│       ├── my_biodata/           # Processed JSON files from your bio-data
│       │   └── images/           # Extracted/processed images
│       └── ppl_biodata/          # Processed JSON files from other people
│           └── images/           # Extracted/processed images
├── local/
│   └── hf_cache/                 # Hugging Face model cache
├── data_processing/
│   ├── config/
│   │   └── bio_extraction_config.yaml
│   ├── extraction/
│   │   ├── bio_extractor.py      # Main extraction logic
│   │   └── main.py               # CLI interface
│   ├── models/
│   │   └── bio_models.py         # Pydantic models with image support
│   └── utils/
│       ├── config_manager.py     # Configuration management
│       └── image_processor.py    # AI-powered image processing
├── logs/                         # Log files
├── bio_cli.py                    # Simple CLI utility
├── requirements.txt              # Dependencies (updated for AI models)
└── README.md                     # This file
```

## AI Model Requirements

### System Requirements
- **GPU Support**: CUDA, MPS, or CPU
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Storage**: ~2GB for model cache

### Models Used
- **Person Detection**: facebook/detr-resnet-50
- **Image Captioning**: Salesforce/blip-image-captioning-large

Models are automatically downloaded on first use and cached locally.

## Best Practices

1. **File Quality**: Ensure high-quality files for better extraction
2. **GPU Usage**: Use GPU for faster image processing when available
3. **Batch Processing**: Use batch mode for processing multiple files
4. **Model Cache**: Keep model cache to avoid re-downloading
5. **Image Organization**: Images are automatically organized by source
6. **Privacy**: Separate storage for different bio-data types
7. **Resource Management**: Call `cleanup()` after processing

## Troubleshooting

### Common Issues

1. **Model Download Failures**: Check internet connection and disk space
2. **GPU Memory Issues**: Reduce batch size or use CPU
3. **Image Processing Errors**: Ensure PIL/Pillow supports your image format
4. **Slow Processing**: Use GPU acceleration when available

### Performance Tips

- Use GPU for faster image processing
- Process images in batches for efficiency
- Cache models locally to avoid re-downloading
- Use appropriate confidence thresholds

## License

This project is part of the MeLov bio-data matching application.

## Contributing

1. Follow PEP-8 coding standards
2. Add tests for new functionality
3. Update documentation
4. Use type hints for all functions
5. Consider privacy implications for image processing
