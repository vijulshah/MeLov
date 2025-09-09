# Bio-Data Extraction Application

This application extracts bio-data from PDF files using docling and structures it into JSON format with proper validation using Pydantic models. It supports processing both your own bio-data and other people's bio-data, saving them in separate locations for matching purposes.

## Features

- 🔍 **PDF Extraction**: Uses docling for high-quality PDF text and table extraction
- 📊 **Structured Data**: Converts extracted text into structured JSON format
- ✅ **Data Validation**: Uses Pydantic models for robust data validation
- ⚙️ **Configurable**: YAML-based configuration for easy customization
- 📝 **Logging**: Comprehensive logging for debugging and monitoring
- 🎯 **Flexible Parsing**: Supports various bio-data formats and layouts
- 👤 **Dual Processing**: Separate handling for your bio-data vs. other people's bio-data
- 📦 **Batch Processing**: Process multiple PDFs in a directory at once

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create the necessary directories:
```bash
mkdir -p data/my_biodata data/ppl_biodata data/processed/my_biodata data/processed/ppl_biodata logs
```

## Configuration

The application uses a YAML configuration file located at `data_processing/config/bio_extraction_config.yaml`. You can customize:

- Input/output paths for both bio-data types
- Bio-data fields to extract
- Extraction settings
- Logging configuration

### Configuration Structure

```yaml
pdf_processing:
  my_biodata:
    input_path: "data/my_biodata"
    output_path: "data/processed/my_biodata"
  ppl_biodata:
    input_path: "data/ppl_biodata"
    output_path: "data/processed/ppl_biodata"
```

## Usage

### Command Line

#### Process Your Bio-Data
```bash
python -m data_processing.extraction.main path/to/your/resume.pdf --bio-type my_biodata
```

#### Process Other People's Bio-Data
```bash
python -m data_processing.extraction.main path/to/other/resume.pdf --bio-type ppl_biodata
```

#### Batch Processing
```bash
# Process all PDFs in my_biodata directory
python -m data_processing.extraction.main --batch --bio-type my_biodata

# Process all PDFs in ppl_biodata directory
python -m data_processing.extraction.main --batch --bio-type ppl_biodata
```

#### With Custom Configuration
```bash
python -m data_processing.extraction.main path/to/resume.pdf --bio-type my_biodata --config custom_config.yaml
```

### Programmatic Usage

```python
from data_processing.extraction.bio_extractor import BioDataExtractor
from data_processing.utils.config_manager import ConfigManager

# Initialize
config = ConfigManager()
extractor = BioDataExtractor(config)

# Process your bio-data
result = extractor.process_my_biodata("path/to/your/resume.pdf")

# Process other people's bio-data
result = extractor.process_people_biodata("path/to/other/resume.pdf")

# Batch processing
my_results = extractor.batch_process_directory("my_biodata")
people_results = extractor.batch_process_directory("ppl_biodata")

if result.success:
    print(f"Extracted data for: {result.bio_data.personal_info.name}")
    # Access structured data
    bio_data = result.bio_data
```

### Convenience Functions

```python
from data_processing.extraction.main import process_my_biodata, process_people_biodata

# Simple processing functions
success = process_my_biodata("path/to/your/resume.pdf")
success = process_people_biodata("path/to/other/resume.pdf")
```

## Data Structure

The extracted bio-data follows this structure and includes metadata about the processing type:

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
  "metadata": {
    "source_file": "path/to/resume.pdf",
    "bio_data_type": "my_biodata",
    "extraction_timestamp": "2025-09-09T12:00:00",
    "processing_time_seconds": 2.5,
    "extraction_method": "docling"
  }
}
```

## Directory Structure

The application organizes data into separate directories for different purposes:

```
MeLov/
├── data/
│   ├── my_biodata/           # Your bio-data PDF files
│   ├── ppl_biodata/          # Other people's bio-data PDF files
│   └── processed/
│       ├── my_biodata/       # Processed JSON files from your bio-data
│       └── ppl_biodata/      # Processed JSON files from other people's bio-data
├── data_processing/
│   ├── config/
│   │   └── bio_extraction_config.yaml
│   ├── extraction/
│   │   ├── bio_extractor.py  # Main extraction logic
│   │   └── main.py           # CLI interface
│   ├── models/
│   │   └── bio_models.py     # Pydantic models
│   └── utils/
│       └── config_manager.py # Configuration management
├── logs/                     # Log files
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Supported Bio-Data Fields

- **Personal Information**: Name, age, gender, location, contact details
- **Education**: Degree, institution, graduation year, GPA, certifications
- **Professional**: Job title, company, experience, skills, industry
- **Interests**: Hobbies, sports, music, travel preferences
- **Lifestyle**: Diet, exercise, smoking/drinking habits, pets
- **Relationship**: Status, preferences, looking for
- **Metadata**: Processing information including bio-data type classification

## Best Practices

1. **PDF Quality**: Ensure PDFs are text-based or high-quality scanned documents
2. **Consistent Format**: Use consistent formatting in bio-data PDFs for better extraction
3. **Configuration**: Customize the YAML config for your specific use case
4. **Validation**: Review extracted data and provide feedback for improvements
5. **Privacy**: Ensure sensitive data is handled according to privacy requirements
6. **Organization**: Keep your bio-data and other people's bio-data in separate directories
7. **Batch Processing**: Use batch processing for efficient handling of multiple files

## Bio-Data Matching Workflow

This extraction system is designed to support a bio-data matching application:

1. **Your Bio-Data**: Process your own bio-data and save to `data/processed/my_biodata/`
2. **Other People's Bio-Data**: Process potential matches and save to `data/processed/ppl_biodata/`
3. **Matching**: Use the structured JSON files for similarity analysis and matching algorithms
4. **Privacy**: Separate storage ensures proper data handling and privacy controls

## Extending the Application

### Adding New Fields

1. Update the Pydantic models in `data_processing/models/bio_models.py`
2. Add extraction logic in `data_processing/extraction/bio_extractor.py`
3. Update the configuration file to include new fields

### Custom Parsers

You can create custom parsers for specific PDF formats by extending the `BioDataExtractor` class.

### Adding New Bio-Data Types

To add support for additional bio-data categories:

1. Add new enum values to `BioDataType` in `bio_models.py`
2. Update the configuration file with new paths
3. Extend the `ConfigManager` methods to handle new types

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure docling is installed: `pip install docling`
2. **PDF Not Found**: Check the file path and ensure the PDF exists
3. **Extraction Failures**: Check the logs for detailed error messages
4. **Poor Quality**: Try with higher-quality PDF files

### Logs

Check the log files in the `logs/` directory for detailed information about the extraction process.

## Contributing

1. Follow PEP-8 coding standards
2. Add tests for new functionality
3. Update documentation
4. Use type hints for all functions

## License

This project is part of the MeLov bio-data matching application.
