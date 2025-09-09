"""
Main bio-data extraction script.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Literal

from .bio_extractor import BioDataExtractor
from ..utils.config_manager import ConfigManager


def main():
    """Main entry point for bio-data extraction."""
    parser = argparse.ArgumentParser(description="Extract bio-data from PDF files")
    parser.add_argument(
        "pdf_path",
        help="Path to PDF file to process"
    )
    parser.add_argument(
        "--bio-type",
        choices=["my_biodata", "ppl_biodata"],
        default="my_biodata",
        help="Type of bio data being processed (default: my_biodata)"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file",
        default=None
    )
    parser.add_argument(
        "--output",
        help="Output filename (optional)",
        default=None
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all PDF files in the configured input directory"
    )
    parser.add_argument(
        "--input-dir",
        help="Custom input directory for batch processing",
        default=None
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        config = ConfigManager(args.config)
        
        # Validate paths
        if not config.validate_paths():
            print("Error: Could not create required directories")
            sys.exit(1)
        
        # Initialize extractor
        extractor = BioDataExtractor(config)
        
        if args.batch:
            # Batch processing
            print(f"🔄 Starting batch processing for {args.bio_type}...")
            results = extractor.batch_process_directory(args.bio_type, args.input_dir)
            
            successful = sum(1 for r in results if r.success)
            total = len(results)
            
            print(f"📊 Batch processing completed: {successful}/{total} files processed successfully")
            
            if successful < total:
                print("❌ Some files failed to process. Check logs for details.")
                sys.exit(1)
        else:
            # Single file processing
            print(f"🔄 Processing {args.bio_type} from: {args.pdf_path}")
            result = extractor.process_pdf_file(args.pdf_path, args.bio_type, args.output)
            
            if result.success:
                print(f"✅ Bio-data extraction completed successfully!")
                if result.bio_data:
                    print(f"📋 Extracted data for: {result.bio_data.personal_info.name}")
                    print(f"📁 Output saved to: {config.get_output_path(args.bio_type)}")
                    if hasattr(result, 'output_path'):
                        print(f"📄 File: {result.output_path}")
            else:
                print("❌ Bio-data extraction failed!")
                if result.errors:
                    for error in result.errors:
                        print(f"   Error: {error}")
                sys.exit(1)
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


def process_my_biodata(pdf_path: str, output_name: Optional[str] = None) -> bool:
    """
    Convenience function to process user's bio-data.
    
    Args:
        pdf_path: Path to PDF file containing user's bio-data
        output_name: Custom output filename
        
    Returns:
        True if successful
    """
    try:
        config = ConfigManager()
        config.validate_paths()
        
        extractor = BioDataExtractor(config)
        result = extractor.process_my_biodata(pdf_path, output_name)
        
        return result.success
    except Exception as e:
        print(f"Error processing my bio-data: {e}")
        return False


def process_people_biodata(pdf_path: str, output_name: Optional[str] = None) -> bool:
    """
    Convenience function to process other people's bio-data.
    
    Args:
        pdf_path: Path to PDF file containing other person's bio-data
        output_name: Custom output filename
        
    Returns:
        True if successful
    """
    try:
        config = ConfigManager()
        config.validate_paths()
        
        extractor = BioDataExtractor(config)
        result = extractor.process_people_biodata(pdf_path, output_name)
        
        return result.success
    except Exception as e:
        print(f"Error processing people bio-data: {e}")
        return False


if __name__ == "__main__":
    main()
