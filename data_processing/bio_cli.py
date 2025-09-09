#!/usr/bin/env python3
"""
Simple CLI utility for bio-data extraction.

This provides convenient commands for processing bio-data files.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_processing.extraction.main import process_my_biodata, process_people_biodata
from data_processing.extraction.bio_extractor import BioDataExtractor
from data_processing.utils.config_manager import ConfigManager


def extract_my_biodata(pdf_path: str) -> None:
    """Extract user's bio-data from PDF."""
    print(f"🔄 Processing your bio-data: {Path(pdf_path).name}")
    
    success = process_my_biodata(pdf_path)
    
    if success:
        print("✅ Your bio-data extracted successfully!")
        config = ConfigManager()
        output_dir = config.get_output_path("my_biodata")
        print(f"📁 Check output in: {output_dir}")
    else:
        print("❌ Failed to extract your bio-data")
        sys.exit(1)


def extract_people_biodata(pdf_path: str) -> None:
    """Extract other person's bio-data from PDF."""
    print(f"🔄 Processing other person's bio-data: {Path(pdf_path).name}")
    
    success = process_people_biodata(pdf_path)
    
    if success:
        print("✅ Person's bio-data extracted successfully!")
        config = ConfigManager()
        output_dir = config.get_output_path("ppl_biodata")
        print(f"📁 Check output in: {output_dir}")
    else:
        print("❌ Failed to extract person's bio-data")
        sys.exit(1)


def batch_extract(bio_type: str) -> None:
    """Batch extract all PDFs in a directory."""
    print(f"📦 Starting batch extraction for {bio_type}...")
    
    try:
        config = ConfigManager()
        extractor = BioDataExtractor(config)
        
        results = extractor.batch_process_directory(bio_type)
        
        successful = sum(1 for r in results if r.success)
        total = len(results)
        
        print(f"📊 Batch processing completed: {successful}/{total} files processed successfully")
        
        if successful > 0:
            output_dir = config.get_output_path(bio_type)
            print(f"📁 Check outputs in: {output_dir}")
        
        if successful < total:
            print("⚠️  Some files failed to process. Check logs for details.")
    
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        sys.exit(1)


def show_status() -> None:
    """Show current status of the bio-data extraction system."""
    print("📋 Bio-Data Extraction System Status")
    print("=" * 40)
    
    try:
        config = ConfigManager()
        
        # Check input directories
        my_input = Path(config.get_pdf_input_path("my_biodata"))
        ppl_input = Path(config.get_pdf_input_path("ppl_biodata"))
        
        my_pdfs = list(my_input.glob("*.pdf")) if my_input.exists() else []
        ppl_pdfs = list(ppl_input.glob("*.pdf")) if ppl_input.exists() else []
        
        print(f"\n📁 Input Directories:")
        print(f"   📋 My bio-data: {len(my_pdfs)} PDFs in {my_input}")
        print(f"   👥 People bio-data: {len(ppl_pdfs)} PDFs in {ppl_input}")
        
        # Check output directories
        my_output = Path(config.get_output_path("my_biodata"))
        ppl_output = Path(config.get_output_path("ppl_biodata"))
        
        my_jsons = list(my_output.glob("*.json")) if my_output.exists() else []
        ppl_jsons = list(ppl_output.glob("*.json")) if ppl_output.exists() else []
        
        print(f"\n📂 Output Directories:")
        print(f"   📋 My bio-data: {len(my_jsons)} JSON files in {my_output}")
        print(f"   👥 People bio-data: {len(ppl_jsons)} JSON files in {ppl_output}")
        
        # Configuration info
        print(f"\n⚙️ Configuration:")
        print(f"   📄 Config file: {config.config_path}")
        print(f"   📝 Log level: {config.get_logging_config().get('level', 'INFO')}")
        
        print(f"\n🎯 Usage Tips:")
        if not my_pdfs and not ppl_pdfs:
            print("   💡 Add PDF files to input directories to start processing")
        else:
            print("   💡 Use 'batch-my' or 'batch-people' for bulk processing")
            print("   💡 Use 'extract-my' or 'extract-people' for single files")
    
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        sys.exit(1)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Bio-Data Extraction CLI Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bio_cli.py extract-my /path/to/my_resume.pdf
  python bio_cli.py extract-people /path/to/other_resume.pdf
  python bio_cli.py batch-my
  python bio_cli.py batch-people
  python bio_cli.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract my bio-data
    parser_my = subparsers.add_parser('extract-my', help='Extract your bio-data from PDF')
    parser_my.add_argument('pdf_path', help='Path to your bio-data PDF file')
    
    # Extract people's bio-data
    parser_people = subparsers.add_parser('extract-people', help='Extract other person\'s bio-data from PDF')
    parser_people.add_argument('pdf_path', help='Path to other person\'s bio-data PDF file')
    
    # Batch processing
    subparsers.add_parser('batch-my', help='Batch process all PDFs in my_biodata directory')
    subparsers.add_parser('batch-people', help='Batch process all PDFs in ppl_biodata directory')
    
    # Status
    subparsers.add_parser('status', help='Show system status and file counts')
    
    args = parser.parse_args()
    
    if args.command == 'extract-my':
        extract_my_biodata(args.pdf_path)
    elif args.command == 'extract-people':
        extract_people_biodata(args.pdf_path)
    elif args.command == 'batch-my':
        batch_extract('my_biodata')
    elif args.command == 'batch-people':
        batch_extract('ppl_biodata')
    elif args.command == 'status':
        show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
