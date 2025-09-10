#!/usr/bin/env python3
"""
Simple CLI utility for bio-data extraction supporting multiple file formats.

This provides convenient commands for processing bio-data files including PDFs, images, and documents.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_processing.extraction.bio_extractor import BioDataExtractor
from data_processing.extraction.main import process_my_biodata, process_people_biodata
from data_processing.utils.config_manager import ConfigManager


def extract_my_biodata(file_path: str) -> None:
    """Extract user's bio-data from any supported file."""
    print(f"🔄 Processing your bio-data: {Path(file_path).name}")

    success = process_my_biodata(file_path)

    if success:
        print("✅ Your bio-data extracted successfully!")
        config = ConfigManager()
        output_dir = config.get_output_path("my_biodata")
        print(f"📁 Check output in: {output_dir}")

        # Check for images
        images_dir = config.get_images_output_path("my_biodata")
        if Path(images_dir).exists() and any(Path(images_dir).iterdir()):
            print(f"🖼️  Images saved in: {images_dir}")
    else:
        print("❌ Failed to extract your bio-data")
        sys.exit(1)


def extract_people_biodata(file_path: str) -> None:
    """Extract other person's bio-data from any supported file."""
    print(f"🔄 Processing other person's bio-data: {Path(file_path).name}")

    success = process_people_biodata(file_path)

    if success:
        print("✅ Person's bio-data extracted successfully!")
        config = ConfigManager()
        output_dir = config.get_output_path("ppl_biodata")
        print(f"📁 Check output in: {output_dir}")

        # Check for images
        images_dir = config.get_images_output_path("ppl_biodata")
        if Path(images_dir).exists() and any(Path(images_dir).iterdir()):
            print(f"🖼️  Images saved in: {images_dir}")
    else:
        print("❌ Failed to extract person's bio-data")
        sys.exit(1)


def batch_extract(bio_type: str) -> None:
    """Batch extract all supported files in a directory."""
    print(f"📦 Starting batch extraction for {bio_type}...")

    try:
        config = ConfigManager()
        extractor = BioDataExtractor(config)

        results = extractor.batch_process_directory(bio_type)

        successful = sum(1 for r in results if r.success)
        total = len(results)

        print(f"📊 Batch processing completed: {successful}/{total} files processed successfully")

        # Image processing summary
        total_images = sum(len(r.extracted_images or []) for r in results if r.success)
        images_with_persons = sum(
            sum(1 for img in (r.extracted_images or []) if img.contains_person) for r in results if r.success
        )

        if total_images > 0:
            print(f"🖼️  Images processed: {total_images} total, {images_with_persons} with persons detected")

        if successful > 0:
            output_dir = config.get_output_path(bio_type)
            print(f"📁 Check outputs in: {output_dir}")

            images_dir = config.get_images_output_path(bio_type)
            if Path(images_dir).exists() and any(Path(images_dir).iterdir()):
                print(f"🖼️  Images saved in: {images_dir}")

        if successful < total:
            print("⚠️  Some files failed to process. Check logs for details.")

        # Cleanup
        extractor.cleanup()

    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        sys.exit(1)


def show_supported_formats() -> None:
    """Show supported file formats."""
    print("📋 Supported File Formats")
    print("=" * 30)

    try:
        extractor = BioDataExtractor()
        formats = extractor.get_supported_formats()

        print("\n📄 Documents:")
        doc_formats = [f for f in formats if f in {".pdf", ".docx", ".doc"}]
        for fmt in doc_formats:
            print(f"   {fmt}")

        print("\n🖼️  Images:")
        img_formats = [f for f in formats if f not in {".pdf", ".docx", ".doc"}]
        for fmt in img_formats:
            print(f"   {fmt}")

        print(f"\n💡 Total supported formats: {len(formats)}")

    except Exception as e:
        print(f"❌ Error checking formats: {e}")


def show_status() -> None:
    """Show current status of the bio-data extraction system."""
    print("📋 Bio-Data Extraction System Status")
    print("=" * 40)

    try:
        config = ConfigManager()
        extractor = BioDataExtractor(config)

        # Check input directories
        my_input = Path(config.get_file_input_path("my_biodata"))
        ppl_input = Path(config.get_file_input_path("ppl_biodata"))

        # Count all supported files
        supported_exts = extractor.get_supported_formats()

        my_files = []
        ppl_files = []

        if my_input.exists():
            for ext in supported_exts:
                my_files.extend(my_input.glob(f"*{ext}"))
                my_files.extend(my_input.rglob(f"*{ext}"))

        if ppl_input.exists():
            for ext in supported_exts:
                ppl_files.extend(ppl_input.glob(f"*{ext}"))
                ppl_files.extend(ppl_input.rglob(f"*{ext}"))

        # Remove duplicates
        my_files = list(set(my_files))
        ppl_files = list(set(ppl_files))

        print(f"\n📁 Input Directories:")
        print(f"   📋 My bio-data: {len(my_files)} files in {my_input}")
        print(f"   👥 People bio-data: {len(ppl_files)} files in {ppl_input}")

        # Check output directories
        my_output = Path(config.get_output_path("my_biodata"))
        ppl_output = Path(config.get_output_path("ppl_biodata"))

        my_jsons = list(my_output.glob("*.json")) if my_output.exists() else []
        ppl_jsons = list(ppl_output.glob("*.json")) if ppl_output.exists() else []

        print(f"\n📂 Output Directories:")
        print(f"   📋 My bio-data: {len(my_jsons)} JSON files in {my_output}")
        print(f"   👥 People bio-data: {len(ppl_jsons)} JSON files in {ppl_output}")

        # Check image directories
        my_images = Path(config.get_images_output_path("my_biodata"))
        ppl_images = Path(config.get_images_output_path("ppl_biodata"))

        my_image_count = len(list(my_images.rglob("*.*"))) if my_images.exists() else 0
        ppl_image_count = len(list(ppl_images.rglob("*.*"))) if ppl_images.exists() else 0

        print(f"\n🖼️  Image Directories:")
        print(f"   📋 My bio-data: {my_image_count} images in {my_images}")
        print(f"   👥 People bio-data: {ppl_image_count} images in {ppl_images}")

        # Configuration info
        print(f"\n⚙️ Configuration:")
        print(f"   📄 Config file: {config.config_path}")
        print(f"   📝 Log level: {config.get_logging_config().get('level', 'INFO')}")
        print(f"   🖼️  Image processing: {'Enabled' if extractor.image_processor else 'Disabled'}")

        # Model info
        if extractor.image_processor:
            image_config = config.get_image_processing_config()
            person_model = image_config.get("person_detection", {}).get("model", "N/A")
            caption_model = image_config.get("image_description", {}).get("model", "N/A")
            print(f"   🤖 Person detection: {person_model}")
            print(f"   📝 Image captioning: {caption_model}")

        print(f"\n🎯 Usage Tips:")
        if not my_files and not ppl_files:
            print("   💡 Add supported files to input directories to start processing")
            print(f"   📋 Supported formats: {', '.join(supported_exts)}")
        else:
            print("   💡 Use 'batch-my' or 'batch-people' for bulk processing")
            print("   💡 Use 'extract-my' or 'extract-people' for single files")
            print("   💡 Use 'formats' to see all supported file formats")

    except Exception as e:
        print(f"❌ Error checking status: {e}")
        sys.exit(1)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Bio-Data Extraction CLI Utility - Multiple Format Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bio_cli.py extract-my /path/to/my_resume.pdf
  python bio_cli.py extract-my /path/to/my_photo.jpg
  python bio_cli.py extract-people /path/to/other_resume.docx
  python bio_cli.py batch-my
  python bio_cli.py batch-people
  python bio_cli.py status
  python bio_cli.py formats
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract my bio-data
    parser_my = subparsers.add_parser("extract-my", help="Extract your bio-data from any supported file")
    parser_my.add_argument("file_path", help="Path to your bio-data file (PDF, image, DOCX)")

    # Extract people's bio-data
    parser_people = subparsers.add_parser(
        "extract-people", help="Extract other person's bio-data from any supported file"
    )
    parser_people.add_argument("file_path", help="Path to other person's bio-data file (PDF, image, DOCX)")

    # Batch processing
    subparsers.add_parser("batch-my", help="Batch process all supported files in my_biodata directory")
    subparsers.add_parser("batch-people", help="Batch process all supported files in ppl_biodata directory")

    # Status and info
    subparsers.add_parser("status", help="Show system status and file counts")
    subparsers.add_parser("formats", help="Show supported file formats")

    args = parser.parse_args()

    if args.command == "extract-my":
        extract_my_biodata(args.file_path)
    elif args.command == "extract-people":
        extract_people_biodata(args.file_path)
    elif args.command == "batch-my":
        batch_extract("my_biodata")
    elif args.command == "batch-people":
        batch_extract("ppl_biodata")
    elif args.command == "status":
        show_status()
    elif args.command == "formats":
        show_supported_formats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
