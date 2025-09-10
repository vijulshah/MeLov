"""
Main bio-data extraction script supporting multiple file formats.
"""

import sys
from pathlib import Path

# Handle imports for both script and module execution
try:
    # When run as module: python -m data_processing.extraction.main
    from ..utils.config_manager import ConfigManager
    from .bio_extractor import BioDataExtractor
except ImportError:
    # When run as script: python main.py
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bio_extractor import BioDataExtractor
    from utils.config_manager import ConfigManager


def main():
    """Main entry point for bio-data extraction."""
    # Initialize configuration
    config_manager = ConfigManager()

    # Validate paths
    if not config_manager.validate_paths():
        print("Error: Could not create required directories")
        sys.exit(1)

    # Initialize extractor
    extractor = BioDataExtractor(config_manager=config_manager)

    for bio_type in config_manager._config.processing_bio_types:

        print(f"\n=== Processing {bio_type} ===")

        if config_manager._config.batch_processing:
            # Batch processing
            print(f"🔄 Starting batch processing for {bio_type}...")
            results = extractor.batch_process_directory(bio_type, batch_size=config_manager._config.batch_size)

            successful = sum(1 for r in results if r.success)
            total = len(results)

            print(f"📊 Batch processing completed for {bio_type}: {successful}/{total} files processed successfully")

            # Image processing summary
            total_images = sum(len(r.extracted_images or []) for r in results if r.success)
            images_with_persons = sum(
                sum(1 for img in (r.extracted_images or []) if img.contains_person) for r in results if r.success
            )

            if total_images > 0:
                print(f"🖼️  Images processed: {total_images} total, {images_with_persons} with persons detected")

            if successful < total:
                print("❌ Some files failed to process. Check logs for details.")
        else:
            bio_files_path = config_manager.get_file_input_path(bio_type)
            print(f"🔄 Processing {bio_type} from: {bio_files_path}")
            input_path = Path(bio_files_path)

            # Get all files with supported extensions
            supported_files = []
            for ext in extractor.supported_extensions:
                pattern = f"*{ext}"
                supported_files.extend(input_path.glob(pattern))
                # Also check subdirectories recursively
                supported_files.extend(input_path.rglob(pattern))

            # Remove duplicates and sort
            supported_files = sorted(set(supported_files))

            if not supported_files:
                print(f"⚠️  No supported files found in: {input_path}")
                print(f"📁 Supported formats: {', '.join(extractor.get_supported_formats())}")
                continue

            print(f"📁 Found {len(supported_files)} supported files")

            # Process each file
            successful_files = 0
            total_files = len(supported_files)

            for file_path in supported_files:
                print(f"📄 Processing file: {file_path}")
                result = extractor.process_file(str(file_path), bio_type)

                if result.success:
                    successful_files += 1
                    print(f"✅ Successfully processed: {file_path}")
                    if result.bio_data:
                        print(f"📋 Extracted data for: {result.bio_data.personal_info.name}")
                        if hasattr(result, "output_path") and result.output_path:
                            print(f"📄 Saved to: {result.output_path}")

                        # Image processing summary for this file
                        if result.extracted_images:
                            print(f"🖼️  Processed {len(result.extracted_images)} images")
                            person_count = sum(1 for img in result.extracted_images if img.contains_person)
                            if person_count > 0:
                                print(f"👤 Found {person_count} images with persons")
                else:
                    print(f"❌ Failed to process: {file_path}")
                    if result.errors:
                        for error in result.errors:
                            print(f"   Error: {error}")
                print()  # Add spacing between files

            # Summary for this bio_type
            print(
                f"📊 Processing summary for {bio_type}: {successful_files}/{total_files} files processed successfully"
            )

            if successful_files == 0:
                print("❌ No files were processed successfully!")
            elif successful_files < total_files:
                print("⚠️  Some files failed to process. Check the error messages above for details.")

    # Cleanup resources
    extractor.cleanup()


if __name__ == "__main__":
    main()
