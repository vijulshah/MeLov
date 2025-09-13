"""Profile Index Generator - Creates a comprehensive JSON index of all profile data."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict


class ProfileIndexer:
    """Creates a comprehensive index of all profile data with metadata."""

    def __init__(self, data_root: str = "./data"):
        self.data_root = Path(data_root)
        self.raw_data_path = self.data_root / "raw"
        self.processed_data_path = self.data_root / "processed"

        # File extensions to categorize
        self.image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".webp",
        }
        self.pdf_extensions = {".pdf"}
        self.document_extensions = {".doc", ".docx", ".txt", ".rtf"}

    def get_file_info(self, file_path: Path) -> Dict:
        """Get detailed information about a file."""
        try:
            stat = file_path.stat()
            return {
                "path": str(file_path.resolve()),
                "relative_path": str(file_path.relative_to(self.data_root)),
                "name": file_path.name,
                "extension": file_path.suffix.lower(),
                "size_bytes": stat.st_size,
                "size_human": self._format_bytes(stat.st_size),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        except Exception as e:
            return {
                "path": str(file_path.resolve()),
                "relative_path": str(file_path.relative_to(self.data_root)),
                "name": file_path.name,
                "extension": file_path.suffix.lower(),
                "error": str(e),
            }

    def _format_bytes(self, bytes_size: int) -> str:
        """Convert bytes to human readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"

    def categorize_file(self, file_path: Path) -> str:
        """Categorize file based on extension."""
        ext = file_path.suffix.lower()
        if ext in self.image_extensions:
            return "image"
        elif ext in self.pdf_extensions:
            return "pdf"
        elif ext in self.document_extensions:
            return "document"
        else:
            return "other"

    def scan_profile_directory(self, profile_path: Path, source: str, category: str) -> Dict:
        """Scan a single profile directory and return file information."""
        profile_data = {
            "profile_name": profile_path.name,
            "source": source,
            "category": category,
            "raw_data_path": str(profile_path.resolve()),
            "files": {"pdf": [], "image": [], "document": [], "other": []},
            "processed_data": None,
            "statistics": {
                "total_files": 0,
                "total_size_bytes": 0,
                "file_counts": {"pdf": 0, "image": 0, "document": 0, "other": 0},
            },
        }

        # Scan raw data files
        if profile_path.exists() and profile_path.is_dir():
            for file_path in profile_path.rglob("*"):
                if file_path.is_file():
                    file_info = self.get_file_info(file_path)
                    file_category = self.categorize_file(file_path)

                    profile_data["files"][file_category].append(file_info)
                    profile_data["statistics"]["file_counts"][file_category] += 1
                    profile_data["statistics"]["total_files"] += 1

                    if "size_bytes" in file_info:
                        profile_data["statistics"]["total_size_bytes"] += file_info["size_bytes"]

        # Check for processed data
        processed_profile_path = self._get_processed_path(profile_path, source, category)
        if processed_profile_path and processed_profile_path.exists():
            profile_data["processed_data"] = self._scan_processed_data(processed_profile_path)

        # Add human readable total size
        profile_data["statistics"]["total_size_human"] = self._format_bytes(
            profile_data["statistics"]["total_size_bytes"]
        )

        return profile_data

    def _get_processed_path(self, raw_profile_path: Path, source: str, category: str) -> Path:
        """Get the corresponding processed data path for a raw profile."""
        if category == "my_biodata":
            # For my_biodata, the processed structure is different
            profile_name = raw_profile_path.name if raw_profile_path.is_dir() else raw_profile_path.stem
            return self.processed_data_path / category / profile_name
        else:
            # For ppl_biodata, maintain the source structure
            relative_path = raw_profile_path.relative_to(self.raw_data_path / category)
            return self.processed_data_path / category / relative_path

    def _scan_processed_data(self, processed_path: Path) -> Dict:
        """Scan processed data directory."""
        processed_info = {
            "path": str(processed_path.resolve()),
            "subdirectories": {},
            "files": [],
            "statistics": {
                "total_files": 0,
                "total_size_bytes": 0,
                "subdirectory_counts": {},
            },
        }

        if not processed_path.exists():
            return processed_info

        # Scan subdirectories
        for item in processed_path.iterdir():
            if item.is_dir():
                subdir_info = {
                    "path": str(item.resolve()),
                    "files": [],
                    "file_count": 0,
                    "total_size_bytes": 0,
                }

                for file_path in item.rglob("*"):
                    if file_path.is_file():
                        file_info = self.get_file_info(file_path)
                        subdir_info["files"].append(file_info)
                        subdir_info["file_count"] += 1

                        if "size_bytes" in file_info:
                            subdir_info["total_size_bytes"] += file_info["size_bytes"]

                subdir_info["total_size_human"] = self._format_bytes(subdir_info["total_size_bytes"])
                processed_info["subdirectories"][item.name] = subdir_info
                processed_info["statistics"]["subdirectory_counts"][item.name] = subdir_info["file_count"]
                processed_info["statistics"]["total_files"] += subdir_info["file_count"]
                processed_info["statistics"]["total_size_bytes"] += subdir_info["total_size_bytes"]

            elif item.is_file():
                file_info = self.get_file_info(item)
                processed_info["files"].append(file_info)
                processed_info["statistics"]["total_files"] += 1

                if "size_bytes" in file_info:
                    processed_info["statistics"]["total_size_bytes"] += file_info["size_bytes"]

        processed_info["statistics"]["total_size_human"] = self._format_bytes(
            processed_info["statistics"]["total_size_bytes"]
        )

        return processed_info

    def create_comprehensive_index(self) -> Dict:
        """Create a comprehensive index of all profile data."""
        index = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "data_root": str(self.data_root.resolve()),
                "scan_summary": {
                    "total_profiles": 0,
                    "profiles_by_source": {},
                    "total_files": 0,
                    "total_size_bytes": 0,
                    "file_type_summary": {
                        "pdf": 0,
                        "image": 0,
                        "document": 0,
                        "other": 0,
                    },
                    "processing_status": {
                        "processed_profiles": 0,
                        "unprocessed_profiles": 0,
                    },
                },
            },
            "profiles": {},
        }

        # Scan my_biodata
        my_biodata_path = self.raw_data_path / "my_biodata"
        if my_biodata_path.exists():
            # Handle files directly in my_biodata (like PDFs)
            for item in my_biodata_path.iterdir():
                if item.is_file():
                    # Create a profile entry for the file
                    profile_name = item.stem
                    profile_data = self.scan_profile_directory(item.parent, "my_biodata", "my_biodata")
                    profile_data["profile_name"] = profile_name
                    # Filter files to only include this specific file
                    all_files = []
                    for category, files in profile_data["files"].items():
                        all_files.extend([f for f in files if Path(f["path"]).name == item.name])

                    # Reset file categories and add only the specific file
                    for category in profile_data["files"]:
                        profile_data["files"][category] = []

                    for file_info in all_files:
                        file_category = self.categorize_file(Path(file_info["path"]))
                        profile_data["files"][file_category].append(file_info)

                    index["profiles"][profile_name] = profile_data

                elif item.is_dir():
                    # Handle directories in my_biodata
                    profile_data = self.scan_profile_directory(item, "my_biodata", "my_biodata")
                    index["profiles"][item.name] = profile_data

        # Scan ppl_biodata sources
        ppl_biodata_path = self.raw_data_path / "ppl_biodata"
        if ppl_biodata_path.exists():
            for source_dir in ppl_biodata_path.iterdir():
                if source_dir.is_dir():
                    source_name = source_dir.name

                    for profile_dir in source_dir.iterdir():
                        if profile_dir.is_dir():
                            profile_data = self.scan_profile_directory(profile_dir, source_name, "ppl_biodata")
                            profile_key = f"{source_name}_{profile_dir.name}"
                            index["profiles"][profile_key] = profile_data

        # Calculate summary statistics
        for profile_name, profile_data in index["profiles"].items():
            index["metadata"]["scan_summary"]["total_profiles"] += 1

            source = profile_data["source"]
            if source not in index["metadata"]["scan_summary"]["profiles_by_source"]:
                index["metadata"]["scan_summary"]["profiles_by_source"][source] = 0
            index["metadata"]["scan_summary"]["profiles_by_source"][source] += 1

            index["metadata"]["scan_summary"]["total_files"] += profile_data["statistics"]["total_files"]
            index["metadata"]["scan_summary"]["total_size_bytes"] += profile_data["statistics"]["total_size_bytes"]

            for file_type, count in profile_data["statistics"]["file_counts"].items():
                index["metadata"]["scan_summary"]["file_type_summary"][file_type] += count

            if profile_data["processed_data"] and profile_data["processed_data"]["statistics"]["total_files"] > 0:
                index["metadata"]["scan_summary"]["processing_status"]["processed_profiles"] += 1
            else:
                index["metadata"]["scan_summary"]["processing_status"]["unprocessed_profiles"] += 1

        # Add human readable total size
        index["metadata"]["scan_summary"]["total_size_human"] = self._format_bytes(
            index["metadata"]["scan_summary"]["total_size_bytes"]
        )

        # Create comprehensive file paths listing by category
        index["file_paths_by_category"] = self._create_file_paths_listing(index["profiles"])

        return index

    def _create_file_paths_listing(self, profiles: Dict) -> Dict:
        """Create a comprehensive listing of all file paths organized by category."""
        paths_by_category = {"pdf": [], "image": [], "document": [], "other": []}

        for profile_name, profile_data in profiles.items():
            for category, files in profile_data["files"].items():
                for file_info in files:
                    path_entry = {
                        "path": file_info["path"],
                        "relative_path": file_info["relative_path"],
                        "profile": profile_name,
                        "source": profile_data["source"],
                        "filename": file_info["name"],
                        "size_bytes": file_info.get("size_bytes", 0),
                        "size_human": file_info.get("size_human", "Unknown"),
                    }
                    paths_by_category[category].append(path_entry)

        # Sort paths within each category by profile name, then by filename
        for category in paths_by_category:
            paths_by_category[category].sort(key=lambda x: (x["profile"], x["filename"]))

        return paths_by_category

    def save_index(self, index: Dict, output_file: str = "profile_index.json") -> str:
        """Save the index to a JSON file."""
        output_path = Path(output_file)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        return str(output_path.resolve())


def main():
    """Main function to create and save the profile index."""
    print("Creating comprehensive profile index...")

    indexer = ProfileIndexer()
    index = indexer.create_comprehensive_index()

    output_file = indexer.save_index(index)

    print(f"Profile index created successfully!")
    print(f"Output file: {output_file}")
    print(f"\nSummary:")
    print(f"- Total profiles: {index['metadata']['scan_summary']['total_profiles']}")
    print(f"- Total files: {index['metadata']['scan_summary']['total_files']}")
    print(f"- Total size: {index['metadata']['scan_summary']['total_size_human']}")
    print(f"- Processed profiles: {index['metadata']['scan_summary']['processing_status']['processed_profiles']}")
    print(f"- Unprocessed profiles: {index['metadata']['scan_summary']['processing_status']['unprocessed_profiles']}")

    print(f"\nFiles by type:")
    for file_type, count in index["metadata"]["scan_summary"]["file_type_summary"].items():
        if count > 0:
            print(f"- {file_type}: {count}")

    print(f"\nProfiles by source:")
    for source, count in index["metadata"]["scan_summary"]["profiles_by_source"].items():
        print(f"- {source}: {count}")

    print(f"\nFile paths by category:")
    for category, paths in index["file_paths_by_category"].items():
        if paths:
            print(f"- {category}: {len(paths)} files")

    print(f"\nDetailed file paths have been saved to the JSON file for further analysis.")


if __name__ == "__main__":
    main()
