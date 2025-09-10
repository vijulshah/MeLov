#!/usr/bin/env python3
"""
Wrapper script to run bio data extraction from any directory.
This script ensures proper module imports regardless of where it's run from.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run the main extraction script
from data_processing.extraction.main import main

if __name__ == "__main__":
    main()
