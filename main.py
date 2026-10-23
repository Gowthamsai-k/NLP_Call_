#!/usr/bin/env python3
"""Convenience root wrapper allowing `python main.py ...` from repository root."""
import sys
from pathlib import Path

# Add src to python path for direct execution
sys.path.insert(0, str(Path(__file__).parent / "src"))

from customer_support_pipeline.cli.main import main

if __name__ == "__main__":
    main()
