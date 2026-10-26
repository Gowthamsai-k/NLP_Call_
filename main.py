#!/usr/bin/env python3
"""Convenience root wrapper allowing `python main.py ...` from repository root."""
import sys
from pathlib import Path

# If dashboard is requested, launch directly without loading ML dependencies
if "--dashboard" in sys.argv:
    import argparse
    import dashboard_server
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args, _ = parser.parse_known_args()
    dashboard_server.run_server(port=args.port, host=args.host)
    sys.exit(0)

# Add src to python path for direct execution
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Apply compatibility shims before any deep learning / audio framework imports
import customer_support_pipeline.core.shims  # noqa: F401

from customer_support_pipeline.cli.main import main

if __name__ == "__main__":
    main()
