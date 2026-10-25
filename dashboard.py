#!/usr/bin/env python3
"""Start the local browser dashboard from a source checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from customer_support_pipeline.dashboard.server import main


if __name__ == "__main__":
    main()


export HF_TOKEN = "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"