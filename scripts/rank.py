#!/usr/bin/env python3
"""CLI entry point for the Redrob ranking pipeline.

Thin wrapper around ``redrob_ranker.pipeline.main`` so the package stays
import-clean while the command line lives in ``scripts/``.

Usage
-----
    python scripts/rank.py
    python scripts/rank.py --data-dir ./data --top-n 100 --no-ltr
    python scripts/rank.py --output ./submission.csv --no-filter
"""

import os
import sys

# Make the src/ package importable when run directly (no install required).
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from redrob_ranker.pipeline import main

if __name__ == "__main__":
    main()
