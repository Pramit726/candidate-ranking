"""Redrob evidence-grounded candidate ranker.

A modular, CPU-only pipeline that scores candidates on three axes
(Tech / Context / Behavior), ranks them with three interchangeable strategies,
and emits a grounded one-sentence reason per candidate.

Public entry point:
    from redrob_ranker.pipeline import main
"""

__version__ = "1.0.0"
