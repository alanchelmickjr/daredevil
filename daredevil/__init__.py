"""Daredevil — local, private acoustic context for LLMs.

WHO is speaking, WHERE they are, WHAT is happening, and HOW they sound —
computed on-device, never sent to the cloud. The structured awareness map IS
the product: an LLM receives labeled, prioritized, spatially-located context
instead of raw audio.

    from daredevil import Pipeline
    p = Pipeline()
    p.enroll("alan", mic_seconds=3)
    context = p.listen(duration=1.0)

The software is MIT-licensed and runs everywhere (pure-Python fallback). The
hardware module is the patented part; this SDK is the open OAK-D-style stack.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .pipeline import Pipeline  # noqa: E402

__all__ = ["Pipeline", "__version__"]
