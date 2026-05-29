"""Slot interface — the contract every inference slot honours (patent Claim 4/13).

A slot takes a single source's audio window and returns a plain dict. Slots are
backend-agnostic: a real model (torch/ONNX) or a pure-Python heuristic both
satisfy the same interface, which is exactly what lets Daredevil run everywhere
and still accelerate on device. Slot D (user-defined) is just another Slot.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SlotResult:
    slot: str
    data: dict
    ms: float = 0.0


class Slot(abc.ABC):
    """Base class for all Stage-2 inference slots."""

    #: short stable identifier, e.g. "embedding", "events", "prosody"
    name: str = "slot"

    def __init__(self) -> None:
        self._warm = False

    @property
    def backend(self) -> str:
        """Which implementation answered: 'reference' | 'onnx' | 'fallback'."""
        return "fallback"

    def warmup(self) -> None:
        """Load models / allocate buffers once. Safe to call repeatedly."""
        self._warm = True

    @abc.abstractmethod
    def run(self, audio: List[float], sr: int, ctx: Optional[Dict] = None) -> dict:
        """Process one source's audio window and return a result dict."""
        raise NotImplementedError
