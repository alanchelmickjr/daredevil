"""Unknown-source tracking — persistent UNKNOWN-NNN identifiers (patent Claim 6).

Lets the system track an unidentified source across frames without enrolling it:
two frames whose embeddings are similar enough get the same UNKNOWN-NNN id.
"""
from __future__ import annotations

from typing import List, Sequence

from ..audio.utils import cosine


def _ema(old: Sequence[float], new: Sequence[float], alpha: float) -> List[float]:
    n = min(len(old), len(new))
    return [(1 - alpha) * old[i] + alpha * new[i] for i in range(n)]


class UnknownTracker:
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        self._sources: List[dict] = []
        self._counter = 0

    def assign(self, vector: Sequence[float]) -> str:
        best, best_score = None, -1.0
        for s in self._sources:
            score = cosine(vector, s["vector"])
            if score > best_score:
                best_score, best = score, s
        if best is not None and best_score >= self.threshold:
            best["vector"] = _ema(best["vector"], vector, 0.2)
            best["hits"] += 1
            return best["id"]
        self._counter += 1
        sid = f"UNKNOWN-{self._counter:03d}"
        self._sources.append({"id": sid, "vector": list(vector), "hits": 1})
        return sid

    @property
    def count(self) -> int:
        return len(self._sources)
