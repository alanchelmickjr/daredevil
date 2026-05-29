"""Unknown-source tracking — persistent UNKNOWN-NNN identifiers (patent Claim 6).

Lets the system track an unidentified source across frames without enrolling it:
two frames whose embeddings are similar enough get the same UNKNOWN-NNN id.

For sources that can't be discriminated by embedding alone (e.g. a mono TV feed
with multiple speakers), the tracker also considers spatial position and event
class continuity — same position + same class = likely same physical source.
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence

from ..audio.utils import cosine


def _ema(old: Sequence[float], new: Sequence[float], alpha: float) -> List[float]:
    n = min(len(old), len(new))
    return [(1 - alpha) * old[i] + alpha * new[i] for i in range(n)]


class UnknownTracker:
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        self._sources: List[dict] = []
        self._counter = 0

    def assign(self, vector: Sequence[float], position: Optional[dict] = None,
               event_class: Optional[str] = None, single_source: bool = False) -> str:
        now = time.monotonic()
        best, best_score = None, -1.0
        for s in self._sources:
            score = cosine(vector, s["vector"])
            # boost for same event class (speech stays speech)
            if event_class and s.get("event_class") == event_class:
                score += 0.15
            if position and s.get("position") == position:
                score += 0.10
            # recency boost
            age = now - s.get("last_seen", now)
            if age < 3.0:
                score += 0.10
            # single mic continuity — but only for same event class
            if single_source and age < 5.0 and event_class and s.get("event_class") == event_class:
                score += 0.20
            if score > best_score:
                best_score, best = score, s

        # SPRT-style: don't spawn a new source on one bad frame.
        # If the best source was recently active and close-ish, stick with it.
        if best is not None and best_score >= self.threshold:
            # Only update the stored vector if the raw cosine (without boosts) is strong
            raw = cosine(vector, best["vector"])
            if raw >= 0.5:
                best["vector"] = _ema(best["vector"], vector, 0.1)
            best["hits"] += 1
            best["last_seen"] = now
            best["event_class"] = event_class
            best["position"] = position
            best["misses"] = 0
            return best["id"]

        # Near-miss: score is close but below threshold — give benefit of the doubt
        # but only if the source was recently active (prevents permanent stickiness)
        if best is not None and best_score >= self.threshold - 0.15:
            age = now - best.get("last_seen", now)
            if age < 5.0:
                misses = best.get("misses", 0) + 1
                best["misses"] = misses
                if misses < 3:
                    best["last_seen"] = now
                    return best["id"]

        # Genuinely new source
        self._counter += 1
        sid = f"UNKNOWN-{self._counter:03d}"
        self._sources.append({
            "id": sid, "vector": list(vector), "hits": 1,
            "last_seen": now, "event_class": event_class, "position": position,
            "misses": 0,
        })
        return sid

    def prune(self, max_age: float = 30.0) -> None:
        """Remove sources not seen recently."""
        now = time.monotonic()
        self._sources = [s for s in self._sources if now - s.get("last_seen", now) < max_age]

    @property
    def count(self) -> int:
        return len(self._sources)
