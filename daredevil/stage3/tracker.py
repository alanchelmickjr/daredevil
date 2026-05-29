"""Unknown-source tracking — persistent UNKNOWN-NNN identifiers (patent Claim 6).

Uses a slot-based assignment inspired by SORT (Simple Online Realtime Tracking):
each frame's detections are assigned to existing tracks via best-match, with a
low threshold for continuation (the same source sounds different frame to frame).
Tracks have states: ACTIVE → DORMANT → removed. A source keeps its ID across
silence gaps by staying DORMANT until heard again or timed out.
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence

from ..audio.utils import cosine


class UnknownTracker:
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        self._tracks: List[dict] = []
        self._counter = 0
        self._max_dormant = 15.0  # seconds before a dormant track is removed

    def assign(self, vector: Sequence[float], position: Optional[dict] = None,
               event_class: Optional[str] = None, single_source: bool = False) -> str:
        """Assign a detection to an existing track or create a new one.

        Strategy: find the most recent active track. For mono/single-source,
        the most recently active track gets priority — this handles the
        "same physical source, different embeddings" case by trusting recency.
        For multi-source, use cosine to discriminate.
        """
        now = time.monotonic()
        self._prune(now)

        if not self._tracks:
            return self._new_track(vector, now, event_class, position)

        if single_source:
            # Mono mic: only one physical source at a time. The most recent
            # active track IS this source unless it's been dormant too long.
            active = [t for t in self._tracks if now - t["last_seen"] < 5.0]
            if active:
                best = max(active, key=lambda t: t["last_seen"])
                best["last_seen"] = now
                best["event_class"] = event_class
                best["hits"] += 1
                return best["id"]
            # Nothing recent — new source
            return self._new_track(vector, now, event_class, position)

        # Multi-source: use cosine + recency to find best match
        best, best_score = None, -1.0
        for t in self._tracks:
            age = now - t["last_seen"]
            if age > self._max_dormant:
                continue
            score = cosine(vector, t["vector"])
            if age < 3.0:
                score += 0.10
            if event_class and t.get("event_class") == event_class:
                score += 0.10
            if score > best_score:
                best_score, best = score, t

        if best is not None and best_score >= self.threshold:
            best["vector"] = list(vector)
            best["last_seen"] = now
            best["event_class"] = event_class
            best["hits"] += 1
            return best["id"]

        return self._new_track(vector, now, event_class, position)

    def _new_track(self, vector, now, event_class, position) -> str:
        self._counter += 1
        sid = f"UNKNOWN-{self._counter:03d}"
        self._tracks.append({
            "id": sid, "vector": list(vector), "hits": 1,
            "last_seen": now, "event_class": event_class, "position": position,
        })
        return sid

    def _prune(self, now: float) -> None:
        self._tracks = [t for t in self._tracks if now - t["last_seen"] < self._max_dormant]

    def prune(self, max_age: float = 30.0) -> None:
        now = time.monotonic()
        self._tracks = [t for t in self._tracks if now - t.get("last_seen", now) < max_age]

    @property
    def _sources(self):
        return self._tracks

    @property
    def count(self) -> int:
        return len(self._tracks)
