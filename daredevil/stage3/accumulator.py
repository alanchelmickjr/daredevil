"""Quality-gated identity accumulator — collects good embeddings as they coalesce.

The tracker handles spatial/temporal continuity (WHERE). This handles WHO
independently: high-quality frames (Speech-classified, above VAD, strong energy)
feed into a per-source centroid that persists across tracker thrash. Identity
confidence grows as good chunks accumulate — doesn't require continuous signal.

This is the Shazam/Content ID/pyannote pattern: quality gate → per-source
accumulator → progressive match → decay. The SPRT runs against the accumulated
centroid, not frame-by-frame snapshots from a tracker that resets every second.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

from ..audio.utils import cosine


class SourceCentroid:
    """Running embedding centroid for one detected source."""

    __slots__ = ("centroid", "n_frames", "quality_sum", "last_update", "born")

    def __init__(self, embedding: List[float], quality: float):
        self.centroid = list(embedding)
        self.n_frames = 1
        self.quality_sum = quality
        self.last_update = time.monotonic()
        self.born = self.last_update

    def update(self, embedding: Sequence[float], quality: float, alpha: float) -> None:
        m = min(len(self.centroid), len(embedding))
        self.centroid = [
            (1.0 - alpha) * self.centroid[i] + alpha * embedding[i]
            for i in range(m)
        ]
        # Re-normalize to unit sphere so cosine stays meaningful.
        norm = math.sqrt(sum(x * x for x in self.centroid)) or 1.0
        self.centroid = [x / norm for x in self.centroid]
        self.n_frames += 1
        self.quality_sum += quality
        self.last_update = time.monotonic()

    @property
    def age(self) -> float:
        return time.monotonic() - self.born

    @property
    def stale(self) -> float:
        return time.monotonic() - self.last_update

    def confidence(self, tau: float = 5.0) -> float:
        """Enrollment-style saturation: more evidence → higher confidence."""
        return 1.0 - math.exp(-self.quality_sum / tau)


class IdentityAccumulator:
    """Collects good frames into source centroids, matches against enrolled DB.

    Decoupled from the tracker: a frame is assigned to an accumulator source by
    embedding similarity to existing centroids — not by track ID. This means
    identity evidence survives tracker resets, new track IDs, and frame gaps.
    """

    def __init__(self, assoc_threshold: float = 0.40,
                 alpha: float = 0.15,
                 expire_s: float = 30.0,
                 min_quality: float = 0.3):
        self.assoc_threshold = assoc_threshold
        self.alpha = alpha
        self.expire_s = expire_s
        self.min_quality = min_quality
        self._sources: Dict[int, SourceCentroid] = {}
        self._counter = 0

    def ingest(self, embedding: Sequence[float], quality: float) -> Optional[int]:
        """Ingest a frame if it passes the quality gate. Returns source ID or None.

        quality: [0, 1] — derived from energy (above VAD = non-zero).
        The SPRT decides identity; the gate just rejects silence.
        """
        if quality < self.min_quality:
            return None

        self._prune()

        # Associate to the best existing centroid.
        best_id, best_cos = None, -1.0
        for sid, src in self._sources.items():
            c = cosine(embedding, src.centroid)
            if c > best_cos:
                best_cos, best_id = c, sid

        if best_id is not None and best_cos >= self.assoc_threshold:
            self._sources[best_id].update(embedding, quality, self.alpha)
            return best_id

        # New source.
        self._counter += 1
        self._sources[self._counter] = SourceCentroid(embedding, quality)
        return self._counter

    def get_centroid(self, source_id: int) -> Optional[List[float]]:
        src = self._sources.get(source_id)
        return src.centroid if src else None

    def get_source(self, source_id: int) -> Optional[SourceCentroid]:
        return self._sources.get(source_id)

    def best_centroid_for(self, embedding: Sequence[float]) -> Optional[Tuple[int, List[float], float]]:
        """Find the centroid most similar to an embedding. Returns (id, centroid, cosine)."""
        best_id, best_cos, best_cent = None, -1.0, None
        for sid, src in self._sources.items():
            c = cosine(embedding, src.centroid)
            if c > best_cos:
                best_cos, best_id, best_cent = c, sid, src.centroid
        if best_id is None:
            return None
        return best_id, best_cent, best_cos

    @property
    def sources(self) -> Dict[int, SourceCentroid]:
        return self._sources

    def _prune(self) -> None:
        expired = [sid for sid, src in self._sources.items()
                   if src.stale > self.expire_s]
        for sid in expired:
            del self._sources[sid]
