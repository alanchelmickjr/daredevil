"""Multi-target track manager — persistent UNKNOWN-NNN contacts (patent Claim 6).

Modeled on a passive-sonar tracker rather than ad-hoc recency. Each frame's
detection is gated and associated to an existing track by embedding similarity
(and bearing, when a DOA azimuth is available); an unmatched detection starts a
new tentative track. Tracks are confirmed by M-of-N hit logic, coast through
misses, and are deleted after sustained silence. Bearing (azimuth) is smoothed
with an alpha-beta filter so a contact's direction is stable frame to frame.

All time constants and gates live in ``TrackerParams`` (config) — none are baked
into the logic here.
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence

from ..audio.utils import cosine
from ..config import TrackerParams

TENTATIVE, CONFIRMED, COASTING = "tentative", "confirmed", "coasting"


def _bearing_gap(a: float, b: float) -> float:
    """Smallest absolute angular distance between two bearings, in degrees."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


class UnknownTracker:
    def __init__(self, threshold: Optional[float] = None,
                 params: Optional[TrackerParams] = None):
        self.p = params or TrackerParams()
        # `threshold` (if given) overrides the association gate — kept for the
        # historical constructor signature.
        self.threshold = self.p.assoc_cosine if threshold is None else threshold
        self._tracks: List[dict] = []
        self._counter = 0

    # ------------------------------------------------------------- assign
    def assign(self, vector: Sequence[float], position: Optional[dict] = None,
               event_class: Optional[str] = None, single_source: bool = False) -> str:
        """Associate a detection to a track (or open a new one); return the track id."""
        now = time.monotonic()
        self._prune(now)
        az = position.get("azimuth") if position else None

        cand, best = None, -1e9
        for t in self._tracks:
            raw = cosine(vector, t["vector"])
            gap = None
            if az is not None and t.get("azimuth") is not None:
                gap = _bearing_gap(az, t["azimuth"])
                if gap > self.p.bearing_gate_deg:
                    continue  # outside the bearing gate — cannot be the same contact
            # Embedding gate, relaxed slightly when bearing corroborates.
            gate = self.threshold - (self.p.bearing_assist if gap is not None else 0.0)
            if raw < gate:
                continue
            score = raw
            if gap is not None:
                score += (1.0 - gap / self.p.bearing_gate_deg) * 0.1
            if now - t["last_seen"] < self.p.recency_window_s:
                score += self.p.recency_bonus
            if event_class and t.get("event_class") == event_class:
                score += 0.05
            if score > best:
                best, cand = score, t

        if cand is not None:
            self._update(cand, vector, az, event_class, now)
            return cand["id"]
        return self._new(vector, az, event_class, position, now)

    # ------------------------------------------------------------- internals
    def _update(self, t: dict, vector, az, event_class, now) -> None:
        e = self.p.embedding_ema
        m = min(len(t["vector"]), len(vector))
        t["vector"] = [(1.0 - e) * t["vector"][i] + e * vector[i] for i in range(m)]
        if az is not None:
            if t.get("azimuth") is None:
                t["azimuth"], t["az_rate"] = az, 0.0
            else:
                # alpha-beta filter on the shortest signed bearing residual
                resid = ((az - t["azimuth"] + 180.0) % 360.0) - 180.0
                t["azimuth"] = (t["azimuth"] + self.p.bearing_alpha * resid) % 360.0
                t["az_rate"] = t["az_rate"] + self.p.bearing_beta * resid
        t["event_class"] = event_class
        t["hits"] += 1
        t["last_seen"] = now
        if t["hits"] >= self.p.confirm_hits and (now - t["born"]) <= self.p.confirm_window_s:
            t["status"] = CONFIRMED
        elif t["status"] == COASTING:
            t["status"] = CONFIRMED if t["hits"] >= self.p.confirm_hits else TENTATIVE

    def _new(self, vector, az, event_class, position, now) -> str:
        self._counter += 1
        sid = f"UNKNOWN-{self._counter:03d}"
        self._tracks.append({
            "id": sid, "vector": list(vector), "hits": 1,
            "born": now, "last_seen": now, "status": TENTATIVE,
            "event_class": event_class, "azimuth": az, "az_rate": 0.0,
            "position": position,
        })
        return sid

    def _prune(self, now: float) -> None:
        for t in self._tracks:
            if t["status"] != COASTING and (now - t["last_seen"]) > self.p.coast_s:
                t["status"] = COASTING
        self._tracks = [t for t in self._tracks if (now - t["last_seen"]) <= self.p.delete_s]

    def prune(self, max_age: Optional[float] = None) -> None:
        self._prune(time.monotonic())

    # ------------------------------------------------------------- queries
    def live_ids(self) -> List[str]:
        return [t["id"] for t in self._tracks]

    def status_of(self, sid: str) -> Optional[str]:
        for t in self._tracks:
            if t["id"] == sid:
                return t["status"]
        return None

    @property
    def _sources(self):
        return self._tracks

    @property
    def count(self) -> int:
        return len(self._tracks)
