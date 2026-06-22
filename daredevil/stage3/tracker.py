"""Multi-target track manager — persistent UNKNOWN-NNN contacts (patent Claim 6).

Modeled on a passive-sonar tracker: each frame's detection is associated to an
existing track by NMF spectral features (frame-stable "same ongoing sound?") and
bearing gating. ECAPA embeddings are NOT used for association — they answer WHO,
not WHICH, and drift when accumulated. The tracker answers WHERE/WHEN only.

Tracks are confirmed by M-of-N hit logic, coast through misses, and are deleted
after sustained silence. Bearing (azimuth) is smoothed with an alpha-beta filter.

All time constants and gates live in ``TrackerParams`` (config).
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence

from ..audio.utils import cosine
from ..config import TrackerParams

log = logging.getLogger("daredevil.tracker")

TENTATIVE, CONFIRMED, COASTING = "tentative", "confirmed", "coasting"


def _bearing_gap(a: float, b: float) -> float:
    """Smallest absolute angular distance between two bearings, in degrees."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


class UnknownTracker:
    def __init__(self, threshold: Optional[float] = None,
                 params: Optional[TrackerParams] = None):
        self.p = params or TrackerParams()
        self.threshold = self.p.assoc_cosine if threshold is None else threshold
        self._tracks: List[dict] = []
        self._counter = 0

    # ------------------------------------------------------------- assign
    def assign(self, feature: Sequence[float], position: Optional[dict] = None,
               event_class: Optional[str] = None) -> str:
        """Associate a detection to a track (or open a new one); return the track id.

        `feature` is a frame-stable spectral feature (NMF activations or band
        envelope), NOT an ECAPA embedding. It answers "same ongoing sound?" which
        is stable frame-to-frame for the same physical source.
        """
        now = time.monotonic()
        self._prune(now)
        az = position.get("azimuth") if position else None

        cand, best = None, -1e9
        for t in self._tracks:
            raw = cosine(feature, t["feature"])
            gap = None
            if az is not None and t.get("azimuth") is not None:
                gap = _bearing_gap(az, t["azimuth"])
                if gap > self.p.bearing_gate_deg:
                    continue
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
            self._update(cand, feature, az, event_class, now)
            return cand["id"]
        return self._new(feature, az, event_class, position, now)

    # ------------------------------------------------------------- internals
    def _update(self, t: dict, feature, az, event_class, now) -> None:
        # Replace the stored feature with the latest observation — no accumulation.
        # NMF features are frame-stable by construction; the latest is the best
        # representation of what the source sounds like RIGHT NOW.
        t["feature"] = list(feature)
        if az is not None:
            if t.get("azimuth") is None:
                t["azimuth"], t["az_rate"] = az, 0.0
            else:
                resid = ((az - t["azimuth"] + 180.0) % 360.0) - 180.0
                t["azimuth"] = (t["azimuth"] + self.p.bearing_alpha * resid) % 360.0
                t["az_rate"] = t["az_rate"] + self.p.bearing_beta * resid
        t["event_class"] = event_class
        t["hits"] += 1
        t["last_seen"] = now
        prev_status = t["status"]
        if t["hits"] >= self.p.confirm_hits and (now - t["born"]) <= self.p.confirm_window_s:
            t["status"] = CONFIRMED
        elif t["status"] == COASTING:
            t["status"] = CONFIRMED if t["hits"] >= self.p.confirm_hits else TENTATIVE
        if t["status"] != prev_status:
            log.debug("track %s: %s -> %s", t["id"], prev_status, t["status"])

    def _new(self, feature, az, event_class, position, now) -> str:
        self._counter += 1
        sid = f"UNKNOWN-{self._counter:03d}"
        self._tracks.append({
            "id": sid, "feature": list(feature), "hits": 1,
            "born": now, "last_seen": now, "status": TENTATIVE,
            "event_class": event_class, "azimuth": az, "az_rate": 0.0,
            "position": position,
        })
        log.debug("new track %s (class=%s, az=%s)", sid, event_class, az)
        return sid

    def _prune(self, now: float) -> None:
        for t in self._tracks:
            if t["status"] != COASTING and (now - t["last_seen"]) > self.p.coast_s:
                log.debug("track %s -> coasting (silent %.1fs)", t["id"], now - t["last_seen"])
                t["status"] = COASTING
        before = len(self._tracks)
        self._tracks = [t for t in self._tracks if (now - t["last_seen"]) <= self.p.delete_s]
        pruned = before - len(self._tracks)
        if pruned:
            log.debug("pruned %d dead track(s), %d remaining", pruned, len(self._tracks))

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
