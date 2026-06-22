"""Stage 3 — Attention Router. Fuses everything into the awareness map.

Priority (patent Eq. 2):
    P = w_id*S_identity + w_event*S_event + w_prosody*S_prosody + w_temporal*S_temporal
Overrides:
    - SAFETY_CRITICAL: a safety-critical event above T_safety pins priority to 1.0.
    - DISTRESS: an enrolled speaker in distress above T_distress is escalated,
      regardless of event class.
The returned dict is the product: structured context an LLM can consume directly.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..config import Config

log = logging.getLogger("daredevil.router")


class AttentionRouter:
    def __init__(self, config: Config):
        self.config = config
        self.weights = config.weights.normalized()
        self.th = config.thresholds

    def build(self, records: List[dict], timestamp: str, timing: dict,
              array, backend: str) -> dict:
        sources = [self._score(r) for r in records]
        sources.sort(key=lambda s: s["priority"], reverse=True)
        return {
            "timestamp": timestamp,
            "backend": backend,
            "array": {
                "name": array.name,
                "n_mics": array.n_mics,
                "spatial": array.spatial_capable,
            },
            "routed_to_llm": [s["id"] for s in sources if s["attention"] == "surface"],
            "sources": sources,
            "timing": timing,
            "privacy": {
                "cloud_used": False,
                "raw_audio_stored": False,
                "embeddings": "non-reversible",
            },
        }

    def _score(self, r: dict) -> dict:
        ev = r["event"]
        pr = r["prosody"]
        ident = r.get("identity")
        enrolled = bool(ident and ident.get("name"))

        s_identity = (ident["score"] * ident.get("enrollment_confidence", 1.0)) if enrolled else 0.0
        s_event = ev["confidence"] if ev.get("safety_critical") else 0.0
        s_prosody = float(pr.get("distress", 0.0))
        s_temporal = float(r.get("recency", 1.0))

        w = self.weights
        raw = (w.identity * s_identity + w.event * s_event
               + w.prosody * s_prosody + w.temporal * s_temporal)

        wake = r.get("wake")
        addressed = bool(wake and wake.get("detected"))

        # Two ways a source earns attention — the way humans do it: by the SOUND of
        # the voice (a safety-critical event, or an enrolled speaker in distress)
        # and, when that isn't enough, by NAME (it said the wake word). Being named
        # is a real bid for attention, so it escalates too — just under true distress.
        priority, override = raw, None
        if ev.get("safety_critical") and ev["confidence"] >= self.th.safety:
            priority, override = 1.0, "SAFETY_CRITICAL"
            log.debug("SAFETY_CRITICAL override: %s (class=%s, conf=%.2f)",
                       ident.get("name") if ident else "unknown", ev["class"], ev["confidence"])
        elif enrolled and s_prosody >= self.th.distress:
            priority, override = max(raw, 0.85), "DISTRESS"
            log.debug("DISTRESS override: %s (distress=%.2f)", ident["name"], s_prosody)
        elif addressed:
            priority, override = max(raw, 0.80), "ADDRESSED"
        priority = max(0.0, min(1.0, priority))

        # The attention gate — the whole point. A struct is built for EVERY source,
        # but only some are surfaced to the conversational LLM: a safety/distress
        # override, the enrolled owner actually speaking (directed), or anything
        # above the surface threshold. Everything else is "ambient" — heard,
        # tracked, and deliberately NOT sent into the conversation.
        directed = enrolled and ev.get("class") == "speech"
        surfaced = bool(override) or directed or (priority >= self.th.surface)

        # Why are we paying attention? Hand the reason to the LLM/robot so it can act
        # naturally — including using a caller's name back when its own voice won't
        # carry. This mirrors how the attention was earned in the first place.
        reason = None
        if override == "SAFETY_CRITICAL":
            reason = "safety"
        elif override == "DISTRESS":
            reason = "voice"          # grabbed us by the *sound* of the voice
        elif override == "ADDRESSED":
            reason = "name"           # grabbed us by saying our name
        elif directed:
            reason = "owner-speaking"
        elif surfaced:
            reason = "salient"

        event_out = {"class": ev["class"], "confidence": round(ev["confidence"], 3)}
        if ev.get("safety_critical"):
            event_out["safety_critical"] = True

        src = {
            "id": ident["name"] if enrolled else r.get("unknown_id", "UNKNOWN-000"),
            "type": "enrolled" if enrolled else "unknown",
            "attention": "surface" if surfaced else "ambient",
            "event": event_out,
            "prosody": {"state": pr["state"], "distress": round(s_prosody, 3)},
            "priority": round(priority, 3),
        }
        if enrolled:
            src["identity"] = {
                "confidence": round(s_identity, 3),
                "match_score": round(ident["score"], 3),
                "enrollment_confidence": round(ident.get("enrollment_confidence", 1.0), 3),
            }
        pos = r.get("position")
        if pos and pos.get("azimuth") is not None:
            src["position"] = {"azimuth": pos["azimuth"], "elevation": pos.get("elevation", 0.0)}
        if override:
            src["priority_override"] = override
        if r.get("track_status"):
            src["track_status"] = r["track_status"]
        if r.get("identifying"):
            src["identifying"] = r["identifying"]
        if addressed:
            src["addressed"] = True
            src["wake"] = {"phrase": wake.get("phrase"), "score": wake.get("score")}
        if surfaced and reason:
            src["attention_reason"] = reason
        return src


def llm_payload(amap: dict) -> List[dict]:
    """The sources the attention gate actually surfaces to the conversational LLM.

    A struct is built for every detected source (full internal awareness); only
    these cross the gate into the conversation. This is the point of the system.
    """
    return [s for s in amap.get("sources", []) if s.get("attention") == "surface"]
