"""Enroll, match, store, delete voiceprints.

Enrollment confidence follows the patent's exponential saturation:
    C(t) = 1 - exp(-t / tau),  tau ~ 3s   ->  3s:0.63  10s:0.96  20s:0.999
Effective match confidence = cosine_score * C(t_enrollment).
Only non-reversible embedding vectors are persisted — never raw audio.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence

from ..audio.utils import cosine, rms


def enrollment_confidence(seconds: float, tau: float = 3.0) -> float:
    return 1.0 - math.exp(-max(0.0, seconds) / tau)


class EnrollmentManager:
    def __init__(self, config, embedding_slot, store):
        self.config = config
        self.slot = embedding_slot
        self.store = store
        self.tau = config.thresholds.enroll_tau

    # --- enrollment -------------------------------------------------------
    def _mean_embedding(self, audio: List[float], sr: int, win: float = 1.0) -> List[float]:
        n = int(win * sr)
        chunks = [audio] if len(audio) <= n else [audio[i:i + n] for i in range(0, len(audio) - n + 1, n)]
        vecs = [self.slot.run(c, sr)["vector"] for c in chunks if rms(c) > self.config.thresholds.vad]
        if not vecs:
            vecs = [self.slot.run(audio, sr)["vector"]]
        dim = len(vecs[0])
        mean = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
        norm = math.sqrt(sum(x * x for x in mean)) or 1.0
        return [x / norm for x in mean]

    def enroll(self, audio: List[float], sr: int, name: str, seconds: float) -> dict:
        self.slot.warmup()
        vec = self._mean_embedding(audio, sr)
        conf = enrollment_confidence(seconds, self.tau)
        record = {
            "name": name,
            "vector": vec,
            "dim": len(vec),
            "enrollment_confidence": round(conf, 4),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "backend": self.slot.backend,
        }
        self.store.put(name, record)
        return {"name": name, "enrollment_confidence": round(conf, 4),
                "seconds": seconds, "backend": self.slot.backend, "dim": len(vec)}

    # --- matching ---------------------------------------------------------
    def match(self, vector: Sequence[float]) -> Optional[dict]:
        best, best_score = None, -1.0
        for rec in self.store.all():
            score = cosine(vector, rec["vector"])
            if score > best_score:
                best_score, best = score, rec
        if best is None:
            return None
        return {"name": best["name"], "score": best_score,
                "enrollment_confidence": best.get("enrollment_confidence", 1.0)}

    def is_match(self, m: Optional[dict]) -> bool:
        return bool(m and m["score"] >= self.config.thresholds.match)

    def names(self) -> List[str]:
        return [r["name"] for r in self.store.all()]

    def delete(self, name: str) -> None:
        self.store.delete(name)
