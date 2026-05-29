"""Enroll, match, store, delete voiceprints.

Enrollment confidence follows the patent's exponential saturation:
    C(t) = 1 - exp(-t / tau),  tau ~ 3s   ->  3s:0.63  10s:0.96  20s:0.999

Matching uses log-likelihood ratio (LLR) accumulation — a Sequential Probability
Ratio Test. Each frame's score is calibrated via adaptive score normalization
(AS-Norm) against the enrolled cohort, then accumulated over time. Confidence
rises frame by frame like name-that-tune: frame 1 is uncertain, frame 3-5 locks.

Only non-reversible embedding vectors are persisted — never raw audio.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence

from ..audio.utils import cosine, rms


def enrollment_confidence(seconds: float, tau: float = 3.0) -> float:
    return 1.0 - math.exp(-max(0.0, seconds) / tau)


class EnrollmentManager:
    def __init__(self, config, embedding_slot, store):
        self.config = config
        self.slot = embedding_slot
        self.store = store
        self.tau = config.thresholds.enroll_tau
        # LLR accumulators per speaker — confidence builds across frames
        self._llr: Dict[str, float] = {}

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
        # Load existing record to support multi-sample enrollment (Welford update)
        existing = self.store.get(name)
        n_samples = 1
        if existing and existing.get("n_samples"):
            old_vec = existing["vector"]
            n_old = existing["n_samples"]
            n_samples = n_old + 1
            # Welford online mean update
            vec = [(old_vec[i] * n_old + vec[i]) / n_samples for i in range(len(vec))]
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vec = [x / norm for x in vec]
        record = {
            "name": name,
            "vector": vec,
            "dim": len(vec),
            "n_samples": n_samples,
            "enrollment_confidence": round(conf, 4),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "backend": self.slot.backend,
        }
        self.store.put(name, record)
        return {"name": name, "enrollment_confidence": round(conf, 4),
                "seconds": seconds, "backend": self.slot.backend, "dim": len(vec),
                "n_samples": n_samples}

    # --- matching (SPRT with AS-Norm) -------------------------------------
    def _score_calibrated(self, vector: Sequence[float], target_vec: Sequence[float],
                          cohort_vecs: List[Sequence[float]]) -> float:
        """AS-Norm: normalize raw cosine against the cohort distribution."""
        raw = cosine(vector, target_vec)
        if not cohort_vecs:
            return raw
        cohort_scores = [cosine(vector, c) for c in cohort_vecs]
        mu = sum(cohort_scores) / len(cohort_scores)
        var = sum((s - mu) ** 2 for s in cohort_scores) / len(cohort_scores)
        std = math.sqrt(var) if var > 0 else 1.0
        return (raw - mu) / std

    def match(self, vector: Sequence[float], energy: float = 1.0) -> Optional[dict]:
        """Score against all enrolled speakers using calibrated LLR accumulation.

        Each frame accumulates evidence. Confidence rises over time — like
        name-that-tune. A strong single frame can still match immediately
        (raw cosine > threshold), but weaker frames build up sequentially.
        """
        records = self.store.all()
        if not records:
            return None

        # Build cohort from all enrolled speakers (for AS-Norm)
        cohort_vecs = [r["vector"] for r in records]

        best_name, best_conf, best_enroll_conf = None, -999.0, 1.0
        for rec in records:
            name = rec["name"]
            raw = cosine(vector, rec["vector"])

            # If raw cosine is strong enough on its own, match immediately
            if raw >= self.config.thresholds.match:
                if raw > best_conf:
                    best_conf = raw
                    best_name = name
                    best_enroll_conf = rec.get("enrollment_confidence", 1.0)
                # Also boost the accumulator for future frames
                self._llr[name] = self._llr.get(name, 0.0) + raw * 0.5
                continue

            # Calibrated score for weaker signals
            others = [v for v in cohort_vecs if v is not rec["vector"]]
            frame_score = self._score_calibrated(vector, rec["vector"], others)

            # Quality gate — weight by energy
            quality = min(1.0, energy / 0.005) if energy > 0 else 0.1
            weighted_score = frame_score * quality

            # LLR accumulation with decay (SPRT)
            prev = self._llr.get(name, 0.0)
            self._llr[name] = prev * 0.92 + weighted_score * 0.35

            # Confidence: sigmoid of accumulated LLR mapped to [0, 1]
            accumulated = self._llr[name]
            confidence = 1.0 / (1.0 + math.exp(-accumulated))

            if confidence > best_conf:
                best_conf = confidence
                best_name = name
                best_enroll_conf = rec.get("enrollment_confidence", 1.0)

        if best_name is None:
            return None
        return {"name": best_name, "score": best_conf,
                "enrollment_confidence": best_enroll_conf}

    def is_match(self, m: Optional[dict]) -> bool:
        return bool(m and m["score"] >= self.config.thresholds.match)

    def names(self) -> List[str]:
        return [r["name"] for r in self.store.all()]

    def delete(self, name: str) -> None:
        self.store.delete(name)
        self._llr.pop(name, None)

    def reset_accumulators(self) -> None:
        """Reset all LLR accumulators (e.g. on scene change)."""
        self._llr.clear()
