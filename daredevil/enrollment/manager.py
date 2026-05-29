"""Enroll, match, store, delete voiceprints.

Enrollment confidence follows the patent's exponential saturation:
    C(t) = 1 - exp(-t / tau),  tau ~ 3s   ->  3s:0.63  10s:0.96  20s:0.999

Identity matching is a Sequential Probability Ratio Test (Wald) — the same
detect-by-accumulating-evidence scheme passive sonar uses to call a contact.
For the cosine similarity ``s`` between an observed embedding and an enrolled
voiceprint, each frame contributes a log-likelihood ratio comparing:

    H1 (this is speaker k):   s ~ N(target_mean, target_std)
    H0 (background/impostor): s ~ N(impostor_mean, impostor_std)

When two or more speakers are enrolled, the *others* form an adaptive cohort
(AS-Norm) that estimates H0 directly; with a single enrolled speaker the
configured background model stands in, so identity still works. The per-frame
ratios accumulate **per track** (identity is a property of a tracked contact, not
of one frame) until the running total crosses the upper Wald bound
A = log((1-beta)/alpha) — declare the identity — or the lower bound
B = log(beta/(1-alpha)) — reset, it isn't this speaker. The cumulative LLR is a
log-odds, so confidence is sigmoid(LLR): a genuine posterior probability, not a
hand-tuned curve. Weak (low-energy) frames are down-weighted; one very strong
frame still matches outright.

Only non-reversible embedding vectors are persisted — never raw audio.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..audio.utils import cosine, rms

_LOG_2PI = math.log(2.0 * math.pi)


def enrollment_confidence(seconds: float, tau: float = 3.0) -> float:
    return 1.0 - math.exp(-max(0.0, seconds) / tau)


def _log_gaussian(x: float, mu: float, sigma: float) -> float:
    """Log of the Normal pdf N(mu, sigma) evaluated at x."""
    sigma = max(sigma, 1e-6)
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma) - 0.5 * _LOG_2PI


class EnrollmentManager:
    def __init__(self, config, embedding_slot, store):
        self.config = config
        self.slot = embedding_slot
        self.store = store
        self.tau = config.thresholds.enroll_tau
        self.idm = config.identity
        # Wald decision bounds derived from the configured error rates.
        self._A = math.log((1.0 - self.idm.beta) / self.idm.alpha)   # accept identity
        self._B = math.log(self.idm.beta / (1.0 - self.idm.alpha))   # reject identity
        # Cumulative SPRT log-likelihood ratio (a log-odds) per (track, speaker).
        self._llr: Dict[Tuple[str, str], float] = {}
        # CFAR-style running background model — estimated live from ambient frames so
        # the false-alarm rate stays constant as the room changes. Seeded from the
        # (human-calibrated or default) impostor model.
        self._bg_mean = self.idm.impostor_mean
        self._bg_var = self.idm.impostor_std ** 2

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

    # --- matching (Wald SPRT, accumulated per track) ----------------------
    def _background_stats(self, vector: Sequence[float],
                          others: List[Sequence[float]]) -> Tuple[float, float]:
        """H0 parameters: an adaptive AS-Norm cohort if we have one, else the model.

        With >=2 enrolled speakers the *other* voiceprints give the impostor score
        distribution directly. With a single speaker there is no cohort, so the
        configured background model is used — that is what lets one enrolled speaker
        be matched at all.
        """
        if len(others) >= 2:
            scores = [cosine(vector, o) for o in others]
            mu = sum(scores) / len(scores)
            var = sum((x - mu) ** 2 for x in scores) / len(scores)
            sd = math.sqrt(var) if var > 1e-9 else self.idm.impostor_std
            return mu, max(sd, 1e-3)
        if self.idm.adapt_background:
            return self._bg_mean, max(math.sqrt(self._bg_var), 1e-3)
        return self.idm.impostor_mean, self.idm.impostor_std

    def _update_background(self, s: float) -> None:
        """CFAR update from an ambient frame, with a guard band so a present target
        (a high cosine) is excluded — otherwise the target would mask itself into
        the noise estimate, exactly the failure CFAR guard cells prevent."""
        guard = self.idm.impostor_mean + self.idm.bg_guard_sigmas * self.idm.impostor_std
        if s > guard:
            return  # looks like (or near) a target — don't let it pollute the noise floor
        r = self.idm.bg_adapt_rate
        self._bg_mean = (1.0 - r) * self._bg_mean + r * s
        self._bg_var = (1.0 - r) * self._bg_var + r * (s - self._bg_mean) ** 2

    def _maybe_refine(self, rec: dict, vector: Sequence[float], acc: float,
                      raw: float, quality: float) -> None:
        """Guarded online enrollment: nudge the voiceprint toward a frame only when
        the match is overwhelming (well past the Wald bound, very high cosine, loud).
        Off by default — auto-refining identity is poisonable, so the bar is high."""
        if not (acc >= self._A + self.idm.target_adapt_margin
                and raw >= self.idm.immediate_cosine and quality >= 0.9):
            return
        r = self.idm.target_adapt_rate
        old = rec["vector"]
        m = min(len(old), len(vector))
        merged = [(1.0 - r) * old[i] + r * vector[i] for i in range(m)]
        norm = math.sqrt(sum(x * x for x in merged)) or 1.0
        rec["vector"] = [x / norm for x in merged]
        rec["n_samples"] = rec.get("n_samples", 1) + 1
        self.store.put(rec["name"], rec)

    def _frame_quality(self, energy: float) -> float:
        """Evidential weight in [0, 1]: silence carries none, loud-enough carries full."""
        vad = self.config.thresholds.vad
        full = self.idm.quality_full_energy
        if energy <= vad:
            return 0.0
        if full <= vad:
            return 1.0
        return max(0.0, min(1.0, (energy - vad) / (full - vad)))

    def match(self, vector: Sequence[float], energy: float = 1.0,
              key: str = "default") -> Optional[dict]:
        """Score against all enrolled speakers, accumulating an SPRT per (track, speaker).

        ``key`` identifies the tracked contact this observation belongs to, so two
        simultaneous sources don't pool their evidence. Returns the best candidate
        (or None if nobody is enrolled); call ``is_match`` to apply the Wald bound.
        """
        records = self.store.all()
        if not records:
            return None

        cohort = [r["vector"] for r in records]
        quality = self._frame_quality(energy)
        idm = self.idm

        best = None
        raws = []
        for rec in records:
            name = rec["name"]
            s = cosine(vector, rec["vector"])
            raws.append(s)
            others = [v for v in cohort if v is not rec["vector"]]
            mu0, sd0 = self._background_stats(vector, others)

            # Per-frame log-likelihood ratio: log p(s|target) - log p(s|background).
            frame_llr = (_log_gaussian(s, idm.target_mean, idm.target_std)
                         - _log_gaussian(s, mu0, sd0))

            kk = (key, name)
            acc = (1.0 - idm.leak) * self._llr.get(kk, 0.0) + quality * frame_llr
            if acc <= self._B:           # lower bound: definitively not this speaker -> reset
                acc = 0.0
            self._llr[kk] = acc

            posterior = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, acc))))
            immediate = s >= idm.immediate_cosine          # one very clean frame is enough
            decided = immediate or acc >= self._A

            if idm.adapt_target and quality > 0.0:
                self._maybe_refine(rec, vector, acc, s, quality)

            cand = {
                "name": name, "raw": round(s, 4), "llr": acc,
                "score": 1.0 if immediate else posterior,
                "decided": decided,
                "enrollment_confidence": rec.get("enrollment_confidence", 1.0),
            }
            if best is None or cand["llr"] > best["llr"]:
                best = cand

        # CFAR: refresh the live noise floor from the most background-like score.
        if idm.adapt_background and raws and quality > 0.0:
            self._update_background(min(raws))
        return best

    def is_match(self, m: Optional[dict]) -> bool:
        return bool(m and (m.get("decided") or m.get("llr", 0.0) >= self._A))

    # --- accumulator bookkeeping -----------------------------------------
    def retain(self, live_keys: Iterable[str]) -> None:
        """Drop SPRT accumulators for tracks that no longer exist."""
        live = set(live_keys)
        self._llr = {(k, n): v for (k, n), v in self._llr.items() if k in live}

    def forget_track(self, key: str) -> None:
        for kk in [kk for kk in self._llr if kk[0] == key]:
            self._llr.pop(kk, None)

    def top_llr(self) -> str:
        """Compact readout of the strongest accumulator (for logging)."""
        if not self._llr:
            return "none"
        (key, name), v = max(self._llr.items(), key=lambda kv: kv[1])
        return f"{name}@{key}={v:.2f}"

    def names(self) -> List[str]:
        return [r["name"] for r in self.store.all()]

    def delete(self, name: str) -> None:
        self.store.delete(name)
        for kk in [kk for kk in self._llr if kk[1] == name]:
            self._llr.pop(kk, None)

    def reset_accumulators(self) -> None:
        """Reset all SPRT accumulators (e.g. on scene change)."""
        self._llr.clear()
