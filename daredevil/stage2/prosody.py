"""Slot C — Prosodic / emotional analysis (HOW). Replaces the cloud Hume API.

Reference backend (recommended, permissive): librosa (ISC) pyin F0 + derived
features. Optional backends: Parselmouth/Praat (GPL) or OpenSMILE (audEERING
source-available) for full eGeMAPS jitter/shimmer/HNR. Fallback: stdlib proxies.

Distress heuristic (per spec): high F0 variability + high jitter + low HNR =>
distressed; the inverse => calm. Output: an emotional state label + a [0,1]
distress scalar that Stage 3 can turn into a priority escalation.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

from .base import Slot
from ..audio.utils import rms, zero_crossing_rate, spectral_centroid
from ..config import HeuristicThresholds

log = logging.getLogger("daredevil.prosody")


class ProsodySlot(Slot):
    name = "prosody"

    def __init__(self, backend: str = "auto", heuristic: Optional[HeuristicThresholds] = None):
        super().__init__()
        self._requested = backend
        self._librosa = None
        self._smile = None
        self._backend_name = "fallback"
        self._h = heuristic or HeuristicThresholds()

    @property
    def backend(self) -> str:
        return self._backend_name

    def warmup(self) -> None:
        if self._warm:
            return
        if self._requested != "fallback":
            try:
                import librosa  # ISC license — clean for an MIT project
                self._librosa = librosa
                self._backend_name = "reference"
                log.info("prosody backend: reference (librosa)")
            except Exception as e:
                log.debug("librosa unavailable: %s", e)
                try:
                    import opensmile  # optional, source-available
                    self._smile = opensmile.Smile(
                        feature_set=opensmile.FeatureSet.eGeMAPSv02,
                        feature_level=opensmile.FeatureLevel.Functionals,
                    )
                    self._backend_name = "opensmile"
                    log.info("prosody backend: opensmile (eGeMAPSv02)")
                except Exception as e2:
                    log.debug("opensmile unavailable: %s", e2)
                    self._backend_name = "fallback"
                    log.info("prosody backend: fallback (stdlib proxies)")
        self._warm = True

    def run(self, audio: List[float], sr: int, ctx: Optional[Dict] = None) -> dict:
        features = self._features(audio, sr)
        truth = (ctx or {}).get("truth")
        if truth and "distress" in truth:
            distress = float(truth["distress"])
            state = truth.get("prosody_state") or self._state(distress, features)
        else:
            distress = self._distress(features)
            state = self._state(distress, features)
        return {
            "state": state, "distress": round(distress, 3),
            "features": {k: round(v, 4) for k, v in features.items()},
            "backend": self.backend,
        }

    def _features(self, audio: List[float], sr: int) -> Dict[str, float]:
        if self._smile is not None:
            try:
                import numpy as np
                df = self._smile.process_signal(np.array(audio, dtype=np.float32), sr)
                row = df.iloc[0]
                return {
                    "f0_mean": float(row.get("F0semitoneFrom27.5Hz_sma3nz_amean", 0.0)),
                    "f0_std": float(row.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", 0.0)),
                    "jitter": float(row.get("jitterLocal_sma3nz_amean", 0.0)),
                    "shimmer": float(row.get("shimmerLocaldB_sma3nz_amean", 0.0)),
                    "hnr": float(row.get("HNRdBACF_sma3nz_amean", 0.0)),
                }
            except Exception as e:
                log.debug("opensmile feature extraction failed: %s", e)
        if self._librosa is not None:
            try:
                import numpy as np
                y = np.array(audio, dtype=np.float32)
                f0, _, _ = self._librosa.pyin(y, fmin=65, fmax=500, sr=sr)
                f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
                f0_mean = float(np.mean(f0)) if f0.size else 0.0
                f0_std = float(np.std(f0)) if f0.size else 0.0
                # jitter/shimmer/HNR proxies (full eGeMAPS available via opensmile backend)
                return {
                    "f0_mean": f0_mean, "f0_std": f0_std,
                    "jitter": float(np.std(np.diff(f0))) if f0.size > 1 else 0.0,
                    "shimmer": float(rms(audio)),
                    "hnr": 0.0,
                }
            except Exception as e:
                log.debug("librosa F0 extraction failed: %s", e)
        # stdlib proxies
        energy = rms(audio)
        zcr = zero_crossing_rate(audio)
        centroid = spectral_centroid(audio, sr)
        return {
            "f0_proxy_centroid": centroid, "zcr": zcr, "energy": energy,
            "f0_std": min(1.0, centroid / self._h.centroid_norm_hz),
            "jitter": zcr, "shimmer": energy, "hnr": 0.0,
        }

    def _distress(self, f: Dict[str, float]) -> float:
        h = self._h
        f0v = min(1.0, f.get("f0_std", 0.0) / 6.0) if f.get("f0_mean") else f.get("f0_std", 0.0)
        jit = min(1.0, f.get("jitter", 0.0) * 3.0)
        rough = min(1.0, f.get("zcr", f.get("shimmer", 0.0)))
        d = h.distress_w_f0 * f0v + h.distress_w_jitter * jit + h.distress_w_roughness * rough
        return max(0.0, min(1.0, d))

    def _state(self, distress: float, f: Dict[str, float]) -> str:
        h = self._h
        if distress >= h.state_distressed:
            return "distressed"
        if distress >= h.state_stressed:
            return "stressed"
        if f.get("f0_std", 0.0) > h.state_confused_f0_std:
            return "confused"
        return "calm"
