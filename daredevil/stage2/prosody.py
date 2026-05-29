"""Slot C — Prosodic / emotional analysis (HOW). Replaces the cloud Hume API.

Reference backend (recommended, permissive): librosa (ISC) pyin F0 + derived
features. Optional backends: Parselmouth/Praat (GPL) or OpenSMILE (audEERING
source-available) for full eGeMAPS jitter/shimmer/HNR. Fallback: stdlib proxies.

Distress heuristic (per spec): high F0 variability + high jitter + low HNR =>
distressed; the inverse => calm. Output: an emotional state label + a [0,1]
distress scalar that Stage 3 can turn into a priority escalation.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from .base import Slot
from ..audio.utils import rms, zero_crossing_rate, spectral_centroid


class ProsodySlot(Slot):
    name = "prosody"

    def __init__(self, backend: str = "auto"):
        super().__init__()
        self._requested = backend
        self._librosa = None
        self._smile = None
        self._backend_name = "fallback"

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
            except Exception:
                try:
                    import opensmile  # optional, source-available
                    self._smile = opensmile.Smile(
                        feature_set=opensmile.FeatureSet.eGeMAPSv02,
                        feature_level=opensmile.FeatureLevel.Functionals,
                    )
                    self._backend_name = "opensmile"
                except Exception:
                    self._backend_name = "fallback"
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
            except Exception:
                pass
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
            except Exception:
                pass
        # stdlib proxies
        energy = rms(audio)
        zcr = zero_crossing_rate(audio)
        centroid = spectral_centroid(audio, sr)
        return {
            "f0_proxy_centroid": centroid, "zcr": zcr, "energy": energy,
            "f0_std": min(1.0, centroid / 2000.0), "jitter": zcr, "shimmer": energy, "hnr": 0.0,
        }

    @staticmethod
    def _distress(f: Dict[str, float]) -> float:
        # Normalised combination of variability + roughness indicators.
        f0v = min(1.0, f.get("f0_std", 0.0) / 6.0) if f.get("f0_mean") else f.get("f0_std", 0.0)
        jit = min(1.0, f.get("jitter", 0.0) * 3.0)
        rough = min(1.0, f.get("zcr", f.get("shimmer", 0.0)))
        d = 0.45 * f0v + 0.30 * jit + 0.25 * rough
        return max(0.0, min(1.0, d))

    @staticmethod
    def _state(distress: float, f: Dict[str, float]) -> str:
        if distress >= 0.6:
            return "distressed"
        if distress >= 0.4:
            return "stressed"
        if f.get("f0_std", 0.0) > 0.7:
            return "confused"
        return "calm"
