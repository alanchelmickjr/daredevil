"""Slot B — Acoustic event classification (WHAT).

Reference backend: PANNs CNN14 (527 AudioSet classes). Edge alternative noted in
docs: YAMNet (Apache-2.0, ~4MB, ONNX). Fallback: a light spectral heuristic, plus
(in the synthetic demo) the scene's ground-truth class so the router/override path
is exercised honestly. Safety-critical classes (baby cry, alarm, siren, glass,
gunshot...) flip a flag that Stage 3 turns into a priority override.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .base import Slot
from ..config import is_safety_critical, HeuristicThresholds
from ..audio.utils import spectral_centroid, zero_crossing_rate, rms

log = logging.getLogger("daredevil.events")


class EventsSlot(Slot):
    name = "events"

    def __init__(self, backend: str = "auto", heuristic: Optional[HeuristicThresholds] = None):
        super().__init__()
        self._requested = backend
        self._model = None
        self._labels = None
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
                from panns_inference import AudioTagging
                from panns_inference.config import labels

                self._model = AudioTagging(checkpoint_path=None, device="cpu")
                self._labels = labels
                self._backend_name = "reference"
                log.info("events backend: reference (PANNs CNN14)")
            except Exception as e:
                log.debug("PANNs unavailable: %s", e)
                self._model = None
                self._backend_name = "fallback"
                log.info("events backend: fallback (spectral heuristic)")
        self._warm = True

    def run(self, audio: List[float], sr: int, ctx: Optional[Dict] = None) -> dict:
        # Synthetic scene: use ground-truth labels so the demo exercises the full
        # router/override path honestly. Real audio skips this and classifies.
        truth = (ctx or {}).get("truth")
        if truth and truth.get("class"):
            klass = truth["class"]
            return {
                "class": klass, "confidence": 0.95,
                "safety_critical": is_safety_critical(klass),
                "topk": [{"class": klass, "confidence": 0.95}],
                "backend": self._backend_name,
            }

        if self._model is not None:
            try:
                import numpy as np
                import librosa

                clip = np.array(audio, dtype=np.float32)
                if sr != 32000:
                    clip = librosa.resample(clip, orig_sr=sr, target_sr=32000)
                clip = clip[None, :]
                clipwise, _ = self._model.inference(clip)
                probs = clipwise[0]
                order = list(probs.argsort()[::-1][:5])
                topk = [{"class": str(self._labels[i]), "confidence": float(probs[i])} for i in order]
                top = topk[0]
                return {
                    "class": top["class"], "confidence": top["confidence"],
                    "safety_critical": is_safety_critical(top["class"]),
                    "topk": topk, "backend": "reference",
                }
            except Exception as e:
                log.debug("PANNs inference failed, falling through: %s", e)

        # Fallback heuristic for real audio when no model is loaded.
        klass, conf = self._heuristic(audio, sr)
        return {
            "class": klass, "confidence": round(conf, 3),
            "safety_critical": is_safety_critical(klass),
            "topk": [{"class": klass, "confidence": round(conf, 3)}],
            "backend": "fallback",
        }

    def _heuristic(self, audio: List[float], sr: int):
        h = self._h
        energy = rms(audio)
        if energy < h.silence_energy:
            return "silence", 0.5
        centroid = spectral_centroid(audio, sr)
        zcr = zero_crossing_rate(audio)
        if centroid > h.baby_cry_centroid_hz and zcr > h.baby_cry_zcr:
            return "baby_cry", 0.6
        if centroid > h.alarm_centroid_hz:
            return "alarm", 0.55
        if centroid < h.speech_centroid_hz and zcr < h.speech_zcr:
            return "speech", 0.7
        return "music", 0.5
