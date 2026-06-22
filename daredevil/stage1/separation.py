"""Source separation — splits a mixed audio stream into individual sources.

Reference backend: Asteroid ConvTasNet (MIT, pretrained on WHAM!, 2 sources).
Lightweight alternative: SuDORMRFNet (340K params, real-time on any device).
Fallback: pass-through (single source = the mix).

This is the key piece that lets the pipeline discriminate between simultaneous
sounds on a single mic — speech vs music vs keyboard vs baby cry — BEFORE the
slots analyze each one independently.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

from ..audio.utils import rms, resample

log = logging.getLogger("daredevil.separation")


class Separator:
    def __init__(self, n_sources: int = 2, backend: str = "auto"):
        self._requested = backend
        self._model = None
        self._backend_name = "fallback"
        self._n_sources = n_sources
        self._sample_rate = 8000  # ConvTasNet trained at 8kHz
        self._warm = False

    @property
    def backend(self) -> str:
        return self._backend_name

    def warmup(self) -> None:
        if self._warm:
            return
        if self._requested != "fallback":
            try:
                import torch  # noqa: F401
                from asteroid.models import ConvTasNet

                self._model = ConvTasNet.from_pretrained(
                    "mpariente/ConvTasNet_WHAM!_sepclean"
                )
                self._model.eval()
                self._n_sources = self._model.masker.n_src
                self._backend_name = "reference"
                log.info("separation backend: reference (ConvTasNet)")
                # Warm up JIT
                with torch.no_grad():
                    self._model(torch.randn(1, self._sample_rate))
            except Exception as e:
                log.debug("ConvTasNet unavailable: %s", e)
                self._model = None
                self._backend_name = "fallback"
                log.info("separation backend: fallback (pass-through)")
        self._warm = True

    def separate(self, audio: List[float], sr: int,
                 min_energy: float = 0.001) -> List[dict]:
        """Separate mixed audio into individual source streams.

        Returns a list of dicts: [{"audio": [...], "sr": int, "energy": float}]
        Each entry is one separated source with enough energy to be meaningful.
        """
        if self._model is not None:
            try:
                import torch

                # Resample to model's expected rate
                resampled = resample(audio, sr, self._sample_rate)
                x = torch.tensor(resampled, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    separated = self._model(x)  # [1, n_sources, time]

                # ConvTasNet amplifies wildly — normalize total output energy
                # to match the input so downstream slots and tracker see sane values.
                input_energy = rms(resampled) or 1e-6
                raw_streams = []
                for i in range(separated.shape[1]):
                    stream = separated[0, i].tolist()
                    stream_at_sr = resample(stream, self._sample_rate, sr)
                    raw_streams.append(stream_at_sr)
                total_energy = sum(rms(s) for s in raw_streams) or 1e-6
                scale = input_energy / total_energy

                sources = []
                for stream_at_sr in raw_streams:
                    stream_at_sr = [v * scale for v in stream_at_sr]
                    energy = rms(stream_at_sr)
                    if energy > min_energy:
                        sources.append({
                            "audio": stream_at_sr,
                            "sr": sr,
                            "energy": energy,
                        })

                if sources:
                    return sources
            except Exception as e:
                log.debug("ConvTasNet inference failed, falling through: %s", e)

        # Fallback: single source pass-through
        return [{"audio": list(audio), "sr": sr, "energy": rms(audio)}]
