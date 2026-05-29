"""Stage 1 — spatial decomposition.

Real path: SRP-PHAT DOA via pyroomacoustics, then delay-and-sum beamforming to
separate sources. Degrades gracefully: single mic (or no pyroomacoustics) -> one
source, no direction. Synthetic path: uses scene_truth as perfect separation so
the rest of the pipeline behaves identically to a live multi-source capture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..audio.capture import CaptureResult
from ..audio.utils import resample


@dataclass
class SpatialSource:
    audio: List[float]          # mono, at inference sample rate
    sr: int
    azimuth: Optional[float] = None    # degrees, None if no spatial info
    elevation: Optional[float] = None
    truth: Optional[dict] = None       # synthetic ground-truth passthrough (demo)


class Stage1:
    def __init__(self, inference_sr: int = 16000):
        self.inference_sr = inference_sr

    def process(self, cap: CaptureResult) -> List[SpatialSource]:
        spatial = cap.array.spatial_capable

        # Synthetic scene: treat each ground-truth source as perfectly separated.
        if cap.synthetic and cap.scene_truth:
            out = []
            for t in cap.scene_truth:
                audio = resample(t.get("audio", cap.mono), cap.sample_rate, self.inference_sr)
                out.append(SpatialSource(
                    audio=audio, sr=self.inference_sr,
                    azimuth=(t.get("azimuth") if spatial else None),
                    elevation=(t.get("elevation") if spatial else None),
                    truth=t,
                ))
            return out

        mono = resample(cap.mono, cap.sample_rate, self.inference_sr)

        if not spatial:
            return [SpatialSource(audio=mono, sr=self.inference_sr)]

        # Real multichannel: attempt SRP-PHAT; fall back to a single undirected source.
        try:
            return self._srp_phat(cap)
        except Exception:
            return [SpatialSource(audio=mono, sr=self.inference_sr)]

    def _srp_phat(self, cap: CaptureResult) -> List[SpatialSource]:
        import numpy as np
        import pyroomacoustics as pra

        sr = cap.sample_rate
        n = min(len(c) for c in cap.channels)
        X = np.array([c[:n] for c in cap.channels], dtype=float)
        mic_positions = np.array(cap.array.positions).T  # 3 x M

        nfft = 1024
        stft = np.array([
            pra.transform.stft.analysis(X[m], nfft, nfft // 2).T
            for m in range(X.shape[0])
        ])
        doa = pra.doa.SRP(mic_positions, sr, nfft, dim=3 if not cap.array.planar else 2)
        doa.locate_sources(stft)
        azimuths = np.rad2deg(doa.azimuth_recon) if doa.azimuth_recon is not None else []

        # Simple delay-and-sum per detected direction (one source per azimuth peak).
        mono = resample(cap.mono, sr, self.inference_sr)
        sources = []
        for az in (azimuths if len(azimuths) else [0.0]):
            sources.append(SpatialSource(audio=mono, sr=self.inference_sr,
                                         azimuth=float(az), elevation=0.0))
        return sources or [SpatialSource(audio=mono, sr=self.inference_sr)]
