"""Pure-stdlib DSP helpers.

These are intentionally dependency-free so the core pipeline runs anywhere. When
numpy is available we use it (faster); otherwise we fall back to plain Python.
None of this is meant to beat librosa — it's the floor that guarantees the
awareness map is always computable.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

try:  # optional acceleration
    import numpy as _np
except Exception:  # pragma: no cover - numpy is optional
    _np = None


def have_numpy() -> bool:
    return _np is not None


def to_mono(channels: Sequence[Sequence[float]]) -> List[float]:
    """Average a list of channels into a mono signal."""
    if not channels:
        return []
    if len(channels) == 1:
        return list(channels[0])
    n = min(len(c) for c in channels)
    if _np is not None:
        arr = _np.array([c[:n] for c in channels], dtype=float)
        return arr.mean(axis=0).tolist()
    out = [0.0] * n
    k = len(channels)
    for c in channels:
        for i in range(n):
            out[i] += c[i]
    return [v / k for v in out]


def rms(signal: Sequence[float]) -> float:
    if not signal:
        return 0.0
    if _np is not None:
        a = _np.asarray(signal, dtype=float)
        return float(_np.sqrt(_np.mean(a * a)))
    return math.sqrt(sum(v * v for v in signal) / len(signal))


def zero_crossing_rate(signal: Sequence[float]) -> float:
    if len(signal) < 2:
        return 0.0
    crossings = 0
    prev = signal[0]
    for v in signal[1:]:
        if (v >= 0) != (prev >= 0):
            crossings += 1
        prev = v
    return crossings / (len(signal) - 1)


def resample(signal: Sequence[float], sr_in: int, sr_out: int) -> List[float]:
    """Linear-interpolation resample. Good enough to feed 16k models from 44.1/48k."""
    if sr_in == sr_out or not signal:
        return list(signal)
    ratio = sr_out / sr_in
    n_out = max(1, int(round(len(signal) * ratio)))
    if _np is not None:
        x_old = _np.linspace(0.0, 1.0, num=len(signal), endpoint=False)
        x_new = _np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return _np.interp(x_new, x_old, _np.asarray(signal, dtype=float)).tolist()
    out = []
    last = len(signal) - 1
    for i in range(n_out):
        pos = i / ratio
        lo = int(math.floor(pos))
        hi = min(lo + 1, last)
        frac = pos - lo
        out.append(signal[lo] * (1 - frac) + signal[hi] * frac)
    return out


def is_speech(signal: Sequence[float], threshold: float) -> bool:
    """Trivial energy-gate VAD. Real backends can swap in Silero VAD (MIT)."""
    return rms(signal) >= threshold


def spectral_centroid(signal: Sequence[float], sr: int) -> float:
    """Rough spectral centroid (Hz) via a small DFT. Used by heuristic slots."""
    mags, freqs = _dft_mag(signal, sr, n_bins=64)
    total = sum(mags)
    if total <= 0:
        return 0.0
    return sum(f * m for f, m in zip(freqs, mags)) / total


def is_speech_quality(audio: Sequence[float], sr: int,
                      min_energy: float = 0.004, max_zcr: float = 3000.0) -> bool:
    """Energy + ZCR gate for speech quality. Modeled on WebRTC VAD mode 0.

    Production systems (Silero, pyannote, SpeechBrain) all gate on energy
    with hysteresis. We use energy threshold + ZCR to reject clicks/noise.
    No spectral tilt — real speech through a laptop mic in a noisy room
    doesn't have clean spectral characteristics.

    The energy floor mirrors the model's VAD threshold (Config.Thresholds.vad):
    a frame only has to clear "someone is speaking" to be *eligible* — the SPRT
    then down-weights quiet frames via EnrollmentManager._frame_quality. The
    floor and the ZCR ceiling are passed in from config, not hardcoded, so one
    tunable governs the energy decision. (A previously hardcoded 0.05 RMS floor
    silently rejected normal laptop-mic speech, so WHO never fired on live mic.)

    Non-speech frames are non-evidence — they don't vote in the SPRT.
    """
    n = len(audio)
    if n < 100:
        return False
    energy = (sum(x * x for x in audio) / n) ** 0.5
    if energy < min_energy:
        return False
    zcr = sum(1 for i in range(1, n) if audio[i - 1] * audio[i] < 0)
    zcr_rate = zcr * sr / n
    if zcr_rate > max_zcr:
        return False
    return True


def fingerprint(signal: Sequence[float], sr: int, dim: int = 192) -> List[float]:
    """A cheap, deterministic spectral fingerprint, L2-normalised to `dim`.

    This is the *fallback* speaker embedding: it is NOT ECAPA and makes no claim
    to discriminate real voices. Its only job is to let the enroll -> identify
    demo behave sensibly with zero dependencies, by being stable for the same
    input and different across different sources. On a real install, the ECAPA
    backend replaces this entirely.
    """
    mags, _ = _dft_mag(signal, sr, n_bins=dim)
    norm = math.sqrt(sum(m * m for m in mags))
    if norm <= 0:
        return [0.0] * dim
    return [m / norm for m in mags]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1] (patent Eq. 1)."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if _np is not None:
        av = _np.asarray(a[:n], dtype=float)
        bv = _np.asarray(b[:n], dtype=float)
        da = float(_np.linalg.norm(av))
        db = float(_np.linalg.norm(bv))
        if da == 0 or db == 0:
            return 0.0
        return float(av.dot(bv) / (da * db))
    dot = sum(a[i] * b[i] for i in range(n))
    da = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    db = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if da == 0 or db == 0:
        return 0.0
    return dot / (da * db)


# --- internal -------------------------------------------------------------

_ANALYSIS_SR = 4000    # fixed analysis rate -> fingerprint is clip-length & rate invariant
_ANALYSIS_WIN = 1.0    # seconds analyzed, anchored to the start of the clip


def _dft_mag(signal: Sequence[float], sr: int, n_bins: int):
    """Magnitude spectrum at `n_bins` log-spaced physical freqs (60Hz .. ~2kHz).

    Computed over a fixed-duration window resampled to a fixed analysis rate, so
    the result reflects the source's *spectral content*, not the clip length —
    which is what lets a 3s enrollment match a 1s or 1.5s identification.
    """
    if not signal:
        return [0.0] * n_bins, [0.0] * n_bins
    win_n = int(_ANALYSIS_WIN * sr)
    win = list(signal[:win_n]) if len(signal) > win_n else list(signal)
    xa = resample(win, sr, _ANALYSIS_SR)
    n = int(_ANALYSIS_WIN * _ANALYSIS_SR)
    if len(xa) < n:
        xa = xa + [0.0] * (n - len(xa))
    else:
        xa = xa[:n]
    f_lo, f_hi = 60.0, _ANALYSIS_SR / 2.0
    freqs = [f_lo * (f_hi / f_lo) ** (k / max(1, n_bins - 1)) for k in range(n_bins)]
    if _np is not None:
        xv = _np.asarray(xa, dtype=float)
        idx = _np.arange(n)
        mags = []
        for f in freqs:
            ang = -2.0 * math.pi * f * idx / _ANALYSIS_SR
            re = float(xv.dot(_np.cos(ang)))
            im = float(xv.dot(_np.sin(ang)))
            mags.append(math.hypot(re, im))
        return mags, freqs
    mags = []
    for f in freqs:
        w = -2.0 * math.pi * f / _ANALYSIS_SR
        re = im = 0.0
        for i in range(n):
            ang = w * i
            re += xa[i] * math.cos(ang)
            im += xa[i] * math.sin(ang)
        mags.append(math.hypot(re, im))
    return mags, freqs
