"""NMF spectral-basis tracking — "name that tune in 3 notes".

Frame-to-frame ECAPA cosine is unstable: ECAPA encodes *speaker identity* over
whole utterances, not *spectral continuity* across 1-second windows, so two
consecutive frames of the same source can score a low cosine and the tracker
loses the contact. This module gives the tracker a different signal — a
frame-level feature that says "this slice is the same ongoing sound" even as the
content changes.

Non-negative Matrix Factorization decomposes a frame's magnitude spectrum into
activations over spectral bases (v ≈ W·h): the activation vector says *which
spectral components are firing now*. Same physical source → nearly identical
activations frame to frame — exactly the continuity the tracker needs.

WHO (ECAPA, identity) and "same ongoing sound?" (NMF activations) are different
questions; this answers the second. See docs/NMF_TRACKING_DESIGN.md.

v1 uses a **fixed** overlapping-triangular basis (a supervised/universal basis,
per the design doc): deterministic, dimensionally stable, and frame-stable.
`learn_basis` is provided for the next step — learning per-source bases at
enrollment / online — but the tracker feature stays a fixed dimension so a
contact's stored feature never silently changes meaning mid-track.

Graceful degradation (non-negotiable #1): the decomposition uses numpy when
present; with no numpy the feature falls back to grouping the spectrum into the
same number of bands — itself a valid, frame-stable, same-dimensioned feature —
so the core still runs on the standard library alone.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from ..audio.utils import _dft_mag

try:  # optional acceleration; the fallback feature works without it
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

_EPS = 1e-10


def frame_spectrum(audio: Sequence[float], sr: int, n_bins: int = 48) -> List[float]:
    """Non-negative log-spaced magnitude spectrum for one window (stdlib path)."""
    mags, _ = _dft_mag(audio, sr, n_bins)
    return [abs(m) for m in mags]


def _l1norm(v: List[float]) -> List[float]:
    s = sum(v) or 1.0
    return [x / s for x in v]


def learn_basis(spectra: List[List[float]], n_components: int = 6,
                n_iter: int = 150, seed: int = 0):
    """NMF V ≈ W·H via multiplicative updates; returns W (freq × k) with
    L1-normalised columns, or None if numpy is unavailable / not enough data.

    Reserved for the next step (per-source / online bases). The live tracker uses
    the fixed bootstrap basis so feature dimensions never shift mid-track.
    """
    if _np is None or len(spectra) < 2:
        return None
    V = _np.asarray(spectra, dtype=float).T + _EPS     # F × T (each spectrum a column)
    F, T = V.shape
    k = max(1, min(n_components, T))
    rng = _np.random.default_rng(seed)
    W = rng.random((F, k)) + 0.1
    H = rng.random((k, T)) + 0.1
    for _ in range(n_iter):
        H *= (W.T @ V) / (W.T @ W @ H + _EPS)
        W *= (V @ H.T) / (W @ H @ H.T + _EPS)
    W /= (W.sum(axis=0, keepdims=True) + _EPS)
    return W


def decompose_frame(v: Sequence[float], W, n_iter: int = 50) -> List[float]:
    """Solve v ≈ W·h with h ≥ 0 (multiplicative updates); return L1-normalised h.

    Deterministic: h starts uniform, so the same spectrum always yields the same
    activations — important for a stable tracker feature."""
    vv = _np.asarray(v, dtype=float) + _EPS
    k = W.shape[1]
    h = _np.full(k, 1.0 / k)
    WtW = W.T @ W
    Wtv = W.T @ vv
    for _ in range(n_iter):
        h *= Wtv / (WtW @ h + _EPS)
    return (h / (h.sum() + _EPS)).tolist()


def _bootstrap_basis(n_bins: int, k: int):
    """A fixed universal basis: k overlapping triangular spectral bands across the
    log-spaced bins. Captures where a source's energy sits (its spectral tilt /
    formant region) — which is stable across frames of the same source."""
    if _np is None:
        return None
    idx = _np.arange(n_bins, dtype=float)
    centers = _np.linspace(0.0, n_bins - 1, k) if k > 1 else _np.array([(n_bins - 1) / 2.0])
    width = max(1.0, (n_bins - 1) / (k - 1)) if k > 1 else float(n_bins)
    W = _np.maximum(0.0, 1.0 - _np.abs(idx[:, None] - centers[None, :]) / width)
    W /= (W.sum(axis=0, keepdims=True) + _EPS)
    return W


def _band_group(spec: List[float], k: int) -> List[float]:
    """Stdlib fallback: sum the spectrum into k contiguous bands (same dimension as
    the numpy activations), L1-normalised. Frame-stable like the basis path."""
    F = len(spec)
    out = [0.0] * k
    for i, m in enumerate(spec):
        out[min(k - 1, i * k // max(1, F))] += m
    return _l1norm(out)


class SpectralLibrary:
    """Per-frame tracking features: a fixed-dimension (k) spectral activation that
    is frame-stable for a given source — the continuity signal ECAPA cannot give.
    Dimension never changes during a session, so a track's stored feature stays
    comparable frame to frame."""

    def __init__(self, n_bins: int = 48, n_components: int = 6, fit_after: int = 24,
                 learn_iters: int = 150, decompose_iters: int = 50,
                 vad: float = 0.0, seed: int = 0):
        self.n_bins = n_bins
        self.k = n_components
        self.decompose_iters = decompose_iters
        self.W = _bootstrap_basis(n_bins, n_components)   # None when no numpy

    def feature(self, audio: Sequence[float], sr: int) -> List[float]:
        """Frame feature for tracker association — a k-dim activation vector."""
        spec = frame_spectrum(audio, sr, self.n_bins)
        if self.W is not None and _np is not None:
            return decompose_frame(spec, self.W, self.decompose_iters)
        return _band_group(spec, self.k)

    @property
    def has_basis(self) -> bool:
        return self.W is not None
