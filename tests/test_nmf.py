"""NMF spectral-basis tracking-feature tests (pure stdlib).

The point of the feature: frame-to-frame STABILITY for the same source and
SEPARATION between different sources — the continuity signal ECAPA can't give.
Works with numpy (true NMF activations) or without (band-group fallback); both
return a fixed-dimension, frame-stable feature.
"""
import math

from daredevil.stage1.nmf import SpectralLibrary, frame_spectrum
from daredevil.audio.utils import cosine


def _tone(freqs, sr=16000, n=16000, amps=None):
    amps = amps or [1.0] * len(freqs)
    out = []
    for i in range(n):
        t = i / sr
        out.append(sum(a * math.sin(2 * math.pi * f * t) for f, a in zip(freqs, amps)))
    return out


def test_feature_is_fixed_dimension():
    lib = SpectralLibrary(n_bins=48, n_components=6)
    f1 = lib.feature(_tone([180, 430]), 16000)
    f2 = lib.feature(_tone([900]), 16000)
    assert len(f1) == 6 and len(f2) == 6   # dimension never shifts


def test_same_source_is_frame_stable():
    """Two different 1s windows of the same source -> high feature cosine."""
    lib = SpectralLibrary()
    # same source content, two distinct windows (phase/segment differ)
    a1 = _tone([160, 320, 640], n=16000)
    a2 = _tone([160, 320, 640], n=16000)[2000:] + _tone([160, 320, 640], n=2000)
    fa1, fa2 = lib.feature(a1, 16000), lib.feature(a2, 16000)
    assert cosine(fa1, fa2) > 0.95


def test_different_sources_separate():
    """A low-pitched voice-like source vs a high tonal source -> lower cosine."""
    lib = SpectralLibrary()
    voice = lib.feature(_tone([140, 280, 560], amps=[1.0, 0.6, 0.3]), 16000)
    high = lib.feature(_tone([1600, 2000]), 16000)
    same = lib.feature(_tone([140, 280, 560], amps=[1.0, 0.6, 0.3]), 16000)
    assert cosine(voice, high) < cosine(voice, same)
    assert cosine(voice, high) < 0.9


def test_tracker_holds_one_source_on_spectral_feature():
    """End-to-end: the tracker keeps one ongoing source as one track using the
    NMF feature across frames where ECAPA would jitter."""
    from daredevil.stage3.tracker import UnknownTracker
    lib = SpectralLibrary()
    t = UnknownTracker()
    ids = set()
    for seg in range(6):
        # same source, different 1s slice each frame
        audio = _tone([150, 300, 600], n=16000)
        ids.add(t.assign(lib.feature(audio, 16000)))
    assert len(ids) == 1
