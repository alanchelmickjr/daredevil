"""Wake-word (attention-by-name) tests — pure stdlib, deterministic.

We can't play real speech in CI, so we prove the *mechanism*: the by-example
detector matches a phrase against a time-warped copy of itself, discriminates a
different temporal contour, rejects broadband noise, persists to disk, and wires
into the pipeline/awareness-map without breaking anything. Thresholds still want
on-device tuning with a real voice — that's called out in the docs.
"""
from __future__ import annotations

import math
import random

from daredevil.config import Config
from daredevil.stage2.wake import (
    WakeWordDetector, feature_sequence, subsequence_dtw,
)


def _chirp(f0: float, f1: float, dur: float, sr: int, amp: float = 0.3) -> list:
    """A tonal sweep f0->f1 — a moving spectral peak, the way a phrase's formants
    move. Distinct sweeps stand in for distinct words."""
    n = int(dur * sr)
    out = []
    for i in range(n):
        t = i / sr
        f = f0 + (f1 - f0) * (t / dur)
        out.append(amp * math.sin(2 * math.pi * f * t))
    return out


def _noise(dur: float, sr: int, seed: int = 7) -> list:
    r = random.Random(seed)
    return [r.uniform(-0.3, 0.3) for _ in range(int(dur * sr))]


SR = 16000


def test_feature_sequence_trims_silence_and_drops_silent_frames():
    p = Config().wakeword
    voiced = _chirp(300, 1500, 0.8, SR)
    padded = [0.0] * int(0.3 * SR) + voiced + [0.0] * int(0.3 * SR)
    seq = feature_sequence(padded, SR, p)
    # Trimmed back to ~the voiced span (silence excluded), every frame unit-norm.
    assert len(seq) > 30
    assert len(seq) < int(1.4 / (p.hop_ms / 1000.0))  # shorter than the padded length
    for v in seq:
        assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-6
    assert feature_sequence([0.0] * SR, SR, p) == []


def test_dtw_self_similarity_is_high_and_direction_sensitive():
    p = Config().wakeword
    up = feature_sequence(_chirp(300, 1500, 0.8, SR), SR, p)
    down = feature_sequence(_chirp(1500, 300, 0.8, SR), SR, p)
    self_sim = subsequence_dtw(up, up)
    cross = subsequence_dtw(up, down)         # reversed contour, monotonic DTW can't match
    assert self_sim > 0.95
    assert self_sim - cross > 0.1


def test_enroll_then_detect_roundtrip(tmp_path):
    p = Config().wakeword
    det = WakeWordDetector(p, tmp_path, phrase="Hey Radar")
    assert not det.ready()                      # nothing learned yet

    name_audio = _chirp(300, 1500, 0.8, SR)
    res = det.enroll(name_audio, SR)
    assert res["ok"] and res["samples"] == 1
    assert det.ready() and det.enrolled()

    # Same phrase (with a little noise + silence around it) wakes it.
    noisy = [x + random.Random(1).uniform(-0.02, 0.02) for x in name_audio]
    embedded = [0.0] * int(0.2 * SR) + noisy + [0.0] * int(0.2 * SR)
    hit = det.detect(embedded, SR)
    assert hit["detected"] is True
    assert hit["score"] >= p.threshold
    assert hit["backend"] == "template"

    # Broadband noise is not the name.
    miss = det.detect(_noise(1.0, SR), SR)
    assert miss["detected"] is False
    assert miss["score"] < hit["score"]


def test_template_persists_across_instances(tmp_path):
    p = Config().wakeword
    WakeWordDetector(p, tmp_path, phrase="Hey Radar").enroll(_chirp(300, 1500, 0.8, SR), SR)
    reloaded = WakeWordDetector(p, tmp_path, phrase="Hey Radar")   # fresh process
    assert reloaded.enrolled()
    assert reloaded.detect(_chirp(300, 1500, 0.8, SR), SR)["detected"] is True


def test_disabled_and_unenrolled_are_safe(tmp_path):
    p = Config().wakeword
    det = WakeWordDetector(p, tmp_path, phrase="Hey Radar")
    # Not enrolled -> never fires, never errors.
    assert det.detect(_chirp(300, 1500, 0.8, SR), SR)["detected"] is False
    p.enabled = False
    out = det.detect(_chirp(300, 1500, 0.8, SR), SR)
    assert out["detected"] is False and out["backend"] == "disabled"


def test_pipeline_attaches_wake_summary(tmp_path, monkeypatch):
    """listen() always carries a stable `wake` field; unenrolled => detected None,
    and the map still builds cleanly."""
    monkeypatch.setenv("DAREDEVIL_HOME", str(tmp_path))
    from daredevil.pipeline import Pipeline
    from daredevil.stage1.mic_arrays import MACBOOK_3
    pipe = Pipeline(config=Config(), array=MACBOOK_3)
    pipe.enroll("alan", 3, source="synthetic")
    amap = pipe.listen(duration=1.0, source="synthetic")
    assert "wake" in amap
    assert amap["wake"]["detected"] is None
    assert amap["wake"]["name"] == "Hey Radar"
