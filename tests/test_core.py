"""Core plumbing tests — all run on pure stdlib (no heavy deps required).

These pin the deterministic guarantees: the enrollment confidence curve, cosine
similarity, the priority math + safety/distress overrides, UNKNOWN-NNN tracking,
and an end-to-end synthetic listen producing a valid awareness map.
"""
import math

import pytest

from daredevil import Pipeline
from daredevil.config import Config, is_safety_critical
from daredevil.audio.utils import cosine, fingerprint
from daredevil.enrollment.manager import enrollment_confidence
from daredevil.stage3.router import AttentionRouter
from daredevil.stage3.tracker import UnknownTracker


def test_enrollment_confidence_curve():
    # Patent: C(t) = 1 - exp(-t/3). 3s≈0.63, 10s≈0.96, 20s≈0.999.
    assert enrollment_confidence(3) == pytest.approx(0.632, abs=0.01)
    assert enrollment_confidence(10) == pytest.approx(0.964, abs=0.01)
    assert enrollment_confidence(20) == pytest.approx(0.999, abs=0.005)


def test_cosine_basics():
    assert cosine([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_fingerprint_stable_and_discriminative():
    import random
    rng = random.Random(0)
    a = [math.sin(2 * math.pi * 150 * i / 16000) for i in range(16000)]
    a_noisy = [v + rng.uniform(-0.01, 0.01) for v in a]
    b = [math.sin(2 * math.pi * 480 * i / 16000) for i in range(16000)]
    fa, fa2, fb = (fingerprint(a, 16000), fingerprint(a_noisy, 16000), fingerprint(b, 16000))
    assert cosine(fa, fa2) > 0.95        # same source, stable
    assert cosine(fa, fb) < cosine(fa, fa2)  # different source, less similar


def test_is_safety_critical():
    assert is_safety_critical("baby_cry")
    assert is_safety_critical("Baby cry, infant cry")
    assert is_safety_critical("Smoke detector, smoke alarm")
    assert not is_safety_critical("speech")
    assert not is_safety_critical("")


def test_router_safety_override_pins_priority():
    r = AttentionRouter(Config())
    rec = {
        "identity": None, "unknown_id": "UNKNOWN-001",
        "event": {"class": "baby_cry", "confidence": 0.95, "safety_critical": True},
        "prosody": {"state": "distressed", "distress": 0.9},
        "position": {"azimuth": 45.0, "elevation": 0.0},
    }
    amap = r.build([rec], "t", {"parallel_ms": 1, "sequential_ms": 2}, _Arr(), "fallback")
    src = amap["sources"][0]
    assert src["type"] == "unknown"
    assert src["priority"] == 1.0
    assert src["priority_override"] == "SAFETY_CRITICAL"
    assert amap["privacy"]["cloud_used"] is False


def test_router_distress_escalates_enrolled_speaker():
    r = AttentionRouter(Config())
    rec = {
        "identity": {"name": "alan", "score": 0.9, "enrollment_confidence": 0.96},
        "event": {"class": "speech", "confidence": 0.97, "safety_critical": False},
        "prosody": {"state": "distressed", "distress": 0.8},
        "position": None,
    }
    amap = r.build([rec], "t", {"parallel_ms": 1, "sequential_ms": 2}, _Arr(), "fallback")
    src = amap["sources"][0]
    assert src["type"] == "enrolled" and src["id"] == "alan"
    assert src["priority_override"] == "DISTRESS"
    assert src["priority"] >= 0.85


def test_tracker_persistent_unknown_ids():
    t = UnknownTracker(threshold=0.8)
    v1 = fingerprint([math.sin(2 * math.pi * 150 * i / 16000) for i in range(8000)], 16000)
    v2 = fingerprint([math.sin(2 * math.pi * 470 * i / 16000) for i in range(8000)], 16000)
    id1 = t.assign(v1)
    assert t.assign(v1) == id1          # same source -> same id
    id2 = t.assign(v2)
    assert id2 != id1                   # different source -> new id
    assert id1 == "UNKNOWN-001" and id2 == "UNKNOWN-002"


def test_end_to_end_synthetic(tmp_path):
    from daredevil.stage1.mic_arrays import MACBOOK_3
    cfg = Config(data_dir=str(tmp_path))
    pipe = Pipeline(config=cfg, array=MACBOOK_3)

    enr = pipe.enroll("alan", mic_seconds=3, source="synthetic")
    assert enr["enrollment_confidence"] == pytest.approx(0.632, abs=0.01)

    # Two windows: since the anti-blip frame-LLR clip, no single frame can decide
    # an identity alone — acquisition takes two agreeing frames (track persists).
    pipe.listen(duration=1.0, source="synthetic")
    amap = pipe.listen(duration=1.0, source="synthetic")
    assert amap["privacy"]["cloud_used"] is False
    assert amap["array"]["spatial"] is True
    ids = {s["id"]: s for s in amap["sources"]}

    assert "alan" in ids and ids["alan"]["type"] == "enrolled"
    assert ids["alan"]["identity"]["match_score"] > 0.7   # fallback fingerprint identifies

    baby = [s for s in amap["sources"] if s["event"]["class"] == "baby_cry"][0]
    assert baby["priority"] == 1.0
    assert baby["priority_override"] == "SAFETY_CRITICAL"
    # safety-critical source outranks the calm enrolled speaker
    assert amap["sources"][0]["id"] == baby["id"]


def test_attention_gate_routes_subset(tmp_path):
    """A struct is built for every source, but the radio is gated out of the LLM."""
    from daredevil.stage1.mic_arrays import MACBOOK_3
    from daredevil.stage3.router import llm_payload
    pipe = Pipeline(config=Config(data_dir=str(tmp_path)), array=MACBOOK_3)
    pipe.enroll("alan", mic_seconds=3, source="synthetic")
    pipe.listen(duration=1.0, source="synthetic")   # frame 1 arms (2-frame acquisition)
    amap = pipe.listen(duration=1.0, source="synthetic")

    classes = {s["event"]["class"] for s in amap["sources"]}
    assert {"speech", "baby_cry", "music"} <= classes   # all detected -> all get structs

    routed = amap["routed_to_llm"]
    music = [s for s in amap["sources"] if s["event"]["class"] == "music"][0]
    assert music["attention"] == "ambient"        # the radio is heard...
    # The music source is gated out — it's ambient. With only one enrollment and a
    # real model, its embedding may still cosine-match above threshold (no other
    # speaker to discriminate against). What matters is the gate, not the id.
    surfaced_sources = [s for s in amap["sources"] if s["attention"] == "surface"]
    assert music not in surfaced_sources          # ...but gated out of the conversation

    baby = [s for s in amap["sources"] if s["event"]["class"] == "baby_cry"][0]
    assert baby["id"] in routed                   # safety-critical surfaces
    assert "alan" in routed                       # the enrolled owner speaking surfaces
    assert {s["id"] for s in llm_payload(amap)} == set(routed)


class _Arr:
    name = "test"
    n_mics = 3
    spatial_capable = True


def test_speech_gate_passes_quiet_speech():
    """The SPRT speech gate must admit normal-distance laptop speech.

    A hardcoded 0.05 RMS floor once muted identification for anyone not
    leaning into the mic (speech at arm's length is typically 0.005-0.03 RMS).
    The gate values are config tunables now; quiet-but-real speech must vote.
    """
    from daredevil.audio.utils import is_speech_quality

    sr = 16000
    # Voiced-speech stand-in: 150 Hz fundamental + harmonics at 0.02 RMS —
    # below the old 0.05 floor, above the configured 0.012 default.
    n = sr  # 1 second
    voiced = [(math.sin(2 * math.pi * 150 * t / sr)
               + 0.5 * math.sin(2 * math.pi * 300 * t / sr)
               + 0.25 * math.sin(2 * math.pi * 600 * t / sr)) for t in range(n)]
    rms = math.sqrt(sum(x * x for x in voiced) / n)
    quiet = [x * (0.02 / rms) for x in voiced]

    th = Config().thresholds
    assert is_speech_quality(quiet, sr, energy_gate=th.speech_gate_energy,
                             zcr_gate=th.speech_gate_zcr)
    # Sanity on the other side of the gate: silence is not evidence.
    assert not is_speech_quality([0.0] * n, sr, energy_gate=th.speech_gate_energy,
                                 zcr_gate=th.speech_gate_zcr)
    # Clicks/noise: energetic but far above any voiced zero-crossing rate.
    clicks = [0.2 if t % 2 else -0.2 for t in range(n)]
    assert not is_speech_quality(clicks, sr, energy_gate=th.speech_gate_energy,
                                 zcr_gate=th.speech_gate_zcr)
    # The tunables exist and keep the documented relationship to the VAD floor.
    assert th.speech_gate_energy > th.vad


def test_is_speech_class_across_backends():
    """Gap M3: PANNs emits AudioSet labels verbatim; fallback emits lowercase.
    Exact lowercase comparison made active_speaker permanently None live."""
    from daredevil.config import is_speech_class
    assert is_speech_class("speech")
    assert is_speech_class("Speech")
    assert is_speech_class("Male speech, man speaking")
    assert not is_speech_class("Music")
    assert not is_speech_class("baby_cry")
    assert not is_speech_class(None)
    assert not is_speech_class("")


def test_dedupe_identity_claims_one_track_per_person():
    """Gap M15 slice: two tracks claiming the same name — live speech wins, the
    stale/held claimant is forgotten (observed live: Speech-Alan + noise-Alan)."""
    from daredevil.pipeline import _dedupe_identity_claims

    forgotten = []
    class _Enr:
        def forget_track(self, key): forgotten.append(key)

    speech = {"identity": {"name": "Alan", "score": 0.99}, "unknown_id": "UNKNOWN-001",
              "event": {"class": "Speech"}}
    stale = {"identity": {"name": "Alan", "score": 0.99, "held": True},
             "unknown_id": "UNKNOWN-007", "event": {"class": "White noise"}}
    records = [stale, speech]
    _dedupe_identity_claims(records, _Enr())
    assert speech["identity"] is not None, "the speaking track lost its name"
    assert stale["identity"] is None, "the stale claimant kept the name"
    assert forgotten == ["UNKNOWN-007"]
