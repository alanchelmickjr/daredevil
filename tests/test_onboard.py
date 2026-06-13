"""Onboarding 'pick me out of the crowd' tests — deterministic, no hardware.

The reveal must be *real*: the enrolled speaker is surfaced by genuine cosine-SPRT
matching while the synthetic crowd is heard, tracked, and gated out — not faked.
"""
from __future__ import annotations

from daredevil.config import Config
from daredevil.pipeline import Pipeline
from daredevil.stage1.mic_arrays import MACBOOK_3
from daredevil.audio.crowd import crowd_scene_sources


def test_recognizes_you_against_the_crowd(tmp_path, monkeypatch):
    monkeypatch.setenv("DAREDEVIL_HOME", str(tmp_path))
    pipe = Pipeline(config=Config(), array=MACBOOK_3)
    pipe.enroll("huan", 3, source="synthetic")

    scene = [{"name": "huan", "enrolled": True, "class": "speech", "azimuth": 0.0,
              "elevation": 0.0, "prosody_state": "calm", "distress": 0.1}]
    scene += crowd_scene_sources(n_speakers=4)

    you_surfaced = False
    crowd_gated = 0
    for _ in range(3):  # let the SPRT accumulate over a few windows
        amap = pipe.listen(duration=1.0, source="synthetic", scene=scene)
        you = next((s for s in amap["sources"] if s["id"] == "huan"), None)
        if you and you.get("attention") == "surface":
            you_surfaced = True
        crowd_gated = sum(1 for s in amap["sources"]
                          if s["id"] != "huan" and s.get("attention") != "surface")
        # no crowd voice should ever be mislabeled as you
        assert sum(1 for s in amap["sources"] if s["id"] == "huan") <= 1

    assert you_surfaced, "the enrolled speaker was not picked out of the crowd"
    assert crowd_gated >= 1, "the crowd should be heard but gated out of the conversation"


def test_onboard_synthetic_runs_clean(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DAREDEVIL_HOME", str(tmp_path))
    from daredevil.onboard import onboard
    rc = onboard(name="huan", live=False, crowd=3, windows=2, seconds=3.0)
    out = capsys.readouterr().out
    assert rc == 0
    assert "I'm Radar" in out
    assert "huan" in out
    assert "Picked you out" in out  # the reveal reached its happy path
