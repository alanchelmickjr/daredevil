"""Crowd/babble generator tests — pure stdlib, deterministic.

The crowd is generated DSP (labeled SYNTHETIC), never a recording. We check it is
well-formed, deterministic, distinct per seed, and that playback degrades honestly
when there's no audio device.
"""
from __future__ import annotations

from daredevil.audio.crowd import babble, crowd_scene_sources, CrowdPlayer, CROWD_NAMES


SR = 16000


def test_babble_is_wellformed_and_deterministic():
    a = babble(n_speakers=4, seconds=1.0, sr=SR, seed=3)
    b = babble(n_speakers=4, seconds=1.0, sr=SR, seed=3)
    c = babble(n_speakers=4, seconds=1.0, sr=SR, seed=4)
    assert len(a) == SR
    assert all(-1.0 <= x <= 1.0 for x in a)
    assert a == b                      # same seed -> identical (time-machine friendly)
    assert a != c                      # different seed -> a different crowd
    # it actually carries energy (it's a murmur, not silence)
    assert sum(x * x for x in a) / len(a) > 1e-4


def test_more_speakers_still_bounded():
    big = babble(n_speakers=6, seconds=0.5, sr=SR, seed=1)
    assert len(big) == SR // 2
    assert all(-1.0 <= x <= 1.0 for x in big)


def test_crowd_scene_sources_are_unknown_and_unnamed():
    src = crowd_scene_sources(n_speakers=4)
    assert len(src) == 4
    for s in src:
        assert s["enrolled"] is False
        assert s["class"] == "speech"
        assert s["name"] in CROWD_NAMES or s["name"].startswith("voice-")
    # spread around the room (distinct bearings)
    azimuths = {s["azimuth"] for s in src}
    assert len(azimuths) == 4


def test_player_degrades_honestly_without_audio():
    # No sounddevice in CI: start() must not raise and must report unavailable.
    with CrowdPlayer(n_speakers=3, sr=SR) as p:
        assert p.available is False
