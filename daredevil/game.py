"""`daredevil game` — GUARDIAN: race the machine to the call that matters.

Each round a SYNTHETIC scene plays. You see the awareness map and call it:
  [a] ACT NOW   [w] WATCH   [i] IGNORE
Daredevil + a decision model make the same call in parallel. Both times are
measured, never invented. Rounds include a *wrong-quiet* room: the sound that
should be there is missing — the scariest signal is an absence.

Decider: Jev-Style (local, `pip install jev-style`, needs DAREDEVIL_JEV_SCORER)
when available; otherwise the Stage-3 router rule, labeled as such. No cloud.
"""
from __future__ import annotations

import os
import random
import time

from .config import Config, is_safety_critical
from .pipeline import Pipeline
from .viz.spatial_map import render_ascii

CHOICES = {"a": "ACT NOW", "w": "WATCH", "i": "IGNORE"}

_ROUNDS = [
    ("Nursery, 3am", [
        {"name": None, "class": "baby_cry", "azimuth": 45.0, "elevation": 10.0,
         "prosody_state": "distressed", "distress": 0.95}], "a"),
    ("Living room, Sunday", [
        {"name": None, "class": "music", "azimuth": 270.0, "elevation": 0.0,
         "prosody_state": "calm", "distress": 0.05}], "i"),
    ("Kitchen", [
        {"name": None, "class": "smoke_detector", "azimuth": 120.0, "elevation": 30.0,
         "prosody_state": "calm", "distress": 0.1},
        {"name": None, "class": "music", "azimuth": 270.0, "elevation": 0.0,
         "prosody_state": "calm", "distress": 0.05}], "a"),
    ("Front door", [
        {"name": "stranger", "enrolled": False, "class": "speech", "azimuth": 180.0,
         "elevation": 0.0, "prosody_state": "calm", "distress": 0.2}], "w"),
    ("Hallway, kid's voice", [
        {"name": "kid", "enrolled": False, "class": "speech", "azimuth": 90.0,
         "elevation": 0.0, "prosody_state": "distressed", "distress": 0.9}], "a"),
    # wrong-quiet: the baby monitor should hear breathing; it hears nothing.
    ("Nursery monitor — expecting breathing", [], "a"),
]


def _live(amap: dict) -> list:
    # drop lingering tracks the tracker still carries at zero confidence
    return [s for s in amap.get("sources", []) if s["event"].get("confidence", 0) > 0]


def _describe(amap: dict, expected: str | None) -> str:
    parts = [f"{s['event']['class']} at {s['position']['azimuth']:.0f}°, "
             f"{s['prosody']['state']} (distress {s['prosody']['distress']:.2f})"
             for s in _live(amap)]
    heard = "; ".join(parts) or "silence"
    return f"Heard: {heard}." + (f" Expected: {expected}." if expected else "")


class _Decider:
    """Jev-Style when installed + configured; else the router rule (honestly named)."""

    def __init__(self):
        self.name = "router-rule"
        self._js = None
        scorer = os.environ.get("DAREDEVIL_JEV_SCORER")
        try:
            from jev_style import JevStyle, noul  # lazy, optional, local-only
            if scorer:
                self._js = JevStyle.from_pretrained(
                    "chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF",
                    quant="Q4_K_M", scorer=scorer)
                self._noul = noul
                self.name = "jev-style-0.8B (local)"
        except Exception:
            self._js = None

    def decide(self, amap: dict, expected: str | None) -> str:
        if self._js is not None:
            try:
                labels = {
                    "a": self._noul("Someone may be in danger; act now."),
                    "w": self._noul("Unusual but not dangerous yet; keep watching."),
                    "i": self._noul("Ordinary background; nothing to do."),
                }
                out = self._js.decide(_describe(amap, expected), labels)
                key = getattr(out, "label", None) or (out.get("label") if isinstance(out, dict) else out)
                if key in CHOICES:
                    return key
            except Exception:
                pass
        return self._rule(amap, expected)

    @staticmethod
    def _rule(amap: dict, expected: str | None) -> str:
        srcs = _live(amap)
        if expected and not any(expected in s["event"]["class"] for s in srcs):
            return "a"                                   # wrong-quiet
        if any(s["event"].get("safety_critical") or is_safety_critical(s["event"]["class"])
               or s["prosody"]["distress"] >= 0.8 for s in srcs):
            return "a"
        if any(s["event"]["class"] == "speech" and s.get("type") == "unknown" for s in srcs):
            return "w"
        return "i"


def _silence(pipe: Pipeline) -> dict:
    # synthetic_scene() treats an empty spec as "default scene", so build the
    # silent room directly: a real (zero) capture through the real pipeline.
    from .audio.capture import CaptureResult
    from .stage1.mic_arrays import SINGLE
    sr = pipe.config.capture_rate
    cap = CaptureResult(channels=[[0.0] * sr], sample_rate=sr, array=SINGLE,
                        synthetic=True, scene_truth=[], source="synthetic-silence")
    return pipe.process_capture(cap)


def run_game(rounds: int = 5, seed: int | None = None) -> int:
    rng = random.Random(seed)
    picks = rng.sample(_ROUNDS[:-1], min(rounds - 1, len(_ROUNDS) - 1)) + [_ROUNDS[-1]]
    rng.shuffle(picks)
    from .stage1.mic_arrays import MACBOOK_3
    judge = _Decider()
    you = bot = 0
    print("\n  G U A R D I A N   — you vs Daredevil + " + judge.name)
    print("  SYNTHETIC scenes · all inference on-device · times measured\n")
    for n, (title, scene, truth) in enumerate(picks, 1):
        expected = "breathing" if not scene else None
        pipe = Pipeline(config=Config(), array=MACBOOK_3)   # fresh room each round
        pipe.warmup()
        t0 = time.perf_counter()
        amap = pipe.listen(source="synthetic", scene=scene) if scene else _silence(pipe)
        call = judge.decide(amap, expected)
        machine_ms = (time.perf_counter() - t0) * 1000
        print(f"\n━━ ROUND {n}: {title} " + "━" * max(0, 40 - len(title)))
        print(render_ascii(amap) if scene else "  (the room is silent)" +
              (f"  — expected: {expected}" if expected else ""))
        t1 = time.perf_counter()
        ans = ""
        while ans not in CHOICES:
            ans = input("  [a] ACT NOW  [w] WATCH  [i] IGNORE > ").strip().lower()[:1]
        you_ms = (time.perf_counter() - t1) * 1000
        you_ok, bot_ok = ans == truth, call == truth
        you += you_ok
        bot += bot_ok
        print(f"  truth: {CHOICES[truth]}")
        print(f"  you:     {CHOICES[ans]:8} {'✓' if you_ok else '✗'}  {you_ms:7.0f} ms")
        print(f"  machine: {CHOICES[call]:8} {'✓' if bot_ok else '✗'}  {machine_ms:7.0f} ms")
        if not scene:
            print("  ↳ wrong-quiet: no sound was the alarm.")
    print(f"\n  FINAL  you {you}/{len(picks)}  ·  machine {bot}/{len(picks)}")
    print("  It never sleeps, never blinks, and never sends a byte off-device.\n")
    return 0
