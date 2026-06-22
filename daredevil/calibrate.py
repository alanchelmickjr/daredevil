"""First-run calibration — a short "let's get to know each other" session.

This is the human-in-the-loop seed for the SPRT identity model: the person talks
to Radar for about a minute, and we *measure* the real cosine distributions of
their voice and their room, fit an `IdentityModel`, and save it. From then on the
matcher auto-tunes the background in real time (CFAR) from this seed instead of
cold textbook defaults.

Two halves, deliberately separated:
  * `Calibrator` — pure measurement/fit/save (testable, no I/O prompts).
  * `session()`  — the conversational UX that drives it (prompts + live feedback).

Honest by construction: every number printed is measured from the captured audio.
In synthetic mode the scene is labeled SYNTHETIC and the same real math runs.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import asdict
from typing import List, Optional, Tuple

from .config import Config, IdentityModel, calibration_path
from .audio.capture import capture
from .audio.utils import cosine, resample, rms

log = logging.getLogger("daredevil.calibrate")


class Calibrator:
    """Measures voice/background cosine distributions and fits an IdentityModel."""

    def __init__(self, pipeline):
        self.pipe = pipeline
        self.cfg: Config = pipeline.config
        self.embed = pipeline.slots["embedding"]

    # --- capture + measurement -------------------------------------------
    def capture_audio(self, seconds: float, source: str,
                      file: Optional[str] = None, name: Optional[str] = None,
                      scene: Optional[list] = None):
        cap = capture(seconds=seconds, sr=self.cfg.capture_rate, source=source,
                      file=file, array=self.pipe.array, name=name, scene=scene)
        audio = resample(cap.mono, cap.sample_rate, self.cfg.inference_sr)
        return audio, self.cfg.inference_sr, rms(cap.mono), cap.synthetic

    def _windows(self, audio: List[float], sr: int, win: float = 1.0):
        n = int(win * sr)
        if len(audio) <= n:
            return [audio]
        return [audio[i:i + n] for i in range(0, len(audio) - n + 1, n)]

    def cosines_to(self, audio: List[float], sr: int, voiceprint: List[float]) -> List[float]:
        """Cosine of each voiced window's embedding against the voiceprint."""
        out = []
        for w in self._windows(audio, sr):
            if rms(w) <= self.cfg.thresholds.vad:
                continue
            v = self.embed.run(w, sr)["vector"]
            out.append(cosine(v, voiceprint))
        return out

    # --- fit + persist ----------------------------------------------------
    def fit(self, target_cos: List[float], bg_cos: List[float]) -> Tuple[IdentityModel, float]:
        base = self.cfg.identity
        tm = statistics.fmean(target_cos) if target_cos else base.target_mean
        ts = statistics.pstdev(target_cos) if len(target_cos) > 1 else base.target_std
        if bg_cos:
            im = statistics.fmean(bg_cos)
            istd = statistics.pstdev(bg_cos) if len(bg_cos) > 1 else base.impostor_std
        else:
            im, istd = base.impostor_mean, base.impostor_std
        ts, istd = max(ts, 0.03), max(istd, 0.03)   # floor std so the model isn't over-confident

        model = IdentityModel(
            target_mean=round(tm, 4), target_std=round(ts, 4),
            impostor_mean=round(im, 4), impostor_std=round(istd, 4),
            # carry policy/behaviour knobs through unchanged
            alpha=base.alpha, beta=base.beta, leak=base.leak,
            immediate_cosine=base.immediate_cosine,
            quality_full_energy=base.quality_full_energy,
            adapt_background=base.adapt_background, bg_adapt_rate=base.bg_adapt_rate,
            bg_guard_sigmas=base.bg_guard_sigmas, adapt_target=base.adapt_target,
            target_adapt_rate=base.target_adapt_rate,
            target_adapt_margin=base.target_adapt_margin,
        )
        # d-prime: how separable the two distributions are (higher = cleaner).
        denom = ((ts * ts + istd * istd) / 2.0) ** 0.5
        dprime = round((tm - im) / denom, 2) if denom > 0 else 0.0
        log.info("calibration fit: target=%.3f±%.3f, bg=%.3f±%.3f, d'=%.2f",
                 tm, ts, im, istd, dprime)
        return model, dprime

    def save(self, model: IdentityModel) -> str:
        import json
        path = calibration_path(self.cfg.resolved_data_dir())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(model), indent=2))
        log.info("calibration saved to %s", path)
        return str(path)


# --- the conversational session ------------------------------------------

_VOICE_PROMPTS = [
    "Hey — I'm Radar. Tell me your name and what you're working on today.",
    "Nice. Now, what's the last thing that made you laugh?",
]

# A speaker-free ambient scene for the synthetic room phase (music + noise floor),
# so the simulated background is representative rather than echoing the voice.
_AMBIENT_SCENE = [
    {"name": None, "enrolled": False, "class": "music", "azimuth": 270.0,
     "elevation": 0.0, "prosody_state": "calm", "distress": 0.05},
]


def _level_note(level: float, h=None) -> str:
    from .config import HeuristicThresholds
    h = h or HeuristicThresholds()
    if level < h.level_too_quiet:
        return "barely hear you — scoot a little closer"
    if level > h.level_too_hot:
        return "whoa, a touch hot — ease back a hair"
    return "loud and clear"


def session(name: Optional[str] = None, live: bool = False, seconds: float = 20.0,
            others: bool = False, config: Optional[Config] = None) -> dict:
    """Run the onboarding session. Returns {model, dprime, path, stats}.

    Non-interactive (synthetic / tests) runs straight through; live mode pauses
    for the person between phases (no sleeps — it waits on them, not the clock).
    """
    from .pipeline import Pipeline

    source = "live" if live else "synthetic"
    name = name or "you"
    pipe = Pipeline(config=config or Config())
    pipe.warmup()
    cal = Calibrator(pipe)

    def wait(msg: str):
        if live:
            try:
                input(msg)
            except EOFError:
                pass

    print("\n🦇  Radar — let's get to know each other")
    print("    (not a setup wizard — just talk to me for a minute)")
    if not live:
        print("    SYNTHETIC mode: simulating a voice + room so the flow is exact.\n")

    # Wipe any stale voiceprint so we start clean on this mic/room.
    if pipe.store.get(name):
        pipe.store.delete(name)

    # --- Phase 1: your voice (fresh enrollment + measure same-speaker cosine)
    # ONE recording, used for both enrollment and verification measurement.
    print("▶ 1/3  SPEAK — talk to me like I'm across the table")
    print('   Say anything — what you had for breakfast, curse at the traffic, whatever.')
    wait("   ↵ press Enter, then talk… ")
    audio, sr, level, synth = cal.capture_audio(seconds, source, name=name)
    enr = pipe.enrollment.enroll(audio, sr, name, seconds)
    voiceprint = pipe.store.get(name)["vector"]
    target_cos = cal.cosines_to(audio, sr, voiceprint)
    print(f"   ✓ got you — {_level_note(level)}  (level {level:.2f}, "
          f"{len(target_cos)} voice frames, conf {enr['enrollment_confidence']:.2f})")

    # --- Phase 2: your room (background) ----------------------------------
    print("\n▶ 2/3  SHUT UP — let the world talk for a sec")
    print('   Hands off, mouth closed. Let the room breathe.')
    wait("   ↵ press Enter, then zip it… ")
    # In synthetic mode, the "room" is ambient only (no enrolled voice) so the
    # measured background is representative; live mode records the real room.
    room_scene = None if live else _AMBIENT_SCENE
    room_audio, rsr, rlevel, _ = cal.capture_audio(seconds, source, scene=room_scene)
    bg_cos = cal.cosines_to(room_audio, rsr, voiceprint)
    print(f"   ✓ learned your background  ({len(bg_cos)} ambient frames)")

    # --- Phase 3: anyone else (optional, live only) -----------------------
    if others and live:
        print("\n▶ 3/3  THE WORLD — let the chaos in")
        print('   Turn up the TV, let the horns honk, whatever isn\'t you.')
        wait("   ↵ press Enter when it's noisy… ")
        other_audio, osr, _, _ = cal.capture_audio(seconds, source)
        bg_cos += cal.cosines_to(other_audio, osr, voiceprint)
        print(f"   ✓ noted  ({len(bg_cos)} background frames total)")
    else:
        print("\n▶ 3/3  THE WORLD — skipped (room tone only)")

    # --- fit + save -------------------------------------------------------
    model, dprime = cal.fit(target_cos, bg_cos)
    path = cal.save(model)

    err = _dprime_error_pct(dprime)
    print("\n── what I learned ───────────────────────────────")
    print(f"   your voice cosine:  {model.target_mean:.2f} ± {model.target_std:.2f}")
    print(f"   the background:     {model.impostor_mean:.2f} ± {model.impostor_std:.2f}")
    print(f"   separation (d′):    {dprime}   → I can pick you out with ~{err}")
    print(f"   saved to {path}")
    print("   from here, I auto-tune to your room as we go.\n")

    return {"model": asdict(model), "dprime": dprime, "path": path,
            "stats": {"target_frames": len(target_cos), "bg_frames": len(bg_cos),
                      "synthetic": synth}}


def _dprime_error_pct(d: float) -> str:
    """Rough Gaussian-overlap error rate for a given d-prime, as readable text."""
    if d <= 0:
        return "no separation yet"
    # P(error) ~ Phi(-d/2) for equal-variance Gaussians; approximate Phi via erf.
    import math
    p = 0.5 * (1.0 - math.erf((d / 2.0) / math.sqrt(2.0)))
    if p < 0.0001:
        return "< 0.01% error"
    if p < 0.01:
        return f"{p * 100:.2f}% error"
    return f"{p * 100:.1f}% error"
