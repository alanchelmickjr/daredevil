"""`daredevil onboard` — the fun, anywhere first run.

You talk to it; it learns your voice and your name; then it plays a room full of
*other* people through the speakers and picks **you** out of the crowd — by name.
That's the whole pitch in ninety seconds, on any laptop.

Runs everywhere. With a mic it's live: real enrollment, real crowd out the
speakers, real recognition. Without one it runs the same arc on a labeled
SYNTHETIC scene so the idea still lands — and the recognition is still *real*
(cosine SPRT on the voiceprint), never faked. Every number printed is measured.
"""
from __future__ import annotations

import time
from typing import Optional


def _say(line: str = "", pause: float = 0.0) -> None:
    """Radar speaking. (Text for now; a local TTS voice is a natural next step.)"""
    print(line)
    if pause:
        time.sleep(pause)


def _rule() -> None:
    print("─" * 60)


def _summarize(amap: dict, name: str):
    you = next((s for s in amap.get("sources", []) if s.get("id") == name), None)
    crowd = [s for s in amap.get("sources", []) if s.get("id") != name]
    gated = [s for s in crowd if s.get("attention") != "surface"]
    return you, crowd, gated


def onboard(name: Optional[str] = None, live: bool = False, crowd: int = 4,
            windows: int = 5, seconds: float = 8.0) -> int:
    from .config import Config
    from .pipeline import Pipeline
    from .stage1.mic_arrays import MACBOOK_3
    from .audio.crowd import crowd_scene_sources, CrowdPlayer

    name = (name or "").strip() or "you"

    # Decide live vs synthetic honestly by actually probing the mic.
    if live:
        try:
            from .audio.capture import capture_live
            capture_live(0.2)
        except Exception:
            live = False
            print("(no working microphone — running the SYNTHETIC walkthrough so you\n"
                  " still see the whole arc; `pip install -e \".[audio]\"` for the live one)\n")

    source = "live" if live else "synthetic"
    pipe = Pipeline(config=Config(), array=(None if live else MACBOOK_3))
    radar_name = pipe.config.wake_word

    print()
    _rule()
    _say(f"  👋  Hey — I'm Radar.")
    if not live:
        _say("      (SYNTHETIC walkthrough — same flow, generated voices, real recognition)")
    _say(f"      Give me a few seconds of your voice and I'll find you in any crowd.")
    _rule()
    print()

    # ── 1) learn your voice ────────────────────────────────────────────────
    _say("  ①  Learning your voice.")
    if live:
        _say(f"      Talk to me for ~{int(seconds)}s — anything. Tell me about your day,")
        _say("      read this line twice, whatever. Just keep talking.", pause=0.4)
    enr = pipe.enroll(name, mic_seconds=seconds, source=source)
    _say(f"      ✓ got you, {name}.  enrollment confidence "
         f"{enr['enrollment_confidence']:.0%}  ({enr['backend']}, {enr['dim']}-dim voiceprint)")
    if enr["backend"] == "fallback":
        _say("        (heuristic voiceprint — `pip install -e \".[speaker]\"` swaps in real ECAPA)")
    print()

    # ── 2) learn your name (wake word) ─────────────────────────────────────
    _say(f"  ②  Teaching me my name, so I come when you call: \"{radar_name}\".")
    if live:
        _say(f"      Say \"{radar_name}\" a couple of times.", pause=0.3)
    wk = pipe.enroll_wake(mic_seconds=2.5, source=source)
    if wk.get("ok"):
        _say(f"      ✓ learned \"{wk['phrase']}\" — now I wake when you call me by name.")
    else:
        _say(f"      (couldn't learn the wake phrase this round: {wk.get('reason')})")
    print()

    # ── 3) the reveal: pick me out of the crowd ────────────────────────────
    _say("  ③  Now the fun part — find YOU in a crowd.")
    pipe.warmup()
    hits, best_conf = 0, 0.0

    if live:
        player = CrowdPlayer(n_speakers=crowd, sr=pipe.config.inference_sr).start()
        if player.available:
            _say(f"      🔊 I'm playing {crowd} other voices out your speakers. Keep talking —")
            _say("         I'll light you up against them.", pause=0.4)
        else:
            _say("      (no speaker output for the crowd — talk near a TV or with a friend\n"
                 "       and watch me hold onto you anyway)")
        try:
            for w in range(windows):
                amap = pipe.listen(duration=1.5, source="live")
                hits, best_conf = _report_window(amap, name, w, windows, hits, best_conf)
        finally:
            player.stop()
    else:
        scene = [{"name": name, "enrolled": True, "class": "speech", "azimuth": 0.0,
                  "elevation": 0.0, "prosody_state": "calm", "distress": 0.1}]
        scene += crowd_scene_sources(n_speakers=crowd)
        _say(f"      🔊 {crowd} synthetic voices around you (SYNTHETIC). Watch me hold onto you.")
        print()
        for w in range(windows):
            amap = pipe.listen(duration=1.0, source="synthetic", scene=scene)
            hits, best_conf = _report_window(amap, name, w, windows, hits, best_conf)

    # ── wrap ───────────────────────────────────────────────────────────────
    print()
    _rule()
    if hits:
        _say(f"  🎯  Picked you out in {hits}/{windows} windows"
             f"{f' (best confidence {best_conf:.0%})' if best_conf else ''}.")
        _say(f"      The crowd was heard and tracked — and deliberately kept out of our")
        _say(f"      conversation. That gate is the whole point: I pay attention to you.")
    else:
        _say(f"  …  Didn't lock you in this run.")
        if live:
            _say("      Talk a little longer/closer, or lower the crowd volume — the SPRT")
            _say("      needs a few clear frames. `daredevil onboard --live -s 12` helps.")
    _rule()
    print()
    _say("  Try me in the wild:")
    _say("   • in a café, or with the TV on   • with a friend talking over you")
    _say("   • walk away mid-sentence (sound doesn't need line of sight)")
    if not live:
        _say("   • for real:  pip install -e \".[speaker,audio]\"  &&  daredevil onboard --live")
    _say(f"  Then open the live HUD:  daredevil serve --live")
    print()
    return 0


def _report_window(amap: dict, name: str, w: int, windows: int,
                   hits: int, best_conf: float):
    you, crowd, gated = _summarize(amap, name)
    if you and you.get("attention") == "surface":
        conf = (you.get("identity") or {}).get("confidence", 0.0)
        best_conf = max(best_conf, conf)
        hits += 1
        woke = " 🗣 (you called my name)" if you.get("addressed") else ""
        _say(f"      window {w+1}/{windows}:  ✓ that's {name}"
             f"{f' (conf {conf:.0%})' if conf else ''} — {len(gated)} other voice(s) gated out{woke}")
    else:
        ident = "identifying…" if (you and you.get("identifying")) else "not yet"
        _say(f"      window {w+1}/{windows}:  … {ident}  ({len(crowd)} voice(s) in the room)")
    return hits, best_conf
