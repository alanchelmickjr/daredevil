"""`python -m daredevil.demo` — the one-command end-to-end demo.

Enrolls a speaker, listens to a scene, prints the structured awareness map an
LLM would consume, shows the parallel-vs-sequential timing, and renders the
on-screen radar. Runs everywhere: with no mic/models it uses a deterministic
synthetic scene (clearly labeled); `--live` uses the real microphone.
"""
from __future__ import annotations

import argparse
import json

from . import __version__
from .config import Config
from .pipeline import Pipeline
from .viz.spatial_map import render_ascii, render_matplotlib

BANNER = r"""
  ___                 _           _ _
 |   \ __ _ _ _ ___ __| |_____ __ (_) |
 | |) / _` | '_/ -_) _` / -_) V / | | |
 |___/\__,_|_| \___\__,_\___|\_/  |_|_|   acoustic awareness for LLMs
  WHO · WHERE · WHAT · HOW     on-device · no cloud · MIT
"""


def add_demo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--live", action="store_true", help="use the real microphone")
    parser.add_argument("--file", help="listen to a WAV file instead of the synthetic scene")
    parser.add_argument("--name", default="alan", help="speaker name to enroll (default: alan)")
    parser.add_argument("--enroll-seconds", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=1.0, help="listen window (s)")
    parser.add_argument("--simulate-latency", action="store_true",
                        help="inject representative model latencies (clearly labeled) to "
                             "illustrate the parallel speedup on a machine without the models")
    parser.add_argument("--fallback", action="store_true", help="force the pure-Python backend")
    parser.add_argument("--no-viz", action="store_true", help="skip the on-screen radar")
    parser.add_argument("--save-png", help="also save the awareness map radar as a PNG (needs matplotlib)")
    parser.add_argument("--spectrogram", help="render the spectrogram + awareness overlay to PNG (needs matplotlib)")
    parser.add_argument("--json", action="store_true", help="print only the awareness map JSON")


def run_demo(args: argparse.Namespace) -> int:
    from .stage1.mic_arrays import MACBOOK_3

    synthetic = not args.live and not args.file
    forced_fallback = getattr(args, "fallback", False)
    backend = "fallback" if forced_fallback else Config().resolved_backend()
    # In the synthetic demo with no ML backends, illustrate the parallel win with
    # clearly-labeled representative latencies. Real backends are measured live.
    simulate = getattr(args, "simulate_latency", False) or (synthetic and backend == "fallback")
    config = Config(simulate_latency=simulate)
    if forced_fallback:
        config.backend = "fallback"
    # The synthetic scene simulates a MacBook 3-mic array so WHERE is demonstrated.
    array = MACBOOK_3 if synthetic else None
    pipe = Pipeline(config=config, array=array)

    enroll_source = "live" if args.live else "synthetic"
    listen_source = "live" if args.live else ("file" if args.file else "synthetic")

    if args.json:
        e = pipe.enroll(args.name, mic_seconds=args.enroll_seconds, source=enroll_source)
        amap = pipe.listen(duration=args.duration, source=listen_source, file=args.file)
        print(json.dumps({"enrollment": e, "awareness_map": amap}, indent=2))
        return 0

    print(BANNER)
    dev = pipe.devices()
    backends_live = [k for k, v in dev["deps"].items() if v]
    print(f"  environment   backend={dev['backend']}   array={dev['array']}")
    print(f"  data dir      {dev['data_dir']}")
    print(f"  ml backends   {', '.join(backends_live) if backends_live else 'none installed → pure-Python fallback (full map still computed)'}")
    if synthetic:
        print("  scene         SYNTHETIC (simulating MacBook 3-mic array) — use --live for your mic\n")
    else:
        print()

    # 1) Enroll — WHO, first.
    print(f"▶ enrolling '{args.name}' ({args.enroll_seconds:.0f}s) ...")
    e = pipe.enroll(args.name, mic_seconds=args.enroll_seconds, source=enroll_source)
    print(f"  enrolled: {e['name']}  enrollment_confidence={e['enrollment_confidence']:.3f}  "
          f"(slot backend: {e['backend']}, {e['dim']}-dim)\n")

    # 2) Listen — the awareness map.
    print(f"▶ listening ({args.duration:.0f}s window) ...\n")
    _audio = _sr = None
    if args.spectrogram:
        amap, _audio, _sr = pipe.listen(duration=args.duration, source=listen_source,
                                        file=args.file, return_audio=True)
    else:
        amap = pipe.listen(duration=args.duration, source=listen_source, file=args.file)

    print("── AWARENESS MAP (this is what the LLM receives) " + "─" * 14)
    print(json.dumps(amap, indent=2))
    print()
    if amap["timing"].get("simulated"):
        print("  ⓘ timing is simulated (representative model latencies) — real backends are measured live\n")

    # The attention gate — a struct is built for every source, but only some reach
    # the conversational LLM. This is the point of the system.
    from .stage3.router import llm_payload
    sent = llm_payload(amap)
    print("  ATTENTION GATE → routed to the LLM: "
          + (", ".join(f"{s['id']} ({s['event']['class']})" for s in sent) or "nothing"))
    ambient = [s for s in amap["sources"] if s.get("attention") != "surface"]
    if ambient:
        print("  heard, tracked, gated OUT (ambient): "
              + ", ".join(f"{s['id']} ({s['event']['class']})" for s in ambient))
    print()

    if not args.no_viz:
        print(render_ascii(amap))
        if args.save_png:
            try:
                render_matplotlib(amap, args.save_png)
                print(f"\n  saved radar → {args.save_png}")
            except Exception as ex:
                print(f"\n  (matplotlib unavailable: {ex})")

    print("\n  \"Everything you just saw runs on a laptop's built-in mics — no cloud,")
    print("   no GPU required. The hardware module adds 3D + ultrasonic and runs it")
    print("   on-device. Software proves the architecture; hardware proves it scales.\"")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m daredevil.demo",
                                     description="Daredevil end-to-end demo")
    parser.add_argument("--version", action="version", version=f"daredevil {__version__}")
    add_demo_args(parser)
    return run_demo(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
