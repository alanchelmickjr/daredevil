"""`daredevil` command-line interface.

  daredevil demo [--live|--file ...]   end-to-end demo
  daredevil enroll --name NAME [-s N]  enroll a speaker (3s min)
  daredevil listen [--live|--file ...] one awareness map -> stdout
  daredevil devices                    what was detected / installed
  daredevil version
"""
from __future__ import annotations

import argparse
import json

from . import __version__
from .config import Config
from .pipeline import Pipeline
from .enrollment.manager import enrollment_confidence
from .demo import add_demo_args, run_demo
from .viz.spatial_map import render_ascii


def _cmd_enroll(args) -> int:
    pipe = Pipeline()
    source = "live" if args.live else "synthetic"
    res = pipe.enroll(args.name, mic_seconds=args.seconds, source=source)
    print(f"enrolled '{res['name']}'  confidence={res['enrollment_confidence']:.3f}  "
          f"backend={res['backend']}  dim={res['dim']}")
    print("  confidence curve C(t)=1-exp(-t/3): "
          f"3s={enrollment_confidence(3):.2f}  10s={enrollment_confidence(10):.2f}  "
          f"20s={enrollment_confidence(20):.3f}")
    return 0


def _cmd_listen(args) -> int:
    from .stage1.mic_arrays import MACBOOK_3
    synthetic = not args.live and not args.file
    simulate = args.simulate_latency or (synthetic and Config().resolved_backend() == "fallback")
    config = Config(simulate_latency=simulate)
    pipe = Pipeline(config=config, array=(MACBOOK_3 if synthetic else None))
    source = "live" if args.live else ("file" if args.file else "synthetic")
    amap = pipe.listen(duration=args.duration, source=source, file=args.file)
    if args.json:
        print(json.dumps(amap, indent=2))
    else:
        print(render_ascii(amap))
    return 0


def _cmd_devices(args) -> int:
    print(json.dumps(Pipeline().devices(), indent=2))
    return 0


def _crowd(n: int) -> list:
    """A synthetic 'crowd': the owner + a baby cry + (n-2) ambient chatter sources."""
    scene = [
        {"name": "alan", "enrolled": True, "class": "speech", "azimuth": 0.0,
         "elevation": 0.0, "prosody_state": "calm", "distress": 0.12},
        {"name": None, "enrolled": False, "class": "baby_cry", "azimuth": 45.0,
         "elevation": 10.0, "prosody_state": "distressed", "distress": 0.95},
    ]
    for i in range(max(0, n - 2)):
        scene.append({"name": f"chatter{i}", "enrolled": False, "class": "music",
                      "azimuth": float((90 + i * 37) % 360), "elevation": 0.0,
                      "prosody_state": "calm", "distress": 0.05})
    return scene


def _cmd_bench(args) -> int:
    """Show that response time is driven by the attention gate, not the crowd size."""
    import statistics
    from .stage1.mic_arrays import MACBOOK_3
    pipe = Pipeline(config=Config(), array=MACBOOK_3)
    pipe.enroll("alan", 3, source="synthetic")
    pipe.warmup()
    print(f"backend={pipe.config.resolved_backend()}   iters={args.iters}")
    print(f"  {'sources':>7} | {'pipeline_ms':>11} | {'→ LLM':>6}")
    print("  " + "-" * 32)
    for n in (2, 5, 10, 20):
        scene = _crowd(n)
        times, routed = [], 0
        for _ in range(args.iters):
            amap = pipe.listen(duration=1.0, source="synthetic", scene=scene)
            times.append(amap["timing"]["parallel_ms"])
            routed = len(amap["routed_to_llm"])
        med = statistics.median(times)
        print(f"  {n:>7} | {med:>9.1f}{'✓' if med < 200 else ' '} | {routed:>6}")
    print("  → pipeline scales gently + locally with sources; the LLM payload (→ LLM)")
    print("    stays flat. The attention gate — not the room — drives response time.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="daredevil",
                                description="Local, private acoustic context for LLMs.")
    p.add_argument("--version", action="version", version=f"daredevil {__version__}")
    sub = p.add_subparsers(dest="cmd")

    pd = sub.add_parser("demo", help="end-to-end demo")
    add_demo_args(pd)

    pe = sub.add_parser("enroll", help="enroll a speaker")
    pe.add_argument("--name", required=True)
    pe.add_argument("-s", "--seconds", type=float, default=3.0)
    pe.add_argument("--live", action="store_true")

    pl = sub.add_parser("listen", help="emit one awareness map")
    pl.add_argument("--duration", type=float, default=1.0)
    pl.add_argument("--live", action="store_true")
    pl.add_argument("--file")
    pl.add_argument("--simulate-latency", action="store_true")
    pl.add_argument("--json", action="store_true")

    psv = sub.add_parser("serve", help="run the local web HUD (neumorphic-steampunk)")
    psv.add_argument("--port", type=int, default=8770)
    psv.add_argument("--live", action="store_true")

    pb = sub.add_parser("bench", help="measure pipeline latency vs crowd size")
    pb.add_argument("--iters", type=int, default=10)

    sub.add_parser("devices", help="show detected array + installed backends")
    sub.add_parser("version", help="print version")

    args = p.parse_args(argv)

    if args.cmd in (None, "demo"):
        if args.cmd is None:
            args = p.parse_args(["demo"])
        return run_demo(args)
    if args.cmd == "enroll":
        return _cmd_enroll(args)
    if args.cmd == "listen":
        return _cmd_listen(args)
    if args.cmd == "serve":
        from .viz.server import serve
        return serve(port=args.port, live=args.live)
    if args.cmd == "bench":
        return _cmd_bench(args)
    if args.cmd == "devices":
        return _cmd_devices(args)
    if args.cmd == "version":
        print(f"daredevil {__version__}")
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
