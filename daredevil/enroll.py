"""`python -m daredevil.enroll --name alan --seconds 3`

Thin wrapper so enrollment has its own module entry point (matches the spec).
Shows the confidence climbing with duration: 3s≈0.63, 10s≈0.96, 20s≈0.999.
"""
from __future__ import annotations

import argparse

from .pipeline import Pipeline
from .enrollment.manager import enrollment_confidence


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m daredevil.enroll")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--live", action="store_true", help="use the real microphone")
    args = ap.parse_args(argv)

    pipe = Pipeline()
    source = "live" if args.live else "synthetic"
    res = pipe.enroll(args.name, mic_seconds=args.seconds, source=source)
    print(f"enrolled '{res['name']}'  enrollment_confidence={res['enrollment_confidence']:.3f}")
    print(f"  duration={args.seconds:.0f}s   slot backend={res['backend']}   dim={res['dim']}")
    print(f"  C(3s)={enrollment_confidence(3):.2f}  C(10s)={enrollment_confidence(10):.2f}  "
          f"C(20s)={enrollment_confidence(20):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
