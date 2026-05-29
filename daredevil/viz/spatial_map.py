"""Awareness-map visualization.

`render_ascii` always works (terminal radar + priority bars). `render_matplotlib`
draws a polar plot when matplotlib is installed. The ASCII map is intentionally
nice enough to be the demo's on-screen output anywhere.
"""
from __future__ import annotations

from typing import Optional

_OCTANTS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _octant(azimuth: float) -> str:
    idx = int(((azimuth % 360) + 22.5) // 45) % 8
    return _OCTANTS[idx]


_RULE = "═" * 60


def render_ascii(amap: dict) -> str:
    """A clean, alignment-robust terminal render (no fragile right border)."""
    arr = amap.get("array", {})
    spatial = "spatial" if arr.get("spatial") else "no spatial (single mic)"
    lines = [_RULE, "  ACOUSTIC AWARENESS MAP   — what the LLM receives"]
    lines.append(f"  array: {arr.get('name','?')} ({arr.get('n_mics','?')} mics, {spatial})"
                 f"   backend: {amap.get('backend','?')}")
    lines.append("  sources (high → low priority):")
    for s in amap.get("sources", []):
        bar = ("█" * int(round(s["priority"] * 10)) + "·" * 10)[:10]
        ev = s.get("event", {})
        flag = "⚠ " if ev.get("safety_critical") else "  "
        pos = s.get("position")
        where = f"az {int(pos['azimuth']):>4}° {_octant(pos['azimuth'])}" if pos else "no direction"
        pr = s.get("prosody", {})
        ident = s.get("identity")
        idtxt = f"  id={ident['confidence']:.2f}" if ident else ""
        lines.append(f"  {flag}[{s['priority']:.2f}] {bar} {s['id']:<12} "
                     f"{ev.get('class','?'):<10} {where:<16} {pr.get('state','?'):<11}{idtxt}")
        if s.get("priority_override"):
            lines.append(f"           └─ OVERRIDE: {s['priority_override']}")
    t = amap.get("timing", {})
    sim = " (simulated)" if t.get("simulated") else ""
    speed = ""
    try:
        if t.get("parallel_ms"):
            speed = f"   →  {t['sequential_ms'] / max(t['parallel_ms'], 0.001):.1f}× faster"
    except Exception:
        speed = ""
    lines.append("  " + "─" * 58)
    lines.append(f"  timing{sim}: parallel {t.get('parallel_ms','?')}ms  vs  "
                 f"sequential {t.get('sequential_ms','?')}ms{speed}")
    lines.append("  privacy: on-device · no cloud · embeddings non-reversible")
    lines.append(_RULE)
    return "\n".join(lines)


def render_matplotlib(amap: dict, path: Optional[str] = None):
    try:
        import math
        import matplotlib
        if path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib not installed (pip install daredevil[viz])") from e

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    for s in amap.get("sources", []):
        pos = s.get("position")
        theta = math.radians(pos["azimuth"]) if pos else 0.0
        r = 1.0
        size = 80 + 320 * s["priority"]
        color = "red" if s.get("event", {}).get("safety_critical") else "tab:blue"
        ax.scatter([theta], [r], s=size, c=color, alpha=0.7, edgecolors="k")
        ax.annotate(f"{s['id']}\n{s.get('event',{}).get('class','')}",
                    (theta, r), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_title("Daredevil — Acoustic Awareness Map")
    ax.set_rticks([])
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path
    plt.show()
    return None


def render(amap: dict, mode: str = "auto", path: Optional[str] = None):
    if mode in ("matplotlib", "mpl") or (mode == "auto" and path):
        return render_matplotlib(amap, path)
    out = render_ascii(amap)
    print(out)
    return out
