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
        idtxt = f" id={ident['confidence']:.2f}" if ident else ""
        gate = "→LLM" if s.get("attention") == "surface" else "····"
        lines.append(f"  {flag}[{s['priority']:.2f}] {bar} {gate} {s['id']:<12} "
                     f"{ev.get('class','?'):<9} {where:<15} {pr.get('state','?'):<10}{idtxt}")
        if s.get("priority_override"):
            lines.append(f"           └─ OVERRIDE: {s['priority_override']}")
    routed = amap.get("routed_to_llm", [])
    lines.append(f"  attention gate → LLM: {', '.join(routed) if routed else 'nothing'}"
                 "   (others heard, gated out)")
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


def render_spectrogram(amap: dict, audio, sr: int, path: Optional[str] = None):
    """Two-panel view: the raw (noisy) spectrogram with awareness tags overlaid
    (WHAT/HOW), beside a polar WHERE radar. The 'chaos in -> structure out' shot.

    `audio` is held transiently for visualization only — it is never persisted.
    """
    try:
        import math
        import numpy as np
        import matplotlib
        if path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("spectrogram needs matplotlib+numpy (pip install daredevil[viz])") from e

    y = np.asarray(audio, dtype=float)
    fig = plt.figure(figsize=(13, 5.5), facecolor="#0d0d12")
    fig.suptitle("Daredevil  ·  chaos in  →  structure out", color="white", fontsize=14, weight="bold")

    # LEFT — the real sound itself (spectrogram) + source tags painted on top
    ax = fig.add_subplot(1, 2, 1, facecolor="#0d0d12")
    if y.size:
        ax.specgram(y, NFFT=1024, Fs=sr, noverlap=512, cmap="magma")
    ax.set_title("WHAT / HOW  ·  real sound in", color="white")
    ax.set_xlabel("time (s)", color="white")
    ax.set_ylabel("frequency (Hz)", color="white")
    ax.tick_params(colors="white")
    for i, s in enumerate(amap.get("sources", [])):
        ev = s.get("event", {})
        col = "#ff5a5a" if ev.get("safety_critical") else "#5ad1ff"
        pr = s.get("prosody", {})
        flag = "⚠ " if ev.get("safety_critical") else ""
        label = f"{flag}{s['id']} · {ev.get('class','')} · {pr.get('state','')} · p={s['priority']:.2f}"
        ax.annotate(label, xy=(0.02, 0.95 - i * 0.075), xycoords="axes fraction",
                    color=col, fontsize=9, weight="bold",
                    bbox=dict(boxstyle="round", fc="black", ec=col, alpha=0.65))

    # RIGHT — where each source is (polar radar)
    axp = fig.add_subplot(1, 2, 2, projection="polar", facecolor="#15151c")
    axp.set_theta_zero_location("N")
    axp.set_theta_direction(-1)
    axp.set_title("WHERE  ·  structure out", color="white")
    for s in amap.get("sources", []):
        pos = s.get("position")
        if not pos:
            continue
        theta = math.radians(pos["azimuth"])
        size = 140 + 520 * s["priority"]
        col = "#ff5a5a" if s.get("event", {}).get("safety_critical") else "#5ad1ff"
        axp.scatter([theta], [1.0], s=size, c=col, alpha=0.85, edgecolors="white", linewidths=1.5)
        axp.annotate(s["id"], (theta, 1.0), textcoords="offset points", xytext=(7, 7),
                     fontsize=8, color="white")
    axp.set_ylim(0, 1.25)
    axp.set_rticks([])
    axp.tick_params(colors="white")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
    plt.show()
    return None


def _mini_wave(klass: str, n: int = 80):
    import math
    f = {"speech": 3, "baby_cry": 7, "alarm": 9, "siren": 8, "music": 5}.get(klass, 4)
    return [math.sin(2 * math.pi * f * i / n) * (0.55 + 0.45 * math.sin(2 * math.pi * 0.5 * i / n))
            for i in range(n)]


def render_radar_hud(amap: dict, path: Optional[str] = None, waves: Optional[dict] = None):
    """The 'sees with sound' HUD: device in the center, each source orbiting at its
    azimuth in a box with a mini-waveform + WHO/WHAT/HOW. A static preview of the
    live web HUD (WiFi-client / spy-HUD aesthetic). Waveforms are illustrative
    unless `waves` (id -> samples) is supplied.
    """
    try:
        import math
        import numpy as np
        import matplotlib
        if path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, Circle
    except Exception as e:  # pragma: no cover
        raise RuntimeError("HUD needs matplotlib+numpy (pip install daredevil[viz])") from e

    BG, FG, CY, RED, GRN, DIM = "#0a0e14", "#cfe9ff", "#42d6ff", "#ff5470", "#48f08b", "#22303c"
    fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Daredevil  ·  WHO · WHERE · WHAT · HOW", color=FG, fontsize=14, weight="bold", pad=12)

    for r in (0.35, 0.7, 1.05):
        ax.add_patch(Circle((0, 0), r, fill=False, ec=DIM, lw=1, zorder=0))
    for ang, lbl in [(0, "front"), (90, "right"), (180, "back"), (270, "left")]:
        a = math.radians(ang)
        ax.text(1.18 * math.sin(a), 1.18 * math.cos(a), lbl, color="#5b6b78",
                ha="center", va="center", fontsize=8)

    # center device node (the "router")
    ax.add_patch(FancyBboxPatch((-0.17, -0.1), 0.34, 0.2,
                 boxstyle="round,pad=0.02,rounding_size=0.04", fc="#10202b", ec=CY, lw=1.6, zorder=5))
    ax.text(0, 0.035, "MacBook", color=FG, ha="center", va="center", fontsize=11, weight="bold", zorder=6)
    ax.text(0, -0.05, "Daredevil core", color=CY, ha="center", va="center", fontsize=8, zorder=6)

    for s in amap.get("sources", []):
        pos = s.get("position")
        a = math.radians(pos["azimuth"]) if pos else math.radians(20)
        pr = s["priority"]
        r = 0.55 + 0.42 * (1 - pr)           # higher priority orbits closer in
        x, y = r * math.sin(a), r * math.cos(a)
        ev = s.get("event", {})
        safety = ev.get("safety_critical")
        ambient = s.get("attention") != "surface"   # heard + tracked, but gated out
        col = "#5b6b78" if ambient else (RED if safety else (GRN if s["type"] == "enrolled" else CY))

        ax.plot([0, x], [0, y], color=col, lw=1, alpha=(0.2 if ambient else 0.6),
                ls=("--" if ambient else "-"), zorder=1)
        bw, bh = 0.5, 0.32
        if safety and not ambient:
            ax.add_patch(FancyBboxPatch((x - bw / 2 - 0.025, y - bh / 2 - 0.025), bw + 0.05, bh + 0.05,
                         boxstyle="round,pad=0.02,rounding_size=0.05", fc="none", ec=RED, lw=2, alpha=0.4, zorder=2))
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh,
                     boxstyle="round,pad=0.02,rounding_size=0.05", fc="#0e1922", ec=col, lw=1.6,
                     alpha=(0.45 if ambient else 1.0), zorder=3))

        flag = "⚠ " if safety else ""
        ax.text(x, y + bh / 2 - 0.05, f"{flag}{s['id']}", color=col, ha="center", va="center",
                fontsize=9.5, weight="bold", zorder=4)
        w = (waves or {}).get(s["id"]) or _mini_wave(ev.get("class", ""))
        xs = np.linspace(x - bw / 2 + 0.05, x + bw / 2 - 0.05, len(w))
        ys = y + 0.012 + 0.05 * np.asarray(w, dtype=float)
        ax.plot(xs, ys, color=col, lw=0.9, alpha=(0.5 if ambient else 0.95), zorder=4)
        foot = f"{ev.get('class','')} · {s.get('prosody',{}).get('state','')} · p={pr:.2f}"
        ax.text(x, y - bh / 2 + 0.06, foot, color=(col if ambient else FG), ha="center",
                va="center", fontsize=7.5, zorder=4)
        tag = s.get("priority_override") or ("gated · ambient" if ambient else "→ LLM")
        tagcol = RED if s.get("priority_override") else ("#5b6b78" if ambient else GRN)
        ax.text(x, y - bh / 2 + 0.012, tag, color=tagcol, ha="center", va="center",
                fontsize=6.5, weight="bold", zorder=4)

    ax.text(0, -1.24, "seeing with sound  ·  on-device · no cloud · embeddings non-reversible",
            color="#5b6b78", ha="center", fontsize=9)
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=BG)
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
