"""Known microphone-array geometries + detection.

The whole spatial stage is *geometry-agnostic* (patent Claim 5): it consumes a
coordinate map and adapts. We ship a few public, well-known geometries plus a
loader for arbitrary arrays via a user-provided coordinate map. We deliberately
do NOT ship the geometry of the patented hardware module — that lives in the
device's own coordinate map, which is exactly how the firmware does it.

Coordinates are in metres, (x, y, z), array centroid at the origin.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

__all__ = ["MicArray", "SINGLE", "MACBOOK_3", "RESPEAKER_4", "REGISTRY", "detect", "load_coordinate_map"]

Coord = Tuple[float, float, float]


@dataclass
class MicArray:
    name: str
    positions: List[Coord]
    notes: str = ""
    source: str = "builtin"

    @property
    def n_mics(self) -> int:
        return len(self.positions)

    @property
    def spatial_capable(self) -> bool:
        # Need at least 2 spatially-separated mics to estimate direction.
        return self.n_mics >= 2

    @property
    def planar(self) -> bool:
        return all(abs(z) < 1e-6 for (_, _, z) in self.positions)

    def summary(self) -> str:
        kind = "planar" if self.planar else "3D"
        cap = "spatial" if self.spatial_capable else "no spatial (single mic)"
        return f"{self.name}: {self.n_mics} mics, {kind}, {cap}"


# --- Single mic: the universal fallback. No direction, but WHO/WHAT/HOW still work.
SINGLE = MicArray(
    name="single",
    positions=[(0.0, 0.0, 0.0)],
    notes="One mic. No DOA/separation; speaker-ID, events and prosody work fully.",
)

# --- MacBook built-in array (approximate public geometry: mics along the top bezel).
# Spacing is approximate; SRP-PHAT only needs relative positions and adapts.
MACBOOK_3 = MicArray(
    name="macbook-3",
    positions=[(-0.13, 0.0, 0.0), (0.0, 0.005, 0.0), (0.13, 0.0, 0.0)],
    notes="MacBook built-in 3-mic beamforming array (approx. top-bezel layout).",
)

# --- ReSpeaker USB 4-mic array (Seeed): 35 mm square, public product geometry.
_R = 0.0247  # half-diagonal of a ~35mm square (radius to each mic from centre)
RESPEAKER_4 = MicArray(
    name="respeaker-4",
    positions=[(_R, _R, 0.0), (-_R, _R, 0.0), (-_R, -_R, 0.0), (_R, -_R, 0.0)],
    notes="Seeed ReSpeaker 4-mic array, ~35mm square.",
)

REGISTRY = {a.name: a for a in (SINGLE, MACBOOK_3, RESPEAKER_4)}


def load_coordinate_map(path: str) -> MicArray:
    """Load an arbitrary array geometry from a JSON coordinate map.

    Expected schema:
        {"name": "my-array", "positions": [[x,y,z], ...], "notes": "..."}

    This is the geometry-agnostic hook (patent Claim 5): any array — including the
    hardware module — plugs in by supplying its own coordinate map. No firmware or
    software change required.
    """
    data = json.loads(Path(path).read_text())
    positions = [tuple(float(v) for v in p) for p in data["positions"]]
    return MicArray(
        name=data.get("name", "custom"),
        positions=positions,
        notes=data.get("notes", "User-provided coordinate map."),
        source=str(path),
    )


def detect(prefer: Optional[str] = None, channels: Optional[int] = None) -> MicArray:
    """Best-guess array for the current machine.

    Resolution order:
      1. explicit `prefer` name from the REGISTRY
      2. a channel count (from the capture device) -> nearest known geometry
      3. query sounddevice for the default input's channel count (lazy import)
      4. SINGLE
    """
    if prefer and prefer in REGISTRY:
        return REGISTRY[prefer]

    if channels is None:
        try:
            import sounddevice as sd  # lazy: optional dependency

            dev = sd.query_devices(kind="input")
            channels = int(dev.get("max_input_channels", 1)) if isinstance(dev, dict) else 1
        except Exception:
            channels = None

    if channels is None:
        return SINGLE
    if channels >= 4:
        return RESPEAKER_4
    if channels >= 2:
        # 2-3 channels -> treat as a small built-in array (MacBook profile).
        return MACBOOK_3
    return SINGLE
