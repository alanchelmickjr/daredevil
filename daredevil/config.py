"""Central configuration: thresholds, weights, safety classes, backend detection.

Everything tunable in one place. Pure stdlib — importing this never pulls a heavy
dependency. The defaults below are taken directly from the patent's algorithmic
formulations (cosine match T=0.70, enrollment time constant ~3s, the weighted
priority sum) so the software and the firmware speak the same language.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

__all__ = [
    "Config",
    "PriorityWeights",
    "Thresholds",
    "SAFETY_CRITICAL_CLASSES",
    "is_safety_critical",
    "detect_backend",
    "default_data_dir",
]

# AudioSet-style labels that trigger a hard priority override in Stage 3.
# Normalised to lowercase, underscores collapsed; matching is substring-aware
# (see is_safety_critical) so "baby cry", "baby_cry", "infant crying" all hit.
SAFETY_CRITICAL_CLASSES = {
    "baby_cry",
    "infant_cry",
    "crying_sobbing",
    "screaming",
    "scream",
    "smoke_detector",
    "fire_alarm",
    "alarm",
    "siren",
    "civil_defense_siren",
    "gunshot",
    "gunfire",
    "explosion",
    "glass",
    "breaking",
    "shatter",
    "car_alarm",
}


def is_safety_critical(label: str) -> bool:
    """True if an event label maps to a safety-critical class.

    Substring-aware so classifier label variants ("Baby cry, infant cry") still
    resolve to the canonical safety set.
    """
    if not label:
        return False
    norm = label.strip().lower().replace(", ", " ").replace(",", " ").replace("-", "_")
    norm = norm.replace(" ", "_")
    for token in SAFETY_CRITICAL_CLASSES:
        if token in norm:
            return True
    return False


@dataclass
class PriorityWeights:
    """Weights for the composite priority score (patent Eq. 2).

    P = w_id*S_identity + w_event*S_event + w_prosody*S_prosody + w_temporal*S_temporal
    Defaults sum to 1.0 so the raw score lands in [0, 1] before overrides.
    """

    identity: float = 0.35
    event: float = 0.30
    prosody: float = 0.20
    temporal: float = 0.15

    def normalized(self) -> "PriorityWeights":
        total = self.identity + self.event + self.prosody + self.temporal
        if total <= 0:
            return PriorityWeights()
        return PriorityWeights(
            identity=self.identity / total,
            event=self.event / total,
            prosody=self.prosody / total,
            temporal=self.temporal / total,
        )


@dataclass
class Thresholds:
    """Decision thresholds. Names mirror the patent symbols."""

    match: float = 0.70            # T — cosine similarity to declare an identity match
    safety: float = 0.50           # T_safety — event confidence that triggers override
    distress: float = 0.60         # T_distress — prosodic distress that escalates priority
    unknown_track: float = 0.65    # cosine to consider two unknown frames the same source
    vad: float = 0.012             # energy gate for "is anyone speaking"
    enroll_tau: float = 3.0        # tau_enroll — enrollment confidence time constant (s)
    surface: float = 0.45          # priority at/above which a source is routed to the LLM


def default_data_dir() -> Path:
    """Where enrolled voiceprints live by default.

    Never inside the installed package. Honours $DAREDEVIL_HOME, else ~/.daredevil.
    """
    env = os.environ.get("DAREDEVIL_HOME")
    base = Path(env) if env else Path.home() / ".daredevil"
    return base


def detect_backend() -> str:
    """Best available inference backend, without importing heavy libs eagerly.

    Returns one of: "cuda", "mps", "cpu", "fallback".
    "fallback" means no torch/onnx — the pure-Python path still produces a full
    awareness map, just with heuristic slots.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        return "fallback"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "fallback"


@dataclass
class Config:
    """Top-level pipeline configuration.

    `allow_cloud` is False and stays False: Daredevil performs all inference
    locally. There is no code path that sends audio or embeddings to a remote
    service. The flag exists only to make that guarantee explicit and auditable.
    """

    # audio
    capture_rate: int = 48000      # native laptop/UAC2 rate
    inference_sr: int = 16000      # rate the models were trained at
    window_seconds: float = 1.0    # analysis window fed to every slot

    # slots
    enabled_slots: tuple = ("embedding", "events", "prosody")
    backend: str = "auto"          # "auto" -> detect_backend()

    # behaviour
    allow_cloud: bool = False      # hard guarantee: no network egress for inference
    simulate_latency: bool = False  # illustrate parallel-vs-sequential on bare machines

    # tuning
    weights: PriorityWeights = field(default_factory=PriorityWeights)
    thresholds: Thresholds = field(default_factory=Thresholds)

    # storage / fleet
    data_dir: Optional[str] = None     # None -> default_data_dir()
    fleet_backend: str = "local"       # "local" (default) | "gun"
    gun_peers: tuple = ()              # e.g. ("http://127.0.0.1:8765/gun",)

    # representative per-slot CPU latencies (ms) used only when simulate_latency=True.
    # These are illustrative model timings, NOT measured live numbers, and the demo
    # labels them as simulated.
    simulated_slot_ms: dict = field(
        default_factory=lambda: {"embedding": 95.0, "events": 110.0, "prosody": 40.0}
    )

    def resolved_backend(self) -> str:
        return detect_backend() if self.backend == "auto" else self.backend

    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir) if self.data_dir else default_data_dir()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["resolved_backend"] = self.resolved_backend()
        d["resolved_data_dir"] = str(self.resolved_data_dir())
        return d
