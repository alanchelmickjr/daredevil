"""Central configuration: thresholds, weights, safety classes, backend detection.

Everything tunable in one place. Pure stdlib — importing this never pulls a heavy
dependency. The defaults below are taken directly from the patent's algorithmic
formulations (cosine match T=0.70, enrollment time constant ~3s, the weighted
priority sum) so the software and the firmware speak the same language.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

__all__ = [
    "Config",
    "PriorityWeights",
    "Thresholds",
    "IdentityModel",
    "TrackerParams",
    "SeparationParams",
    "NMFParams",
    "SAFETY_CRITICAL_CLASSES",
    "is_safety_critical",
    "detect_backend",
    "default_data_dir",
    "load_calibration",
    "calibration_path",
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
    vad: float = 0.004             # energy gate for "is anyone speaking"
    enroll_tau: float = 3.0        # tau_enroll — enrollment confidence time constant (s)
    surface: float = 0.45          # priority at/above which a source is routed to the LLM


@dataclass
class IdentityModel:
    """Score models for the SPRT identity test — the sonar/radar detection paradigm.

    Per-frame cosine similarity ``s`` between an observed embedding and an enrolled
    voiceprint is modelled as Gaussian under each hypothesis:

        H1 (target speaker):  s ~ N(target_mean, target_std)
        H0 (background/impostor): s ~ N(impostor_mean, impostor_std)

    The per-frame log-likelihood ratio is accumulated (Wald's Sequential
    Probability Ratio Test) until it crosses an upper bound A = log((1-beta)/alpha)
    — accept the identity — or a lower bound B = log(beta/(1-alpha)) — reject it.
    Because the cumulative LLR is a log-odds, the reported confidence is simply
    sigmoid(LLR): a real posterior, not a tuned curve. H0 is a *model*, so a single
    enrolled speaker is matchable (no impostor cohort required). With two or more
    enrolled speakers the others form an adaptive cohort (AS-Norm) for H0.

    Defaults reflect published ECAPA-TDNN/VoxCeleb cosine statistics.
    """

    target_mean: float = 0.65      # mean same-speaker cosine
    target_std: float = 0.12
    impostor_mean: float = 0.18    # mean different-speaker cosine
    impostor_std: float = 0.11
    alpha: float = 0.01            # tolerable false-accept rate -> upper bound A
    beta: float = 0.05             # tolerable miss rate -> lower bound B
    leak: float = 0.0              # per-frame LLR leak for non-stationarity (0 = pure SPRT)
    immediate_cosine: float = 0.80 # a single frame this strong matches outright
    quality_full_energy: float = 0.02  # energy at which a frame carries full evidential weight

    # --- real-time auto-calibration (CFAR: hold the false-alarm rate constant as
    # the room changes). The human onboarding session seeds the first model; these
    # keep H0 — and optionally the voiceprint — tuned as we go.
    adapt_background: bool = True       # online-estimate H0 from live ambient audio
    bg_adapt_rate: float = 0.02         # EMA rate for the running background model
    bg_guard_sigmas: float = 1.5        # only frames within this band of H0 feed it (CFAR guard cells)
    adapt_target: bool = False          # online-refine the voiceprint when very confidently matched
    target_adapt_rate: float = 0.05     # EMA rate for adaptive enrollment (guarded)
    target_adapt_margin: float = 2.0    # LLR must clear bound A by this margin before refining


@dataclass
class TrackerParams:
    """Multi-target track manager parameters (data association + track lifecycle).

    Mirrors a passive-sonar tracker: detections are gated and associated to
    existing tracks, tracks are confirmed by M-of-N logic, coast through misses,
    and are deleted after sustained silence. Bearing (azimuth from DOA) is smoothed
    with an alpha-beta filter.
    """

    assoc_cosine: float = 0.55     # min embedding cosine to associate a detection to a track
    bearing_gate_deg: float = 25.0 # max azimuth gap to associate when both have bearing
    bearing_assist: float = 0.15   # embedding slack permitted when bearing agrees
    confirm_hits: int = 2          # M-of-N: hits to promote TENTATIVE -> CONFIRMED
    confirm_window_s: float = 3.0  # window over which confirm_hits must accrue
    coast_s: float = 5.0           # keep a missed track COASTING this long
    delete_s: float = 15.0         # delete a track after this much silence
    bearing_alpha: float = 0.5     # alpha-beta position-filter gains
    bearing_beta: float = 0.1
    recency_bonus: float = 0.10    # association tie-break bonus for very recent tracks
    recency_window_s: float = 3.0
    embedding_ema: float = 0.30    # how fast a track's stored voiceprint adapts


@dataclass
class SeparationParams:
    """When to actually split a mixed stream into multiple contacts.

    Separation only helps when there really are concurrent sources; on a single
    talker it manufactures a low-energy artifact stream. We keep a separated stream
    only if it is both energetic *and* spectrally distinct from the dominant one —
    otherwise the source is treated as single and WHO runs on the clean wideband
    audio rather than the band-limited separation output.
    """

    min_energy: float = 0.003      # below this RMS a separated stream is silence
    dominance_ratio: float = 0.35  # a 2nd stream must reach this fraction of the primary's energy
    distinct_cosine: float = 0.60  # streams more similar than this are the same source
    energy_ratio_cap: float = 0.80 # if streams are this similar in energy, it's one source split in half


@dataclass
class NMFParams:
    """Spectral-basis tracking features ("name that tune in 3 notes").

    The tracker associates contacts on frame-stable NMF activations rather than
    ECAPA embeddings (which are unstable frame to frame). numpy learns the basis;
    with no numpy the feature degrades to the normalised spectral envelope.
    See docs/NMF_TRACKING_DESIGN.md.
    """

    enabled: bool = True
    n_bins: int = 48              # frequency bands in the magnitude spectrum
    n_components: int = 6         # learned spectral bases (k)
    fit_after: int = 24           # observed frames before a global basis is learned
    learn_iters: int = 150        # NMF iterations when learning the basis
    decompose_iters: int = 50     # NMF iterations when decomposing a frame


def default_data_dir() -> Path:
    """Where enrolled voiceprints live by default.

    Never inside the installed package. Honours $DAREDEVIL_HOME, else ~/.daredevil.
    """
    env = os.environ.get("DAREDEVIL_HOME")
    base = Path(env) if env else Path.home() / ".daredevil"
    return base


def calibration_path(data_dir) -> Path:
    return Path(data_dir) / "calibration.json"


def load_calibration(data_dir) -> Optional["IdentityModel"]:
    """Load a human-seeded IdentityModel from <data_dir>/calibration.json, if present.

    Returns None when there is no calibration file (cold start uses defaults). The
    file is written by the onboarding session (`daredevil calibrate`).
    """
    p = calibration_path(data_dir)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        fields = IdentityModel.__dataclass_fields__
        return IdentityModel(**{k: v for k, v in d.items() if k in fields})
    except Exception:
        return None


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

    # focus control
    wake_word: str = "Hey Radar"   # local wake word (openWakeWord) to grab/steer focus
    focus: Optional[str] = None    # active focus hint (azimuth label or source id), if any

    # tuning
    weights: PriorityWeights = field(default_factory=PriorityWeights)
    thresholds: Thresholds = field(default_factory=Thresholds)
    identity: IdentityModel = field(default_factory=IdentityModel)
    tracker: TrackerParams = field(default_factory=TrackerParams)
    separation: SeparationParams = field(default_factory=SeparationParams)
    nmf: NMFParams = field(default_factory=NMFParams)

    # storage / fleet
    data_dir: Optional[str] = None     # None -> default_data_dir()
    fleet_backend: str = "local"       # "local" (default) | "gun"
    gun_peers: tuple = ()              # e.g. ("http://127.0.0.1:8765/gun",)
    fleet_timeout_s: float = 2.0       # fleet-sync network timeout (tunable, not a magic literal)

    def resolved_backend(self) -> str:
        return detect_backend() if self.backend == "auto" else self.backend

    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir) if self.data_dir else default_data_dir()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["resolved_backend"] = self.resolved_backend()
        d["resolved_data_dir"] = str(self.resolved_data_dir())
        return d
