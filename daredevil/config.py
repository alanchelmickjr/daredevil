"""Central configuration: thresholds, weights, safety classes, backend detection.

Everything tunable in one place. Pure stdlib — importing this never pulls a heavy
dependency. The defaults below are taken directly from the patent's algorithmic
formulations (cosine match T=0.70, enrollment time constant ~3s, the weighted
priority sum) so the software and the firmware speak the same language.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger("daredevil.config")

__all__ = [
    "Config",
    "PriorityWeights",
    "Thresholds",
    "IdentityModel",
    "TrackerParams",
    "SeparationParams",
    "NMFParams",
    "WakeWordParams",
    "HeuristicThresholds",
    "SyntheticParams",
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


def is_speech_class(label) -> bool:
    """True if an event label is human speech, across backend label conventions.

    PANNs emits AudioSet labels verbatim ("Speech", "Male speech, man speaking");
    the fallback/synthetic paths emit lowercase "speech". Exact lowercase
    comparison made `active_speaker` and the router's owner-speaking rule
    permanently dead on the real backend (gap M3) — normalize once, here.
    """
    return bool(label) and "speech" in str(label).casefold()


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
    speech_gate_energy: float = 0.012  # RMS floor for SPRT-grade speech (≈3× vad; 0.05 muted normal-distance talk)
    speech_gate_zcr: float = 3000.0    # max zero-crossings/s — above this it's clicks/noise, not voice
    enroll_tau: float = 3.0        # tau_enroll — enrollment confidence time constant (s)
    surface: float = 0.45          # priority at/above which a source is routed to the LLM
    display: float = 0.5            # centroid confidence below which a source is hidden from HUD


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
    immediate_cosine: float = 0.80 # two consecutive frames this strong match outright
    demote_bound_scale: float = 2.0  # revoke a held identity when post-decision contrary
                                     # evidence reaches B*this. 2.0 = symmetric with the
                                     # 2-frame acquisition: ~2 agreeing frames in, ~2
                                     # sustained contrary frames out; one borderline frame
                                     # never kills a hold
    frame_llr_clip_scale: float = 0.9  # per-frame |LLR| is clipped to A*this: no SINGLE
                                       # frame (outlier, blip, replay spike) can cross the
                                       # accept bound alone — a decision always takes >=2
                                       # frames of agreeing evidence (gaps B2/M16)
    quality_full_energy: float = 0.02  # energy at which a frame carries full evidential weight
    h0_source: str = "default"     # provenance of the impostor stats: "default" (textbook,
                                   # nothing measured) | "room-tone" | "room+world" — recorded
                                   # in calibration.json so a silence-fit H0 is never mistaken
                                   # for one that has heard other voices (gap B1)

    # --- real-time auto-calibration (CFAR: hold the false-alarm rate constant as
    # the room changes). The human onboarding session seeds the first model; these
    # keep H0 — and optionally the voiceprint — tuned as we go.
    adapt_background: bool = True       # online-estimate H0 from live ambient audio
    bg_adapt_rate: float = 0.02         # EMA rate for the running background model
    bg_guard_sigmas: float = 1.5        # (legacy band; superseded by the censored window)
    bg_censor_target_sigmas: float = 2.0  # censor background samples above
                                          # target_mean - this*target_std: plausibly-target
                                          # frames (e.g. the owner pre-decision) must never
                                          # teach H0, or one strong frame inverts the test
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
    coast_s: float = 15.0          # keep a missed track COASTING this long
    delete_s: float = 45.0         # delete a track after this much silence
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


@dataclass
class WakeWordParams:
    """The 'use their name' channel of attention (see stage2/wake.py).

    Humans grab attention first by the *sound* of the voice (urgency/distress —
    handled by prosody + the router's DISTRESS escalation) and, when that doesn't
    land, by *name*. These tune the by-example name detector; openWakeWord upgrades
    it when ``oww_model`` is set and the package is installed (no cloud, no key).
    """

    enabled: bool = True
    analysis_sr: int = 8000        # contour analysis rate (phonetic detail lives < 4 kHz)
    frame_ms: float = 25.0         # per-frame analysis window
    hop_ms: float = 10.0           # frame step
    n_bands: int = 12              # log-spaced spectral bands per frame
    threshold: float = 0.72        # by-example DTW similarity at/above which the name fires
    min_phrase_s: float = 0.30     # reject enrollments shorter than a plausible name
    grab_focus: bool = True        # on hearing its name, focus the caller — the natural response
    oww_model: str = ""            # optional openWakeWord model name ("" => by-example only)
    oww_threshold: float = 0.5     # openWakeWord score at/above which the name fires


@dataclass
class HeuristicThresholds:
    """Thresholds for the fallback heuristic event classifier (Slot B) and
    prosody distress calculation (Slot C). Centralised here so no magic
    numbers live in the slot files."""

    # events.py heuristic
    silence_energy: float = 0.01
    baby_cry_centroid_hz: float = 1500.0
    baby_cry_zcr: float = 0.20
    alarm_centroid_hz: float = 800.0
    speech_centroid_hz: float = 500.0
    speech_zcr: float = 0.18

    # prosody.py distress weights (must sum to 1.0)
    distress_w_f0: float = 0.45
    distress_w_jitter: float = 0.30
    distress_w_roughness: float = 0.25

    # prosody.py state thresholds
    state_distressed: float = 0.60
    state_stressed: float = 0.40
    state_confused_f0_std: float = 0.70

    # prosody.py stdlib proxy normalisation
    centroid_norm_hz: float = 2000.0

    # speech quality gate (utils.py)
    speech_energy_floor: float = 0.015

    # calibrate.py audio level feedback
    level_too_quiet: float = 0.01
    level_too_hot: float = 0.40


@dataclass
class SyntheticParams:
    """Constants for deterministic synthetic audio generation (capture.py).
    Centralised so they're visible and tunable in one place."""

    voice_noise_floor: float = 0.02
    ambient_noise_floor: float = 0.05

    # PANNs model target sample rate
    panns_sr: int = 32000
    # ConvTasNet training sample rate
    convtasnet_sr: int = 8000

    # SRP-PHAT FFT parameters (spatial.py)
    srp_fft_size: int = 1024
    srp_hop_size: int = 512


def default_data_dir() -> Path:
    """Where enrolled voiceprints live by default.

    Never inside the installed package. Honours $DAREDEVIL_HOME, else ~/.daredevil.
    """
    env = os.environ.get("DAREDEVIL_HOME")
    base = Path(env) if env else Path.home() / ".daredevil"
    return base


def calibration_path(data_dir) -> Path:
    return Path(data_dir) / "calibration.json"


# Below this voice/background separability (d′) a saved calibration carries no
# usable signal — using it would make the matcher reject the very person it should
# recognize, so we fall back to the (more separable) textbook defaults instead.
CALIBRATION_MIN_DPRIME = 0.5


def _calibration_dprime(m: "IdentityModel") -> float:
    """Separability (d′) between the target and background cosine distributions."""
    denom = ((m.target_std ** 2 + m.impostor_std ** 2) / 2.0) ** 0.5
    if denom <= 0:
        return 0.0
    return (m.target_mean - m.impostor_mean) / denom


def load_calibration(data_dir) -> Optional["IdentityModel"]:
    """Load a human-seeded IdentityModel from <data_dir>/calibration.json, if present.

    Returns None when there is no calibration file (cold start uses defaults), or
    when the saved model is degenerate. The file is written by the onboarding
    session (`daredevil calibrate`).
    """
    p = calibration_path(data_dir)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        fields = IdentityModel.__dataclass_fields__
        model = IdentityModel(**{k: v for k, v in d.items() if k in fields})
    except Exception:
        log.debug("calibration file %s unreadable, starting fresh", p)
        return None
    # A calibration fit while the mic was broken (silence/dropped audio) can have no
    # real voice/background separation. Running it is worse than the defaults — it can
    # make WHO reject the enrolled owner. Ignore it (and say so) rather than fail
    # silently: a protection system that silently fails is worse than none.
    sep = _calibration_dprime(model)
    if sep < CALIBRATION_MIN_DPRIME:
        log.warning(
            "ignoring calibration %s: d'=%.2f below %.2f (no usable voice/background "
            "separation). Using defaults — re-run `daredevil calibrate` once the mic "
            "is working.", p, sep, CALIBRATION_MIN_DPRIME)
        return None
    return model


def detect_backend() -> str:
    """Best available inference backend, without importing heavy libs eagerly.

    Returns one of: "cuda", "mps", "cpu", "fallback".
    "fallback" means no torch/onnx — the pure-Python path still produces a full
    awareness map, just with heuristic slots.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        log.debug("torch not available, using fallback backend")
        return "fallback"
    try:
        import torch

        if torch.cuda.is_available():
            log.debug("detected backend: cuda")
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            log.debug("detected backend: mps")
            return "mps"
        log.debug("detected backend: cpu (torch present, no GPU)")
        return "cpu"
    except Exception:
        log.debug("torch probe failed, using fallback backend")
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
    hop_seconds: float = 0.5       # continuous-stream hop: new analysis every hop, on the
                                   # trailing window_seconds of audio (viz/stream.py)

    # slots
    enabled_slots: tuple = ("embedding", "events", "prosody")
    backend: str = "auto"          # "auto" -> detect_backend()

    # behaviour
    allow_cloud: bool = False      # hard guarantee: no network egress for inference

    # focus control — attention by name (see stage2/wake.py)
    wake_word: str = "Hey Radar"   # the system's name; hearing it grabs/steers focus
    focus: Optional[str] = None    # active focus hint (azimuth label or source id), if any

    # tuning
    weights: PriorityWeights = field(default_factory=PriorityWeights)
    thresholds: Thresholds = field(default_factory=Thresholds)
    identity: IdentityModel = field(default_factory=IdentityModel)
    tracker: TrackerParams = field(default_factory=TrackerParams)
    separation: SeparationParams = field(default_factory=SeparationParams)
    nmf: NMFParams = field(default_factory=NMFParams)
    wakeword: WakeWordParams = field(default_factory=WakeWordParams)
    heuristic: HeuristicThresholds = field(default_factory=HeuristicThresholds)
    synth: SyntheticParams = field(default_factory=SyntheticParams)

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
