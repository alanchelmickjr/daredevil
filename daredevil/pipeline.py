"""Pipeline — the main orchestrator.

Stage 1 (spatial) -> Stage 2 (parallel slot bank) -> Stage 3 (attention router).
Slots run concurrently in a thread pool: with real backends (torch/ONNX) native
inference releases the GIL, so latency ~= the slowest slot, not the sum (patent's
core claim). The pure-Python fallback is GIL-bound — honest about that — but the
architecture, schema, and router are identical either way.
"""
from __future__ import annotations

import logging

import importlib.util as _ilu
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

from .config import Config
from .audio.capture import capture
from .audio.utils import resample
from .stage1.mic_arrays import MicArray, detect as detect_array
from .stage1.spatial import Stage1, SpatialSource
from .stage1.separation import Separator
from .stage1.nmf import SpectralLibrary
from .stage2.embedding import EmbeddingSlot
from .stage2.events import EventsSlot
from .stage2.prosody import ProsodySlot
from .stage3.tracker import UnknownTracker
from .stage3.accumulator import IdentityAccumulator
from .stage3.router import AttentionRouter
from .enrollment.manager import EnrollmentManager
from .fleet.store import make_store

SLOT_FACTORIES = {
    "embedding": EmbeddingSlot,
    "events": EventsSlot,
    "prosody": ProsodySlot,
}

log = logging.getLogger("daredevil")

_OPTIONAL_DEPS = [
    "numpy", "scipy", "torch", "torchaudio", "speechbrain", "panns_inference",
    "opensmile", "librosa", "pyroomacoustics", "sounddevice", "soundfile", "matplotlib",
]


class Pipeline:
    """Auto-detects the mic array, runs the three-stage pipeline, returns an
    awareness map dict."""

    def __init__(self, config: Optional[Config] = None,
                 array: Optional[MicArray] = None, warmup: bool = False):
        self.config = config or Config()
        # Use the human-seeded identity model from the onboarding session, if any.
        from .config import load_calibration
        _cal = load_calibration(self.config.resolved_data_dir())
        if _cal is not None:
            self.config.identity = _cal
        backend = self.config.resolved_backend()
        slot_backend = "fallback" if backend == "fallback" else "auto"

        self.array = array or detect_array()
        self.stage1 = Stage1(self.config.inference_sr)
        self.separator = Separator(backend=slot_backend)

        self.slots: Dict[str, object] = {}
        for name in self.config.enabled_slots:
            factory = SLOT_FACTORIES.get(name)
            if factory:
                self.slots[name] = factory(backend=slot_backend)
        # Identity is the headline capability — always have an embedding slot.
        if "embedding" not in self.slots:
            self.slots["embedding"] = EmbeddingSlot(backend=slot_backend)

        self.store = make_store(self.config)
        self.enrollment = EnrollmentManager(self.config, self.slots["embedding"], self.store)
        self.tracker = UnknownTracker(params=self.config.tracker)
        self.accumulator = IdentityAccumulator()
        # Frame-stable spectral features for tracker association (NMF). ECAPA stays
        # for identity; this answers the different question of "same ongoing sound?".
        n = self.config.nmf
        self.spectral = (SpectralLibrary(
            n_bins=n.n_bins, n_components=n.n_components, fit_after=n.fit_after,
            learn_iters=n.learn_iters, decompose_iters=n.decompose_iters,
            vad=self.config.thresholds.vad) if n.enabled else None)
        self.router = AttentionRouter(self.config)
        if warmup:
            self.warmup()

    # ------------------------------------------------------------------ setup
    def warmup(self) -> None:
        self.separator.warmup()
        for slot in self.slots.values():
            slot.warmup()

    def devices(self) -> dict:
        return {
            "array": self.array.summary(),
            "backend": self.config.resolved_backend(),
            "data_dir": str(self.config.resolved_data_dir()),
            "fleet_backend": self.config.fleet_backend,
            "enrolled": self.enrollment.names(),
            "slots": {n: s.backend for n, s in self.slots.items()},
            "deps": {m: (_ilu.find_spec(m) is not None) for m in _OPTIONAL_DEPS},
        }

    # ------------------------------------------------------------- enrollment
    def enroll(self, name: str, mic_seconds: float = 3.0,
               source: str = "auto", file: Optional[str] = None) -> dict:
        cap = capture(seconds=mic_seconds, sr=self.config.capture_rate,
                      source=source, file=file, array=self.array, name=name)
        audio = resample(cap.mono, cap.sample_rate, self.config.inference_sr)
        return self.enrollment.enroll(audio, self.config.inference_sr, name, mic_seconds)

    def delete(self, name: str) -> None:
        self.enrollment.delete(name)

    # ----------------------------------------------------------------- listen
    def listen(self, duration: float = 1.0, source: str = "auto",
               file: Optional[str] = None, return_audio: bool = False,
               scene: Optional[list] = None):
        self.warmup()
        cap = capture(seconds=duration, sr=self.config.capture_rate,
                      source=source, file=file, array=self.array, scene=scene)
        from .audio.utils import rms as _rms_fn
        log.info(f"capture: {cap.source} {cap.duration:.1f}s rms={_rms_fn(cap.mono):.4f}")
        spatial_sources: List[SpatialSource] = self.stage1.process(cap)

        # --- Stage 1.5: source separation — split when there are genuinely
        # multiple distinct contacts. Single source keeps clean wideband audio.
        sep = self.config.separation
        sources: List[SpatialSource] = []
        for src in spatial_sources:
            if src.truth is not None:
                sources.append(src)
                continue
            streams = self.separator.separate(src.audio, src.sr, min_energy=sep.min_energy)
            active = self._active_streams(streams, sep)
            log.info(f"separation: {len(streams)} stream(s) -> {len(active)} active contact(s)")
            if len(active) <= 1:
                sources.append(src)
            else:
                for st in active:
                    sources.append(SpatialSource(
                        audio=st["audio"], sr=st["sr"],
                        azimuth=src.azimuth, elevation=src.elevation, truth=None))

        # --- Stage 2: parallel pass (this is the one we report results from)
        t0 = time.perf_counter()
        per_source = []
        with ThreadPoolExecutor(max_workers=max(1, len(self.slots))) as ex:
            for src in sources:
                ctx = {"truth": src.truth}
                futs = {n: ex.submit(self._run_slot, s, src.audio, src.sr, ctx)
                        for n, s in self.slots.items()}
                per_source.append({n: f.result() for n, f in futs.items()})
        parallel_ms = (time.perf_counter() - t0) * 1000

        # --- Stage 2: sequential pass (for the latency comparison only)
        t1 = time.perf_counter()
        for src in sources:
            ctx = {"truth": src.truth}
            for s in self.slots.values():
                self._run_slot(s, src.audio, src.sr, ctx)
        sequential_ms = (time.perf_counter() - t1) * 1000

        # --- assemble per-source records from the parallel results.
        # The identity accumulator collects good frames (Speech + energetic) into
        # per-source centroids that persist across tracker thrash. The SPRT runs
        # against the accumulated centroid — not one-shot frame embeddings — so
        # identity confidence grows as good chunks coalesce over time.
        from .audio.utils import rms as _rms
        records = []
        for src, res in zip(sources, per_source):
            emb = res.get("embedding", {}).get("vector")
            if emb is None:
                emb = self.slots["embedding"].run(src.audio, src.sr)["vector"]
            ev = res.get("events") or {"class": "unknown", "confidence": 0.0, "safety_critical": False}
            pr = res.get("prosody") or {"state": "calm", "distress": 0.0}
            energy = _rms(src.audio)

            pos = None
            if src.azimuth is not None:
                pos = {"azimuth": src.azimuth, "elevation": src.elevation or 0.0}

            # Tracker for spatial continuity / display.
            track_feat = self.spectral.feature(src.audio, src.sr) if self.spectral else emb
            track_id = self.tracker.assign(track_feat, position=pos, event_class=ev.get("class"))

            # Identity: one SPRT per enrolled speaker, keyed by speaker name alone.
            # Every frame above VAD gets scored. No track ID, no source ID — just
            # "does this embedding sound like alan?" Evidence accumulates across all
            # frames regardless of tracker thrash. The $2 chip approach.
            match = self.enrollment.match(emb, energy=energy, key="global")

            identity = None
            if self.enrollment.is_match(match):
                identity = {"name": match["name"], "score": match["score"],
                            "enrollment_confidence": match["enrollment_confidence"]}
                log.info(f"  {track_id} → MATCHED {match['name']} "
                         f"raw={match['raw']:.3f} llr={match['llr']:.2f} score={match['score']:.3f}")
            else:
                bl = match["llr"] if match else 0.0
                log.info(f"  {track_id} event={ev.get('class')} energy={energy:.4f} best_llr={bl:.2f}")

            t_status = self.tracker.status_of(track_id) or "confirmed"
            records.append({"identity": identity, "unknown_id": track_id,
                            "event": ev, "prosody": pr, "position": pos,
                            "track_status": t_status})

        # Include coasting tracks that weren't seen this frame — they persist
        # on the radar until they age out. Status tells the HUD what to do:
        # confirmed = on radar, coasting = fade to sidebar, tentative = dim.
        seen_ids = {r["unknown_id"] for r in records}
        for t in self.tracker._tracks:
            if t["id"] not in seen_ids:
                records.append({
                    "identity": None, "unknown_id": t["id"],
                    "event": {"class": t.get("event_class") or "unknown",
                              "confidence": 0.0, "safety_critical": False},
                    "prosody": {"state": "calm", "distress": 0.0},
                    "position": {"azimuth": t.get("azimuth"), "elevation": 0.0}
                    if t.get("azimuth") is not None else None,
                    "track_status": t["status"],
                })

        # Forget SPRT accumulators for contacts that have aged out.
        self.enrollment.retain(self.tracker.live_ids())

        timing = {"parallel_ms": round(parallel_ms, 1),
                  "sequential_ms": round(sequential_ms, 1)}

        amap = self.router.build(records, datetime.now().isoformat(), timing,
                                 cap.array, self.config.resolved_backend())
        if return_audio:
            # audio returned transiently for visualization only; never persisted.
            return amap, cap.mono, cap.sample_rate
        return amap

    # ----------------------------------------------------------------- helper
    def _run_slot(self, slot, audio, sr, ctx):
        return slot.run(audio, sr, ctx)

    def _active_streams(self, streams: list, sep) -> list:
        """Keep only separated streams that are genuinely distinct, energetic contacts.

        The loudest stream is the primary contact; a secondary stream is kept only
        if it carries a meaningful fraction of the primary's energy AND is
        spectrally distinct from it. Otherwise the secondary is a separation
        artifact of a single source and is discarded.

        Key heuristic: if two streams have nearly equal energy (ratio > energy_ratio_cap),
        ConvTasNet almost certainly split one source in half — real concurrent sources
        rarely have identical loudness.
        """
        if not streams:
            return []
        from .audio.utils import fingerprint, cosine
        ordered = sorted(streams, key=lambda s: s["energy"], reverse=True)
        primary = ordered[0]
        pf = fingerprint(primary["audio"], primary["sr"])
        active = [primary]
        for st in ordered[1:]:
            if st["energy"] < sep.dominance_ratio * primary["energy"]:
                continue
            # Near-equal energy = one source split in half, not two real sources.
            if primary["energy"] > 0 and st["energy"] / primary["energy"] > sep.energy_ratio_cap:
                continue
            if cosine(fingerprint(st["audio"], st["sr"]), pf) >= sep.distinct_cosine:
                continue
            active.append(st)
        return active
