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
from .stage2.embedding import EmbeddingSlot
from .stage2.events import EventsSlot
from .stage2.prosody import ProsodySlot
from .stage3.tracker import UnknownTracker
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
        self.tracker = UnknownTracker(self.config.thresholds.unknown_track)
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

        # --- Stage 1.5: source separation
        sources: List[SpatialSource] = []
        for src in spatial_sources:
            if src.truth is not None:
                sources.append(src)
            else:
                separated = self.separator.separate(src.audio, src.sr)
                log.info(f"separation: {len(separated)} streams from 1 spatial source")
                for i, stream in enumerate(separated):
                    log.info(f"  stream {i}: energy={stream['energy']:.4f}")
                    sources.append(SpatialSource(
                        audio=stream["audio"], sr=stream["sr"],
                        azimuth=src.azimuth, elevation=src.elevation,
                        truth=None,
                    ))

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

        # --- assemble per-source records from the parallel results
        records = []
        for src, res in zip(sources, per_source):
            emb = res.get("embedding", {}).get("vector")
            if emb is None:
                emb = self.slots["embedding"].run(src.audio, src.sr)["vector"]
            ev = res.get("events") or {"class": "unknown", "confidence": 0.0, "safety_critical": False}
            pr = res.get("prosody") or {"state": "calm", "distress": 0.0}

            from .audio.utils import rms as _rms
            energy = _rms(src.audio)

            pos = None
            if src.azimuth is not None:
                pos = {"azimuth": src.azimuth, "elevation": src.elevation or 0.0}
            single_source = (len(sources) == 1)

            # Match with LLR accumulation — confidence builds across frames
            match = self.enrollment.match(emb, energy=energy)
            identity, unknown_id = None, None
            if self.enrollment.is_match(match):
                identity = {"name": match["name"], "score": match["score"],
                            "enrollment_confidence": match["enrollment_confidence"]}
                log.info(f"  → MATCHED {match['name']} score={match['score']:.3f}")
            else:
                unknown_id = self.tracker.assign(
                    emb, position=pos, event_class=ev.get("class"),
                    single_source=single_source)
                llr_state = self.enrollment._llr.get("alan", 0)
                log.info(f"  → {unknown_id} event={ev.get('class')} energy={energy:.4f} llr_alan={llr_state:.3f}")

            position = None
            if src.azimuth is not None:
                position = {"azimuth": src.azimuth, "elevation": src.elevation or 0.0}

            records.append({"identity": identity, "unknown_id": unknown_id,
                            "event": ev, "prosody": pr, "position": position})

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
