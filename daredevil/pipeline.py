"""Pipeline — the main orchestrator.

Stage 1 (spatial) -> Stage 2 (parallel slot bank) -> Stage 3 (attention router).
Slots run concurrently in a thread pool: with real backends (torch/ONNX) native
inference releases the GIL, so latency ~= the slowest slot, not the sum (patent's
core claim). The pure-Python fallback is GIL-bound — honest about that — but the
architecture, schema, and router are identical either way.
"""
from __future__ import annotations

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
               file: Optional[str] = None) -> dict:
        self.warmup()
        cap = capture(seconds=duration, sr=self.config.capture_rate,
                      source=source, file=file, array=self.array)
        sources: List[SpatialSource] = self.stage1.process(cap)

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

            match = self.enrollment.match(emb)
            identity, unknown_id = None, None
            if self.enrollment.is_match(match):
                identity = {"name": match["name"], "score": match["score"],
                            "enrollment_confidence": match["enrollment_confidence"]}
            else:
                unknown_id = self.tracker.assign(emb)

            position = None
            if src.azimuth is not None:
                position = {"azimuth": src.azimuth, "elevation": src.elevation or 0.0}

            records.append({"identity": identity, "unknown_id": unknown_id,
                            "event": ev, "prosody": pr, "position": position})

        timing = {"parallel_ms": round(parallel_ms, 1),
                  "sequential_ms": round(sequential_ms, 1)}
        if self.config.simulate_latency:
            timing["simulated"] = True

        return self.router.build(records, datetime.now().isoformat(), timing,
                                 cap.array, self.config.resolved_backend())

    # ----------------------------------------------------------------- helper
    def _run_slot(self, slot, audio, sr, ctx):
        if self.config.simulate_latency:
            ms = self.config.simulated_slot_ms.get(slot.name, 0.0)
            if ms:
                time.sleep(ms / 1000.0)  # sleep releases the GIL -> parallel wins
        return slot.run(audio, sr, ctx)
