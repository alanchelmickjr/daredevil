"""Continuous stream loop — the always-on ears (gap B3 in docs/GAP_VOICE_DISCRIMINATOR.md).

Two daemon threads around a small ring buffer:

  reader   — blocking-reads hop-sized chunks from the persistent mic stream (self-pacing,
             gapless: the read itself is the clock) and appends them to the ring.
  inferer  — every hop, analyzes the trailing `window_seconds` of the ring. Inference
             (~0.4s measured on the M2) fits inside the 0.5s default hop, so analysis
             keeps pace with capture instead of sampling ≤40% of the room's audio the
             way the old capture-inside-GET design did.

Dependency-injected (read_chunk / analyze / pause callables) so the scheduling logic is
unit-testable with synthetic chunks and no audio hardware (tests/test_stream.py). The
ring is cleared on pause (calibration owns the mic during its phases) so a resumed
stream never analyzes a window stitched across the gap.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, List, Optional, Tuple

log = logging.getLogger("daredevil.stream")

# Seconds of audio kept in the ring beyond the analysis window — headroom for jitter,
# small and named rather than a magic literal.
RING_HEADROOM_WINDOWS = 2.0


class StreamLoop:
    """Continuous capture + analysis scheduler.

    read_chunk() -> (channels, sample_rate, meta): blocking, hop-sized, contiguous.
    analyze(channels, sample_rate, meta): full window; called from the inferer thread only.
    pause() -> bool: while True, reading/analysis stop and the ring clears (calibration).
    """

    def __init__(self,
                 read_chunk: Callable[[], Tuple[List[List[float]], int, object]],
                 analyze: Callable[[List[List[float]], int, object], None],
                 window_seconds: float,
                 hop_seconds: float,
                 pause: Optional[Callable[[], bool]] = None):
        self.read_chunk = read_chunk
        self.analyze = analyze
        self.window_s = window_seconds
        self.hop_s = hop_seconds
        self.pause = pause or (lambda: False)
        self._lock = threading.Lock()
        self._ring: List[List[float]] = []      # per-channel sample lists
        self._sr: Optional[int] = None
        self._meta = None
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self.windows_analyzed = 0               # lifetime counter (the B3 check reads logs)
        self._minute_count = 0
        self._minute_mark = time.monotonic()
        # Borrowing: the mic has ONE owner — this loop. Calibration must not open a
        # second input stream (PortAudio -9986, observed live 2026-07-02); instead it
        # borrows chunks: while a borrower is registered, the reader routes chunks to
        # the borrow queue instead of the ring, and analysis naturally idles.
        self._borrow_q: Optional[list] = None
        self._borrow_ready = threading.Condition(self._lock)

    # ------------------------------------------------------------- lifecycle
    def start(self):
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._reader, name="dd-stream-reader", daemon=True),
            threading.Thread(target=self._inferer, name="dd-stream-inferer", daemon=True),
        ]
        for t in self._threads:
            t.start()
        log.info("stream: continuous loop started (window=%.2fs hop=%.2fs)",
                 self.window_s, self.hop_s)

    def stop(self):
        self._stop.set()
        with self._lock:
            self._borrow_ready.notify_all()
        for t in self._threads:
            t.join(timeout=2.0)

    # ------------------------------------------------------------- borrowing
    def borrow_start(self):
        """Divert incoming chunks to the borrower (calibration). Single borrower."""
        with self._lock:
            self._borrow_q = []

    def borrow_chunk(self, timeout: float):
        """Next diverted chunk as (channels, sr); raises TimeoutError if the reader
        produced nothing in `timeout` seconds (mic dead — surface it, don't hang)."""
        deadline = time.monotonic() + timeout
        with self._lock:
            while not self._borrow_q:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stop.is_set():
                    raise TimeoutError("stream produced no audio for the borrower")
                self._borrow_ready.wait(remaining)
            return self._borrow_q.pop(0)

    def borrow_end(self):
        with self._lock:
            self._borrow_q = None

    # --------------------------------------------------------------- threads
    def _reader(self):
        max_len_s = self.window_s + RING_HEADROOM_WINDOWS * self.window_s
        while not self._stop.is_set():
            if self.pause():
                with self._lock:
                    self._ring = []             # never stitch across a pause
                self._stop.wait(self.hop_s)
                continue
            try:
                channels, sr, meta = self.read_chunk()
            except Exception as e:
                log.warning("stream: read_chunk failed (%s) — retrying in one hop", e)
                self._stop.wait(self.hop_s)
                continue
            with self._lock:
                if self._borrow_q is not None:
                    # A borrower (calibration) owns the audio right now.
                    self._borrow_q.append((channels, sr))
                    self._borrow_ready.notify_all()
                    self._ring = []             # never stitch across a borrow
                    continue
                if self._sr is not None and sr != self._sr:
                    self._ring = []             # device/rate changed mid-flight
                self._sr = sr
                self._meta = meta
                if len(self._ring) != len(channels):
                    self._ring = [[] for _ in channels]
                keep = int(max_len_s * sr)
                for i, ch in enumerate(channels):
                    self._ring[i].extend(ch)
                    if len(self._ring[i]) > keep:
                        del self._ring[i][:len(self._ring[i]) - keep]

    def _inferer(self):
        while not self._stop.is_set():
            tick = time.monotonic()
            if not self.pause():
                window = None
                with self._lock:
                    if self._sr:
                        n = int(self.window_s * self._sr)
                        if self._ring and all(len(ch) >= n for ch in self._ring):
                            window = [ch[-n:] for ch in self._ring]
                            sr, meta = self._sr, self._meta
                if window is not None:
                    try:
                        self.analyze(window, sr, meta)
                        self.windows_analyzed += 1
                        self._minute_count += 1
                    except Exception as e:
                        log.warning("stream: analyze failed (%s)", e)
                now = time.monotonic()
                if now - self._minute_mark >= 60.0:
                    log.info("stream: analyzed %d windows in last %.0fs",
                             self._minute_count, now - self._minute_mark)
                    self._minute_count = 0
                    self._minute_mark = now
            # pace to the hop; never a negative or magic sleep
            remaining = self.hop_s - (time.monotonic() - tick)
            if remaining > 0:
                self._stop.wait(remaining)
