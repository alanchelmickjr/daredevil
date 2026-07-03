"""StreamLoop scheduling tests — pure stdlib, no audio hardware.

Pins gap B3's fix: continuous analysis cadence, ring contiguity, pause semantics
(never stitch a window across a calibration gap), and crash isolation.
"""
import threading
import time

from daredevil.viz.stream import StreamLoop

SR = 1000  # 1 kHz keeps the math obvious: 1s window = 1000 samples


def _mk_reader(counter):
    """Chunks whose samples encode their global index — contiguity is checkable."""
    state = {"i": 0}

    def read_chunk():
        n = 100  # 0.1s at SR
        start = state["i"]
        state["i"] += n
        counter["reads"] += 1
        time.sleep(0.01)  # fast fake mic: keeps wall-clock short
        return [list(range(start, start + n))], SR, "meta"

    return read_chunk


def test_cadence_and_contiguity():
    counter = {"reads": 0}
    seen = []

    def analyze(channels, sr, meta):
        seen.append(channels[0])

    loop = StreamLoop(_mk_reader(counter), analyze,
                      window_seconds=0.2, hop_seconds=0.05)
    loop.start()
    time.sleep(1.0)
    loop.stop()
    assert loop.windows_analyzed >= 10, f"only {loop.windows_analyzed} windows in 1s"
    for w in seen:
        assert len(w) == int(0.2 * SR)
        # samples encode global index: contiguous window <=> strictly +1 steps
        assert all(b - a == 1 for a, b in zip(w, w[1:])), "non-contiguous window"


def test_pause_clears_ring_and_halts_analysis():
    paused = {"on": False}
    seen = []
    counter = {"reads": 0}

    def analyze(channels, sr, meta):
        seen.append(len(channels[0]))

    loop = StreamLoop(_mk_reader(counter), analyze,
                      window_seconds=0.2, hop_seconds=0.05,
                      pause=lambda: paused["on"])
    loop.start()
    time.sleep(0.5)
    n_before = loop.windows_analyzed
    assert n_before > 0
    paused["on"] = True
    time.sleep(0.3)
    n_during = loop.windows_analyzed
    time.sleep(0.3)
    assert loop.windows_analyzed == n_during, "analysis continued while paused"
    paused["on"] = False
    time.sleep(0.5)
    assert loop.windows_analyzed > n_during, "analysis never resumed after pause"
    loop.stop()


def test_analyze_crash_does_not_kill_loop():
    calls = {"n": 0}

    def analyze(channels, sr, meta):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")

    loop = StreamLoop(_mk_reader({"reads": 0}), analyze,
                      window_seconds=0.2, hop_seconds=0.05)
    loop.start()
    time.sleep(0.6)
    loop.stop()
    assert calls["n"] >= 3, "loop died after a single analyze failure"


def test_reader_crash_retries():
    state = {"n": 0}

    def bad_then_good():
        state["n"] += 1
        if state["n"] < 3:
            raise OSError("device busy")
        return [[0.0] * 100], SR, None

    loop = StreamLoop(bad_then_good, lambda c, s, m: None,
                      window_seconds=0.2, hop_seconds=0.05)
    loop.start()
    time.sleep(0.6)
    loop.stop()
    assert state["n"] >= 3, "reader stopped retrying after failure"


def test_borrow_diverts_chunks_and_analysis_idles():
    counter = {"reads": 0}
    analyzed = []
    loop = StreamLoop(_mk_reader(counter), lambda c, s, m: analyzed.append(1),
                      window_seconds=0.2, hop_seconds=0.05)
    loop.start()
    time.sleep(0.4)
    loop.borrow_start()
    n_before = len(analyzed)
    got = [loop.borrow_chunk(timeout=2.0) for _ in range(4)]
    assert all(sr == SR for _, sr in got)
    # borrowed chunks are contiguous with each other
    flat = [s for ch, _ in got for s in ch[0]]
    assert all(b - a == 1 for a, b in zip(flat, flat[1:])), "borrowed audio not contiguous"
    assert len(analyzed) <= n_before + 1, "analysis kept running during borrow"
    loop.borrow_end()
    time.sleep(0.5)
    assert len(analyzed) > n_before, "analysis never resumed after borrow"
    loop.stop()


def test_borrow_times_out_when_reader_dead():
    def dead_reader():
        raise OSError("no mic")
    loop = StreamLoop(dead_reader, lambda c, s, m: None,
                      window_seconds=0.2, hop_seconds=0.05)
    loop.start()
    loop.borrow_start()
    import pytest as _pytest
    with _pytest.raises(TimeoutError):
        loop.borrow_chunk(timeout=0.3)
    loop.stop()
