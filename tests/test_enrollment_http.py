"""End-to-end test of the enrollment HTTP flow — the path Dexter uses.

Starts the server in-process on a test port, drives the full calibration flow
with curl-equivalent requests, then verifies the enrolled person appears in the
awareness map. All synthetic, no mic needed.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
import urllib.error

import pytest


def _post(url, body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def _poll_phase(base, target_phase, timeout=60):
    """Poll /calibrate/status until phase matches target_phase or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = _get(f"{base}/calibrate/status")
        if status["phase"] == target_phase:
            return status
        time.sleep(0.3)
    raise TimeoutError(f"phase never reached {target_phase!r}, last: {status}")


@pytest.fixture(scope="module")
def server_url():
    """Start the HUD server in a background thread on an ephemeral port."""
    from http.server import ThreadingHTTPServer
    from daredevil.config import Config
    from daredevil.pipeline import Pipeline
    from daredevil.stage1.mic_arrays import MACBOOK_3
    from daredevil.viz.server import _make_handler, _State

    state = _State(live=False)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    # Wait for readiness.
    for _ in range(30):
        try:
            _get(f"{base}/awareness")
            break
        except Exception:
            time.sleep(0.5)
    yield base
    httpd.shutdown()


def test_awareness_returns_json(server_url):
    amap = _get(f"{server_url}/awareness")
    assert "sources" in amap
    assert amap["privacy"]["cloud_used"] is False


def test_full_calibration_flow(server_url):
    base = server_url

    # 1. Start calibration
    resp = _post(f"{base}/calibrate/start", {
        "name": "testperson", "seconds": 5, "others": False,
    })
    assert resp["ok"] is True

    status = _get(f"{base}/calibrate/status")
    assert status["phase"] == "ready"
    assert status["active"] is True

    # 2. Advance to voice capture
    _post(f"{base}/calibrate/phase")
    # Wait for countdown + voice capture to finish
    status = _poll_phase(base, "voice_done", timeout=60)
    assert status["voice_frames"] > 0, "expected at least one voice frame"

    # 3. Advance to background capture
    _post(f"{base}/calibrate/phase")
    status = _poll_phase(base, "background_done", timeout=60)
    assert status["bg_frames"] >= 0  # background frames (may be 0 if no voiced windows)

    # 4. Advance to fitting (others=False, so skips world phase)
    _post(f"{base}/calibrate/phase")
    status = _poll_phase(base, "done", timeout=30)
    assert status["quality"] in ("good", "fair", "poor")
    assert status["dprime"] is not None
    assert status["active"] is False

    # 5. After calibration, awareness map should show the enrolled person
    # May need a couple of listens for the SPRT to accumulate evidence.
    found = False
    for _ in range(5):
        amap = _get(f"{base}/awareness")
        for src in amap.get("sources", []):
            if src.get("id") == "testperson" and src.get("type") == "enrolled":
                found = True
                break
        if found:
            break
        time.sleep(0.5)
    assert found, (
        f"enrolled 'testperson' not found in awareness map. "
        f"Sources: {[s.get('id') for s in amap.get('sources', [])]}"
    )


def test_focus_endpoint(server_url):
    base = server_url

    # Set focus
    resp = _post(f"{base}/focus", {"id": "testperson"})
    assert resp["ok"] is True
    assert resp["focus"] == "testperson"

    # Awareness map should show focus field
    amap = _get(f"{base}/awareness")
    assert amap.get("focus") == "testperson"

    # Clear focus
    resp = _post(f"{base}/focus", {})
    assert resp["ok"] is True
    assert resp["focus"] is None

    amap = _get(f"{base}/awareness")
    assert amap.get("focus") is None
