"""Local web HUD server — stdlib only, binds to localhost, no cloud.

Serves the neumorphic-steampunk orbital HUD and streams awareness maps:
  GET /            -> the HUD page
  GET /awareness   -> the current awareness map (JSON)
  GET /probe?id=X  -> transcribe source X (local STT) and route it to the LLM
  GET /calibrate/status -> calibration session state (polled during onboarding)
  POST /calibrate/start -> begin a calibration session
  POST /calibrate/phase -> advance to the next phase

`daredevil serve` launches it.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from ..config import Config
from ..pipeline import Pipeline
from ..stage1.mic_arrays import MACBOOK_3

_WEB = Path(__file__).parent / "web"
log = logging.getLogger("daredevil.hud")


class _CalibrationSession:
    """Server-driven calibration state machine, polled by the browser."""

    def __init__(self, state: "_State"):
        self.state = state
        self.pipe = state.pipe
        self._lock = threading.Lock()
        self._thread = None
        self._name = "you"
        self._seconds = 20.0
        self._include_others = True
        self._voiceprint = None
        self._target_cos: list = []
        self._bg_cos: list = []
        self.status = self._idle_status()

    def _idle_status(self) -> dict:
        return {
            "active": False, "phase": "idle", "phase_index": 0,
            "countdown": 0, "elapsed": 0.0, "duration": 20.0,
            "level": 0.0, "prompt": "", "voice_frames": 0,
            "bg_frames": 0, "dprime": None, "error_pct": None,
            "model": None, "quality": None,
        }

    def start(self, name: str = "you", seconds: float = 20.0, others: bool = True):
        with self._lock:
            self._name = name or "you"
            self._seconds = seconds
            self._include_others = others
            self._voiceprint = None
            self._target_cos = []
            self._bg_cos = []
            self.status = {
                "active": True, "phase": "ready", "phase_index": 0,
                "countdown": 0, "elapsed": 0.0, "duration": seconds,
                "level": 0.0,
                "prompt": "Ready to calibrate. Hit NEXT PHASE when you’re set.",
                "voice_frames": 0, "bg_frames": 0,
                "dprime": None, "error_pct": None, "model": None, "quality": None,
            }

    def advance(self):
        with self._lock:
            if self.status["phase"] == "idle":
                return
            if self._thread and self._thread.is_alive():
                return
        nxt = self._next_phase()
        if nxt is None:
            return
        self._thread = threading.Thread(target=self._run_phase, args=(nxt,), daemon=True)
        self._thread.start()

    def _next_phase(self):
        cur = self.status["phase"]
        if cur in ("ready", "idle"):
            return "voice"
        if cur == "voice_done":
            return "background"
        if cur == "background_done":
            return "world" if self._include_others else "fitting"
        if cur == "world_done":
            return "fitting"
        return None

    def _run_phase(self, phase: str):
        prompts = {
            "voice": "SPEAK — talk to me like I’m across the table",
            "background": "SHUT UP — hands off, mouth closed, let the room breathe",
            "world": "THE WORLD — let the chaos in, turn it up",
            "fitting": "Crunching numbers…",
        }
        idx = {"voice": 1, "background": 2, "world": 3, "fitting": 0}[phase]
        self.status["phase_index"] = idx
        self.status["prompt"] = prompts[phase]

        if phase == "fitting":
            self._do_fit()
            return

        # 3-2-1 countdown
        self.status["phase"] = "countdown"
        for i in (3, 2, 1):
            self.status["countdown"] = i
            time.sleep(1.0)
        self.status["countdown"] = 0
        self.status["phase"] = phase
        self._do_capture(phase)
        self.status["phase"] = f"{phase}_done"

    def _do_capture(self, phase: str):
        from ..calibrate import Calibrator, _AMBIENT_SCENE
        from ..audio.utils import rms as _rms

        cal = Calibrator(self.pipe)
        source = "live" if self.state.live else "synthetic"
        seconds = self._seconds
        chunk = 0.5
        n_chunks = int(seconds / chunk)
        all_audio: list = []
        sr = None

        if phase == "voice" and self.pipe.store.get(self._name):
            self.pipe.store.delete(self._name)

        for i in range(n_chunks):
            self.status["elapsed"] = i * chunk
            scene = None if self.state.live else (
                None if phase == "voice" else _AMBIENT_SCENE)
            audio, sample_rate, level, _ = cal.capture_audio(
                chunk, source,
                name=(self._name if phase == "voice" else None),
                scene=scene,
            )
            sr = sample_rate
            all_audio.extend(audio)
            self.status["level"] = min(1.0, level / 0.15)

        self.status["elapsed"] = seconds
        self.status["level"] = 0.0

        if phase == "voice":
            enr = self.pipe.enrollment.enroll(all_audio, sr, self._name, seconds)
            self._voiceprint = self.pipe.store.get(self._name)["vector"]
            self._target_cos = cal.cosines_to(all_audio, sr, self._voiceprint)
            self.status["voice_frames"] = len(self._target_cos)
            self.status["prompt"] = f"Got you! {len(self._target_cos)} voice frames."
        elif phase in ("background", "world"):
            if self._voiceprint:
                cos = cal.cosines_to(all_audio, sr, self._voiceprint)
                self._bg_cos.extend(cos)
                self.status["bg_frames"] = len(self._bg_cos)
                self.status["prompt"] = f"Learned background ({len(self._bg_cos)} frames)."

    def _do_fit(self):
        from ..calibrate import Calibrator, _dprime_error_pct

        self.status["phase"] = "fitting"
        cal = Calibrator(self.pipe)
        model, dprime = cal.fit(self._target_cos, self._bg_cos)
        cal.save(model)
        err = _dprime_error_pct(dprime)
        quality = "good" if dprime >= 2.5 else ("fair" if dprime >= 1.5 else "poor")
        self.status.update({
            "phase": "done", "dprime": round(dprime, 2), "error_pct": err,
            "model": asdict(model), "quality": quality,
            "prompt": f"Done! d′ = {dprime:.2f} ({err})",
        })


class _State:
    def __init__(self, live: bool = False):
        self.live = live
        self.source = "live" if live else "synthetic"
        self.pipe = Pipeline(config=Config(), array=(None if live else MACBOOK_3))
        if not live:
            self.pipe.enroll("alan", 3, source="synthetic")
        self.pipe.warmup()
        self._last = None
        self._busy = False
        self.cal = _CalibrationSession(self)

    def awareness(self) -> dict:
        if self._busy and self._last is not None:
            return self._last
        self._busy = True
        try:
            amap = self.pipe.listen(duration=1.0, source=self.source)
            amap["wake_word"] = self.pipe.config.wake_word
            amap["accumulator"] = {
                "sources": len(self.pipe.accumulator.sources),
                "top_llr": self.pipe.enrollment.top_llr(),
            }
            self._last = amap
            return amap
        finally:
            self._busy = False


def _make_handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def _send(self, code: int, body, ctype="application/json"):
            try:
                data = body.encode() if isinstance(body, str) else body
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self._send(200, (_WEB / "index.html").read_text(), "text/html; charset=utf-8")
            if u.path == "/awareness":
                return self._send(200, json.dumps(state.awareness()))
            if u.path == "/calibrate/status":
                return self._send(200, json.dumps(state.cal.status))
            if u.path == "/probe":
                sid = (parse_qs(u.query).get("id") or ["?"])[0]
                return self._send(200, json.dumps({
                    "id": sid, "implemented": False,
                    "todo": "local STT (whisper.cpp) not wired yet",
                }))
            return self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            u = urlparse(self.path)
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len) if content_len else b""
            if u.path == "/calibrate/start":
                params = json.loads(body) if body else {}
                state.cal.start(
                    name=params.get("name", "you"),
                    seconds=params.get("seconds", 20.0),
                    others=params.get("others", True),
                )
                return self._send(200, json.dumps({"ok": True}))
            if u.path == "/calibrate/phase":
                state.cal.advance()
                return self._send(200, json.dumps({"ok": True}))
            return self._send(404, json.dumps({"error": "not found"}))

    return Handler


def serve(port: int = 8770, live: bool = False, host: str = "127.0.0.1") -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    state = _State(live=live)
    httpd = ThreadingHTTPServer((host, port), _make_handler(state))
    scene = "live mic" if live else "synthetic scene"
    print(f"Daredevil HUD → http://{host}:{port}   ({scene})")
    print("on-device · no cloud · Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
        httpd.shutdown()
    return 0
