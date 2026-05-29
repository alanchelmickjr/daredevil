"""Local web HUD server — stdlib only, binds to localhost, no cloud.

Serves the neumorphic-steampunk orbital HUD and streams awareness maps:
  GET /            -> the HUD page
  GET /awareness   -> the current awareness map (JSON)
  GET /probe?id=X  -> transcribe source X (local STT) and route it to the LLM

`daredevil serve` launches it. STT/LLM are local-only by design; /probe is an
explicit TODO (returns implemented:false) until whisper.cpp is wired in — it never
fakes a transcript.
"""
from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from ..config import Config
from ..pipeline import Pipeline
from ..stage1.mic_arrays import MACBOOK_3

_WEB = Path(__file__).parent / "web"


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

    def awareness(self) -> dict:
        if self._busy and self._last is not None:
            return self._last
        self._busy = True
        try:
            amap = self.pipe.listen(duration=1.0, source=self.source)
            amap["wake_word"] = self.pipe.config.wake_word
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

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self._send(200, (_WEB / "index.html").read_text(), "text/html; charset=utf-8")
            if u.path == "/awareness":
                return self._send(200, json.dumps(state.awareness()))
            if u.path == "/probe":
                sid = (parse_qs(u.query).get("id") or ["?"])[0]
                return self._send(200, json.dumps({
                    "id": sid, "implemented": False,
                    "todo": "local STT (whisper.cpp) not wired yet",
                }))
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
