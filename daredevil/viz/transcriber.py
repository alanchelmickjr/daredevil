"""Transcriber — independent speech-to-text consumer. Never blocks detection.

Buffers audio per source ID while they're speaking. Flushes to whisper-cli
when they pause. If whisper fails, returns empty string. Detection doesn't
know this exists.
"""
from __future__ import annotations

import logging
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("daredevil.transcribe")


class Transcriber:
    """Per-source speech buffer with flush-on-pause."""

    def __init__(self, model_path: Optional[str] = None,
                 vad_threshold: float = 0.012):
        self._model = model_path or str(
            Path.home() / ".daredevil/models/whisper/ggml-base.en.bin")
        self._vad = vad_threshold
        self._buffers: Dict[str, List[float]] = {}
        self._sr: Dict[str, int] = {}
        self._active: Dict[str, bool] = {}
        self._latest: Optional[Dict[str, str]] = None

    def feed(self, source_id: str, audio: List[float], sr: int,
             is_speaking: bool) -> None:
        if is_speaking:
            if source_id not in self._buffers:
                self._buffers[source_id] = []
            self._buffers[source_id].extend(audio)
            self._sr[source_id] = sr
            self._active[source_id] = True
        else:
            if self._active.get(source_id):
                self._active[source_id] = False

    def check_pauses(self) -> List[Tuple[str, str]]:
        results = []
        done = []
        for sid, active in self._active.items():
            if not active and sid in self._buffers and self._buffers[sid]:
                text = self._flush(sid)
                if text:
                    results.append((sid, text))
                    self._latest = {"source": sid, "text": text}
                done.append(sid)
        for sid in done:
            self._buffers.pop(sid, None)
            self._sr.pop(sid, None)
        return results

    def has_text(self) -> bool:
        return self._latest is not None

    def latest(self) -> Optional[Dict[str, str]]:
        r = self._latest
        self._latest = None
        return r

    def _flush(self, source_id: str) -> str:
        audio = self._buffers.get(source_id, [])
        sr = self._sr.get(source_id, 16000)
        if not audio:
            return ""
        if not Path(self._model).exists():
            log.warning("whisper model not found")
            return ""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            with wave.open(tmp.name, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                frames = struct.pack(
                    "<%dh" % len(audio),
                    *[max(-32768, min(32767, int(s * 32767))) for s in audio])
                w.writeframes(frames)
            result = subprocess.run(
                ["whisper-cli", "--model", self._model, "--file", tmp.name,
                 "--no-timestamps", "--no-prints"],
                capture_output=True, text=True, timeout=10)
            text = result.stdout.strip()
            if text:
                log.info(f"transcribe {source_id}: {len(audio)/sr:.1f}s → \"{text}\"")
            return text
        except Exception as e:
            log.warning(f"transcribe {source_id} failed: {e}")
            return ""
        finally:
            Path(tmp.name).unlink(missing_ok=True)
