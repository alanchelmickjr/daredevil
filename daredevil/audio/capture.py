"""Audio capture — auto-detects the mic array; falls back to a synthetic scene.

Key symmetry: the hardware module presents itself as a USB-C UAC2 multichannel
PCM device, so the live-capture path is identical for laptop mics and the module
— only the channel count and coordinate map differ.

When no microphone (or no `sounddevice`) is present, we synthesize a deterministic
multi-source scene so the full pipeline — and the demo — still runs. Synthetic
captures carry `scene_truth`, which downstream stages use as perfect "separation"
so the architecture exercised is identical to the live path.
"""
from __future__ import annotations

import hashlib
import math
import random
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .utils import to_mono
from ..stage1.mic_arrays import MicArray, detect as detect_array, MACBOOK_3, SINGLE

__all__ = [
    "CaptureResult",
    "capture",
    "capture_live",
    "capture_file",
    "synthetic_scene",
    "synthetic_voice",
    "DEFAULT_SCENE",
]


@dataclass
class CaptureResult:
    channels: List[List[float]]
    sample_rate: int
    array: MicArray
    synthetic: bool = False
    scene_truth: Optional[List[dict]] = None
    source: str = "unknown"

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def mono(self) -> List[float]:
        return to_mono(self.channels)

    @property
    def duration(self) -> float:
        if not self.channels or not self.channels[0]:
            return 0.0
        return len(self.channels[0]) / self.sample_rate


# Default demo scene: an enrolled speaker (calm) + an unknown safety-critical baby cry.
# Mirrors the awareness-map example in the spec.
DEFAULT_SCENE = [
    {"name": "alan", "enrolled": True, "class": "speech",
     "azimuth": 0.0, "elevation": 0.0, "prosody_state": "calm", "distress": 0.12},
    {"name": None, "enrolled": False, "class": "baby_cry",
     "azimuth": 45.0, "elevation": 10.0, "prosody_state": "distressed", "distress": 0.95},
    # the "radio": heard, tracked, structured — but the attention gate keeps it out
    # of the conversation. This is the source that proves the point.
    {"name": None, "enrolled": False, "class": "music",
     "azimuth": 270.0, "elevation": 0.0, "prosody_state": "calm", "distress": 0.05},
]


def _seed_for(label: str) -> int:
    return int(hashlib.sha1((label or "anon").encode()).hexdigest(), 16) % 1_000_000


def _synth_source(label: str, klass: str, seconds: float, sr: int) -> List[float]:
    """Deterministic, recognizable-per-label waveform for a given event class."""
    rng = random.Random(_seed_for(label) ^ _seed_for(klass))
    n = int(seconds * sr)
    out = [0.0] * n

    if klass in ("speech",):
        f0 = 110 + rng.random() * 80          # speaker-specific pitch
        formants = [f0, f0 * 2.4, f0 * 3.1, f0 * 4.6]
        amps = [1.0, 0.5, 0.35, 0.2]
    elif klass in ("baby_cry", "infant_cry"):
        f0 = 450 + rng.random() * 120          # high, piercing
        formants = [f0, f0 * 1.9, f0 * 2.7]
        amps = [1.0, 0.7, 0.5]
    elif klass in ("alarm", "siren", "smoke_detector"):
        f0 = 1000 + rng.random() * 400
        formants = [f0, f0 * 1.5]
        amps = [1.0, 0.6]
    elif klass in ("music",):
        f0 = 220.0
        formants = [f0, f0 * 1.26, f0 * 1.5]   # a chord
        amps = [0.8, 0.8, 0.8]
    else:
        formants, amps = [], []

    for i in range(n):
        t = i / sr
        # slow envelope (syllabic for speech, wail for cry, beeping for alarm)
        if klass in ("alarm", "siren", "smoke_detector"):
            env = 1.0 if (int(t * 2) % 2 == 0) else 0.05
        elif klass in ("baby_cry", "infant_cry"):
            env = 0.6 + 0.4 * math.sin(2 * math.pi * 3.0 * t)
        elif klass == "speech":
            env = 0.5 + 0.5 * abs(math.sin(2 * math.pi * 3.5 * t))
        else:
            env = 1.0
        s = 0.0
        for f, a in zip(formants, amps):
            s += a * math.sin(2 * math.pi * f * t)
        out[i] = env * s
    # normalize + light noise so enroll/listen aren't bit-identical
    peak = max((abs(v) for v in out), default=1.0) or 1.0
    noise = 0.02
    return [max(-1.0, min(1.0, v / peak + rng.uniform(-noise, noise))) for v in out]


def synthetic_voice(name: str, seconds: float = 3.0, sr: int = 16000,
                    array: Optional[MicArray] = None) -> CaptureResult:
    """A single enrolled-style speaker — used for synthetic enrollment."""
    sig = _synth_source(name, "speech", seconds, sr)
    arr = array or SINGLE
    return CaptureResult(
        channels=[sig], sample_rate=sr, array=arr, synthetic=True,
        scene_truth=[{"name": name, "class": "speech", "audio": sig,
                      "azimuth": 0.0, "elevation": 0.0,
                      "prosody_state": "calm", "distress": 0.1, "enrolled": True}],
        source="synthetic-voice",
    )


def synthetic_scene(seconds: float = 1.0, sr: int = 16000,
                    array: Optional[MicArray] = None,
                    sources: Optional[List[dict]] = None) -> CaptureResult:
    """A deterministic multi-source scene (clearly labeled synthetic in the demo)."""
    arr = array or MACBOOK_3
    spec = sources or DEFAULT_SCENE
    truth = []
    mix_len = int(seconds * sr)
    mix = [0.0] * mix_len
    for s in spec:
        label = s.get("name") or f"src-{s.get('class','x')}"
        sig = _synth_source(label, s["class"], seconds, sr)
        for i in range(min(mix_len, len(sig))):
            mix[i] += 0.6 * sig[i]
        t = dict(s)
        t["audio"] = sig
        truth.append(t)
    # broadband noise floor so the scene reads like a real (noisy) room. This only
    # affects the mix used for visualization / the real-audio path — synthetic slots
    # read each source's clean audio from scene_truth, so identity is unaffected.
    nrng = random.Random(1337)
    noise = 0.05
    mix = [max(-1.0, min(1.0, v + nrng.uniform(-noise, noise))) for v in mix]

    # replicate the mix across channels (per-source audio is carried in scene_truth)
    channels = [list(mix) for _ in range(max(1, arr.n_mics))]
    return CaptureResult(channels=channels, sample_rate=sr, array=arr,
                         synthetic=True, scene_truth=truth, source="synthetic-scene")


def capture_file(path: str, sr_hint: Optional[int] = None) -> CaptureResult:
    """Read a WAV file (stdlib `wave`); uses `soundfile` if present for other formats."""
    p = Path(path)
    try:
        import soundfile as sf  # optional, handles flac/ogg/etc.
        data, sr = sf.read(str(p), always_2d=True)
        channels = [data[:, c].tolist() for c in range(data.shape[1])]
        return CaptureResult(channels=channels, sample_rate=int(sr),
                             array=detect_array(channels=len(channels)),
                             source=str(p))
    except Exception:
        pass
    with wave.open(str(p), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sampwidth = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    # decode PCM (8/16/32-bit) to float [-1,1]
    import struct
    if sampwidth == 2:
        fmt = "<%dh" % (len(frames) // 2)
        ints = struct.unpack(fmt, frames)
        scale = 32768.0
    elif sampwidth == 4:
        fmt = "<%di" % (len(frames) // 4)
        ints = struct.unpack(fmt, frames)
        scale = 2147483648.0
    elif sampwidth == 1:
        ints = [b - 128 for b in frames]
        scale = 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")
    flat = [v / scale for v in ints]
    channels = [flat[c::nch] for c in range(nch)]
    return CaptureResult(channels=channels, sample_rate=sr,
                         array=detect_array(channels=nch), source=str(p))


def _pick_input_device():
    """Find the first input device that actually delivers audio.

    AirPods and some Bluetooth devices show up as the default input but return
    zeros when not in a call/voice profile.  Fall back to a device that has a
    native sample rate >= 44.1 kHz (rules out dead BT SCO endpoints).
    """
    import sounddevice as sd
    default_idx = sd.default.device[0]
    devs = sd.query_devices()
    default = devs[default_idx] if default_idx is not None else None
    if default and default["max_input_channels"] > 0 and default["default_samplerate"] >= 44100:
        return default_idx, int(default["default_samplerate"])
    # Default looks suspect (e.g. BT at 24kHz) — find the built-in mic.
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0 and d["default_samplerate"] >= 44100:
            return i, int(d["default_samplerate"])
    # Last resort: use whatever the OS says, at its native rate.
    if default and default["max_input_channels"] > 0:
        return default_idx, int(default["default_samplerate"])
    return None, 48000


class MicStream:
    """Persistent mic stream — stays open, no flashing. diart/pyannote pattern.

    The callback pushes chunks into a queue. read(seconds) pulls from the queue.
    Mic opens once on first read and stays open until close().
    """

    _instance = None

    @classmethod
    def get(cls) -> "MicStream":
        if cls._instance is None or not cls._instance._open:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        import sounddevice as sd
        from queue import SimpleQueue
        self._queue: "SimpleQueue" = SimpleQueue()
        dev_idx, native_sr = _pick_input_device()
        self.sr = native_sr
        self._stream = sd.InputStream(
            channels=1,
            samplerate=self.sr,
            blocksize=int(0.1 * self.sr),
            device=dev_idx,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self._open = True

    def _callback(self, indata, frames, time_info, status):
        self._queue.put_nowait(indata[:, 0].copy())

    def read(self, seconds: float) -> List[float]:
        import time
        n_samples = int(seconds * self.sr)
        buf: List[float] = []
        deadline = time.monotonic() + seconds + 0.5
        while len(buf) < n_samples and time.monotonic() < deadline:
            try:
                chunk = self._queue.get(timeout=0.2)
                buf.extend(chunk.tolist())
            except Exception:
                pass
        return buf[:n_samples]

    def close(self):
        if self._open:
            self._stream.stop()
            self._stream.close()
            self._open = False


def capture_live(seconds: float = 1.0, sr: int = 48000,
                 array: Optional[MicArray] = None) -> CaptureResult:
    """Record from the persistent mic stream (always open, no flashing)."""
    arr = array or detect_array()
    try:
        mic = MicStream.get()
        samples = mic.read(seconds)
        return CaptureResult(channels=[samples], sample_rate=mic.sr, array=arr, source="live")
    except Exception:
        import sounddevice as sd
        nch = max(1, arr.n_mics)
        dev_idx, native_sr = _pick_input_device()
        use_sr = native_sr if native_sr >= 44100 else sr
        rec = sd.rec(int(seconds * use_sr), samplerate=use_sr, channels=nch,
                     device=dev_idx, dtype="float32")
        sd.wait()
        channels = [rec[:, c].tolist() for c in range(nch)]
        return CaptureResult(channels=channels, sample_rate=use_sr, array=arr, source="live")


def capture(seconds: float = 1.0, sr: int = 48000, source: str = "auto",
            file: Optional[str] = None, array: Optional[MicArray] = None,
            name: Optional[str] = None, scene: Optional[List[dict]] = None) -> CaptureResult:
    """Dispatch to the right capture path.

    source: "auto" | "live" | "file" | "synthetic"
      - auto: live mic if available, else synthetic scene.
      - scene: optional custom source list for the synthetic scene (e.g. a crowd).
    """
    if source == "file" or (source == "auto" and file):
        if not file:
            raise ValueError("source='file' requires a file path")
        return capture_file(file, sr)
    if source == "synthetic":
        if name:
            return synthetic_voice(name, seconds, sr, array)
        return synthetic_scene(seconds, sr, array, sources=scene)
    if source == "live":
        return capture_live(seconds, sr, array)
    # auto
    try:
        return capture_live(seconds, sr, array)
    except Exception:
        if name:
            return synthetic_voice(name, seconds, sr, array)
        return synthetic_scene(seconds, sr, array, sources=scene)
