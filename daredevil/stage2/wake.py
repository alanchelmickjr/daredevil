"""Wake word — the system's *name*, the second channel of natural attention.

Humans grab each other's attention two ways. First by the **sound** of the voice
— urgency, distress, loudness (that is the prosody slot + the router's DISTRESS
escalation). When the sound alone doesn't land, we use the person's **name**. This
module is that second channel: it lets Daredevil hear *its own name* and wake to
whoever called it — and, because WHO runs on the very same window, it knows who is
calling. Name grabs attention; identity says who's asking.

Two backends, same contract (the project's graceful-degradation pattern):

* **reference** — openWakeWord (Apache-2.0, fully on-device ONNX, *no AccessKey,
  no cloud, no metering*). Optional; used only when a model name is configured and
  the package is installed. Auto-upgrades in ``warmup()``.
* **template** (always works, stdlib only) — *query-by-example keyword spotting*:
  you say the name a few times, we keep a short spectral contour of it, and detect
  by sub-sequence Dynamic Time Warping of the live window against that contour.
  This is a real, classic KWS algorithm — not a stub — and it is inherently
  speaker-personal: the system wakes for the people who taught it its name.

Privacy: we persist only the lossy band-energy contour of the phrase (a few
hundred small floats), never raw audio. It cannot be played back as speech.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..audio.utils import resample, rms

try:  # optional acceleration, mirrors audio.utils
    import numpy as _np
except Exception:  # pragma: no cover - numpy is optional
    _np = None


# --- feature extraction: a tiny per-frame log-band spectral contour -----------

def _hann(n: int) -> List[float]:
    if n <= 1:
        return [1.0] * max(1, n)
    return [0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1)) for i in range(n)]


def _log_freqs(n_bands: int, f_lo: float, f_hi: float) -> List[float]:
    if n_bands <= 1:
        return [f_lo]
    return [f_lo * (f_hi / f_lo) ** (k / (n_bands - 1)) for k in range(n_bands)]


def _band_vector(frame: Sequence[float], sr: int, freqs: List[float],
                 win: List[float]) -> Optional[List[float]]:
    """One frame -> L2-normalised log-magnitude across ``freqs`` (spectral *shape*,
    loudness-invariant). Returns None for a silent frame (no usable shape)."""
    n = len(frame)
    if _np is not None:
        xv = _np.asarray(frame, dtype=float) * _np.asarray(win[:n], dtype=float)
        idx = _np.arange(n)
        mags = []
        for f in freqs:
            ang = -2.0 * math.pi * f * idx / sr
            re = float(xv.dot(_np.cos(ang)))
            im = float(xv.dot(_np.sin(ang)))
            mags.append(math.hypot(re, im))
    else:
        xs = [frame[i] * win[i] for i in range(n)]
        mags = []
        for f in freqs:
            w0 = -2.0 * math.pi * f / sr
            re = im = 0.0
            for i in range(n):
                a = w0 * i
                re += xs[i] * math.cos(a)
                im += xs[i] * math.sin(a)
            mags.append(math.hypot(re, im))
    feat = [math.log1p(m) for m in mags]
    norm = math.sqrt(sum(v * v for v in feat))
    if norm <= 1e-9:
        return None
    return [v / norm for v in feat]


def feature_sequence(audio: Sequence[float], sr: int, p) -> List[List[float]]:
    """Audio -> sequence of band vectors, with leading/trailing silence trimmed so
    the contour is the spoken phrase, not the room before and after it."""
    xa = resample(list(audio), sr, p.analysis_sr)
    sr2 = p.analysis_sr
    fn = max(8, int(p.frame_ms / 1000.0 * sr2))
    hn = max(1, int(p.hop_ms / 1000.0 * sr2))
    freqs = _log_freqs(p.n_bands, 80.0, sr2 / 2.0 * 0.95)
    win = _hann(fn)
    seq: List[Optional[List[float]]] = []
    energies: List[float] = []
    i = 0
    while i + fn <= len(xa):
        frame = xa[i:i + fn]
        energies.append(rms(frame))
        seq.append(_band_vector(frame, sr2, freqs, win))
        i += hn
    if not energies:
        return []
    peak = max(energies) or 1.0
    thr = peak * 0.15  # voice-activity trim relative to the loudest frame
    keep = [k for k, e in enumerate(energies) if e >= thr and seq[k] is not None]
    if not keep:
        return [v for v in seq if v is not None]
    lo, hi = keep[0], keep[-1]
    return [seq[k] for k in range(lo, hi + 1) if seq[k] is not None]


def _cost(a: Sequence[float], b: Sequence[float]) -> float:
    """Local cost = 1 - cosine. Band vectors are unit-norm & non-negative, so this
    lands in [0, 1]: 0 = identical spectral shape, 1 = orthogonal."""
    n = min(len(a), len(b))
    dot = 0.0
    for k in range(n):
        dot += a[k] * b[k]
    return 1.0 - dot


def subsequence_dtw(query: List[List[float]], obs: List[List[float]]) -> float:
    """Best alignment of ``query`` (the enrolled name) to *any* sub-stretch of
    ``obs`` (the live window), returned as a similarity in [0, 1].

    Sub-sequence DTW: the query may begin at any observation frame (free start, so
    the name can occur anywhere in the window) and time-warp to match. Cost is
    accumulated and normalised by the query length — the standard QbE-KWS score.
    """
    I, J = len(query), len(obs)
    if I == 0 or J == 0:
        return 0.0
    # i = 0 row: free start across all observation frames.
    prev = [_cost(query[0], obs[j]) for j in range(J)]
    for i in range(1, I):
        qi = query[i]
        cur = [0.0] * J
        cur[0] = prev[0] + _cost(qi, obs[0])
        for j in range(1, J):
            m = prev[j]
            if prev[j - 1] < m:
                m = prev[j - 1]
            if cur[j - 1] < m:
                m = cur[j - 1]
            cur[j] = _cost(qi, obs[j]) + m
        prev = cur
    best = min(prev)
    sim = 1.0 - best / I
    return 0.0 if sim < 0.0 else (1.0 if sim > 1.0 else sim)


def _slug(phrase: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (phrase or "wake").strip().lower()).strip("_")
    return s or "wake"


class WakeWordDetector:
    """Hears the system's name. ``template`` backend always works; ``openwakeword``
    upgrades it when a model is configured and installed."""

    def __init__(self, params, data_dir, phrase: str = "Hey Radar"):
        self.p = params
        self.phrase = phrase or "Hey Radar"
        self.dir = Path(data_dir) / "wakewords"
        self._templates: Dict[str, dict] = {}
        self._backend = "template"
        self._oww = None
        self._warm = False
        self._load()

    @property
    def backend(self) -> str:
        return self._backend

    def enrolled(self) -> bool:
        return bool(self._templates)

    def ready(self) -> bool:
        """Can we actually detect anything right now?"""
        return self._oww is not None or self.enrolled()

    # --- optional reference backend ------------------------------------------
    def warmup(self) -> None:
        if self._warm:
            return
        model = getattr(self.p, "oww_model", "") or ""
        if model:
            try:  # lazy + guarded; never breaks the template path
                from openwakeword.model import Model
                self._oww = Model(wakeword_models=[model])
                self._backend = "openwakeword"
            except Exception:
                self._oww = None
        self._warm = True

    def _detect_oww(self, audio: Sequence[float], sr: int) -> float:
        import numpy as np
        x = resample(list(audio), sr, 16000)
        pcm = np.clip(np.asarray(x, dtype=float) * 32767.0, -32768, 32767).astype("int16")
        try:
            self._oww.reset()
        except Exception:
            pass
        step = 1280  # 80 ms @ 16 kHz, openWakeWord's frame size
        best = 0.0
        for i in range(0, max(0, len(pcm) - step + 1), step):
            scores = self._oww.predict(pcm[i:i + step])
            if scores:
                best = max(best, max(float(v) for v in scores.values()))
        return best

    # --- enrollment -----------------------------------------------------------
    def enroll(self, audio: Sequence[float], sr: int, phrase: Optional[str] = None) -> dict:
        """Teach (or reinforce) the name from one spoken example."""
        phrase = phrase or self.phrase
        seq = feature_sequence(audio, sr, self.p)
        min_frames = max(2, int(self.p.min_phrase_s / (self.p.hop_ms / 1000.0)))
        if len(seq) < min_frames:
            return {"phrase": phrase, "ok": False, "frames": len(seq),
                    "reason": "too short / too quiet — say the name clearly"}
        slug = _slug(phrase)
        rec = self._templates.get(slug) or {"phrase": phrase, "sequences": [],
                                             "n_bands": self.p.n_bands,
                                             "hop_ms": self.p.hop_ms,
                                             "frame_ms": self.p.frame_ms,
                                             "analysis_sr": self.p.analysis_sr}
        rec["sequences"].append([[round(v, 4) for v in f] for f in seq])
        self._templates[slug] = rec
        self._save(slug, rec)
        return {"phrase": phrase, "ok": True, "frames": len(seq),
                "samples": len(rec["sequences"]), "backend": "template"}

    # --- detection ------------------------------------------------------------
    def detect(self, audio: Sequence[float], sr: int) -> dict:
        out = {"detected": False, "score": 0.0, "phrase": self.phrase,
               "backend": self._backend, "enrolled": self.enrolled()}
        if not getattr(self.p, "enabled", True):
            out["backend"] = "disabled"
            return out
        if self._oww is not None:
            try:
                s = self._detect_oww(audio, sr)
                out.update(detected=s >= self.p.oww_threshold, score=round(s, 3),
                           backend="openwakeword")
                return out
            except Exception:
                pass  # fall through to the always-works template path
        if not self._templates:
            return out
        obs = feature_sequence(audio, sr, self.p)
        best, best_phrase = 0.0, None
        for rec in self._templates.values():
            for seq in rec["sequences"]:
                if len(obs) < max(2, int(len(seq) * 0.4)):
                    continue  # window can't contain a phrase this long
                sim = subsequence_dtw(seq, obs)
                if sim > best:
                    best, best_phrase = sim, rec["phrase"]
        out.update(detected=best >= self.p.threshold, score=round(best, 3),
                   phrase=best_phrase or self.phrase, backend="template")
        return out

    # --- persistence (lossy contour only; never raw audio) --------------------
    def _load(self) -> None:
        if not self.dir.exists():
            return
        for f in self.dir.glob("*.json"):
            try:
                rec = json.loads(f.read_text())
                if rec.get("sequences"):
                    self._templates[f.stem] = rec
            except Exception:
                continue

    def _save(self, slug: str, rec: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{slug}.json").write_text(json.dumps(rec))

    def forget(self, phrase: Optional[str] = None) -> None:
        slug = _slug(phrase or self.phrase)
        self._templates.pop(slug, None)
        try:
            (self.dir / f"{slug}.json").unlink()
        except Exception:
            pass
