"""Slot A — Speaker embedding (WHO). The headline capability.

Reference backend: SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`,
Apache-2.0), 192-dim voiceprint. ONNX backend: the same ECAPA-TDNN exported to
ONNX format via `scripts/export_ecapa_onnx.py`, runnable with onnxruntime on
ARM/edge devices (e.g. Jetson) where PyTorch won't install. Fallback backend: a
deterministic spectral fingerprint so enroll -> identify still demonstrates
end-to-end with zero deps.

Backend priority at warmup:
  1. torch + speechbrain  -> "reference"
  2. onnxruntime          -> "onnx"
  3. pure-Python heuristic -> "fallback"

A voiceprint is a non-reversible 192-dim vector. Raw audio is never stored.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .base import Slot
from ..audio.utils import fingerprint, resample
from ..config import default_data_dir

log = logging.getLogger("daredevil.embedding")

# Path within the data dir where the exported ONNX model is expected.
_ONNX_MODEL_SUBPATH = "models/ecapa/ecapa_tdnn.onnx"


class EmbeddingSlot(Slot):
    name = "embedding"

    def __init__(self, dim: int = 192, backend: str = "auto"):
        super().__init__()
        self.dim = dim
        self._requested = backend
        self._classifier = None       # SpeechBrain EncoderClassifier (reference)
        self._ort_session = None       # onnxruntime InferenceSession (onnx)
        self._ort_input_name = None    # cached input tensor name for the ONNX graph
        self._backend_name = "fallback"

    @property
    def backend(self) -> str:
        return self._backend_name

    # ------------------------------------------------------------------
    # warmup — three-tier cascade
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        if self._warm:
            return
        if self._requested != "fallback":
            # Tier 1: torch + speechbrain (full reference)
            if self._requested in ("auto", "reference"):
                if self._try_reference():
                    self._warm = True
                    return

            # Tier 2: ONNX Runtime (portable, no torch needed)
            if self._requested in ("auto", "onnx"):
                if self._try_onnx():
                    self._warm = True
                    return

        # Tier 3: pure-Python spectral fingerprint (always available)
        self._backend_name = "fallback"
        log.info("embedding backend: fallback (spectral fingerprint)")
        self._warm = True

    def _try_reference(self) -> bool:
        """Attempt to load the SpeechBrain ECAPA-TDNN torch backend."""
        try:
            import torch  # noqa: F401
            from speechbrain.inference.speaker import EncoderClassifier

            savedir = str(default_data_dir() / "models" / "ecapa")
            self._classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", savedir=savedir
            )
            self.dim = 192
            self._backend_name = "reference"
            log.info("embedding backend: reference (SpeechBrain ECAPA-TDNN)")
            return True
        except Exception as e:
            log.debug("reference backend unavailable: %s", e)
            self._classifier = None
            return False

    def _try_onnx(self) -> bool:
        """Attempt to load the ONNX-exported ECAPA model with onnxruntime."""
        try:
            import onnxruntime as ort  # lazy — onnxruntime is optional

            model_path = default_data_dir() / _ONNX_MODEL_SUBPATH
            if not model_path.exists():
                return False

            self._ort_session = ort.InferenceSession(
                str(model_path),
                providers=ort.get_available_providers(),
            )
            self._ort_input_name = self._ort_session.get_inputs()[0].name
            self.dim = 192
            self._backend_name = "onnx"
            log.info("embedding backend: onnx (%s)", model_path)
            return True
        except Exception as e:
            log.debug("onnx backend unavailable: %s", e)
            self._ort_session = None
            self._ort_input_name = None
            return False

    # ------------------------------------------------------------------
    # run — dispatch to whichever backend warmed up
    # ------------------------------------------------------------------

    def run(self, audio: List[float], sr: int, ctx: Optional[Dict] = None) -> dict:
        # Tier 1: torch reference
        if self._classifier is not None:
            try:
                import torch

                sig = torch.tensor(audio, dtype=torch.float32)
                sig = self._classifier.audio_normalizer(sig, sr)
                emb = self._classifier.encode_batch(sig.unsqueeze(0), normalize=True)
                emb = emb.squeeze().detach().cpu().tolist()
                if isinstance(emb, float):
                    emb = [emb]
                return {"vector": emb, "dim": len(emb), "backend": "reference"}
            except Exception as e:
                log.debug("reference inference failed, falling through: %s", e)

        # Tier 2: ONNX Runtime
        if self._ort_session is not None:
            try:
                return self._run_onnx(audio, sr)
            except Exception as e:
                log.debug("onnx inference failed, falling through: %s", e)

        # Tier 3: deterministic heuristic
        vec = fingerprint(audio, sr, self.dim)
        return {"vector": vec, "dim": self.dim, "backend": "fallback"}

    def _run_onnx(self, audio: List[float], sr: int) -> dict:
        """Run inference through the ONNX Runtime session.

        The exported model expects a float32 tensor of shape [1, num_samples]
        at 16 kHz. The output is a 192-dim embedding that we L2-normalise to
        unit length (matching the reference backend's normalize=True).
        """
        import numpy as np  # onnxruntime already requires numpy

        # Resample to 16 kHz if needed (the model's training rate)
        if sr != 16000:
            audio = resample(audio, sr, 16000)

        waveform = np.array(audio, dtype=np.float32).reshape(1, -1)
        outputs = self._ort_session.run(None, {self._ort_input_name: waveform})
        emb = outputs[0].squeeze()

        # L2-normalise to unit length
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm

        emb_list = emb.tolist()
        if isinstance(emb_list, float):
            emb_list = [emb_list]
        return {"vector": emb_list, "dim": len(emb_list), "backend": "onnx"}
