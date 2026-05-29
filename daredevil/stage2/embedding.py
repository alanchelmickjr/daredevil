"""Slot A — Speaker embedding (WHO). The headline capability.

Reference backend: SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`,
Apache-2.0), 192-dim voiceprint. Fallback backend: a deterministic spectral
fingerprint so enroll -> identify still demonstrates end-to-end with zero deps.

A voiceprint is a non-reversible 192-dim vector. Raw audio is never stored.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .base import Slot
from ..audio.utils import fingerprint
from ..config import default_data_dir


class EmbeddingSlot(Slot):
    name = "embedding"

    def __init__(self, dim: int = 192, backend: str = "auto"):
        super().__init__()
        self.dim = dim
        self._requested = backend
        self._classifier = None
        self._backend_name = "fallback"

    @property
    def backend(self) -> str:
        return self._backend_name

    def warmup(self) -> None:
        if self._warm:
            return
        if self._requested != "fallback":
            try:
                import torch  # noqa: F401
                from speechbrain.inference.speaker import EncoderClassifier

                savedir = str(default_data_dir() / "models" / "ecapa")
                self._classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb", savedir=savedir
                )
                self.dim = 192
                self._backend_name = "reference"
            except Exception:
                self._classifier = None
                self._backend_name = "fallback"
        self._warm = True

    def run(self, audio: List[float], sr: int, ctx: Optional[Dict] = None) -> dict:
        if self._classifier is not None:
            try:
                import torch

                wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                emb = self._classifier.encode_batch(wav).squeeze().detach().cpu().tolist()
                if isinstance(emb, float):
                    emb = [emb]
                return {"vector": emb, "dim": len(emb), "backend": "reference"}
            except Exception:
                pass
        vec = fingerprint(audio, sr, self.dim)
        return {"vector": vec, "dim": self.dim, "backend": "fallback"}
