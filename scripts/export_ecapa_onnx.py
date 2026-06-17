#!/usr/bin/env python3
"""Export the SpeechBrain ECAPA-TDNN model to ONNX format.

Run this on a machine where torch + speechbrain are installed (e.g. a MacBook),
then copy the resulting .onnx file to the target device (e.g. Jetson) where only
onnxruntime is available.

Usage:
    python scripts/export_ecapa_onnx.py [--output PATH]

Default output: ~/.daredevil/models/ecapa/ecapa_tdnn.onnx

Requirements (install via `pip install daredevil[speaker]`):
    torch, torchaudio, speechbrain

The exported model accepts:
    Input:  "audio" — float32 tensor, shape [1, num_samples] at 16 kHz
    Output: 192-dim speaker embedding (un-normalised; the runtime L2-normalises)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SpeechBrain ECAPA-TDNN to ONNX for edge deployment."
    )
    default_out = Path.home() / ".daredevil" / "models" / "ecapa" / "ecapa_tdnn.onnx"
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(default_out),
        help=f"Output path for the .onnx file (default: {default_out})",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate the model expects (default: 16000)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Example waveform duration in seconds for tracing (default: 3.0)",
    )
    args = parser.parse_args()

    # --- Import heavy libs (only needed on the export machine) ----------------
    try:
        import torch
    except ImportError:
        print("ERROR: torch is required to export. Install with: pip install daredevil[speaker]",
              file=sys.stderr)
        sys.exit(1)

    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        print("ERROR: speechbrain is required to export. Install with: pip install daredevil[speaker]",
              file=sys.stderr)
        sys.exit(1)

    # --- Load the pretrained ECAPA-TDNN ---------------------------------------
    savedir = str(Path.home() / ".daredevil" / "models" / "ecapa")
    print(f"Loading SpeechBrain ECAPA-TDNN (savedir={savedir}) ...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir=savedir
    )
    model = classifier.mods["embedding_model"]
    model.eval()

    # --- Build a dummy input --------------------------------------------------
    num_samples = int(args.sample_rate * args.duration)
    dummy_wav = torch.randn(1, num_samples)
    print(f"Tracing with dummy input shape {list(dummy_wav.shape)} "
          f"({args.duration}s @ {args.sample_rate} Hz) ...")

    # The ECAPA embedding_model expects the raw waveform features processed
    # through the compute_features + mean_var_norm pipeline. We need to run
    # the feature extraction to get the correct input shape for the embedding
    # model. Instead, we wrap the full forward pass.
    class ECAPAWrapper(torch.nn.Module):
        """Wraps the SpeechBrain classifier to accept raw waveform [1, T]."""

        def __init__(self, clf):
            super().__init__()
            self.clf = clf

        def forward(self, wav: torch.Tensor) -> torch.Tensor:
            # wav: [batch, time] at 16 kHz
            # Use the same pipeline as encode_batch but without normalize
            # so the runtime can normalise itself.
            wav_lens = torch.tensor([1.0], device=wav.device)
            feats = self.clf.mods["compute_features"](wav)
            feats = self.clf.mods["mean_var_norm"](feats, wav_lens)
            emb = self.clf.mods["embedding_model"](feats)
            return emb.squeeze(1)  # [batch, 192]

    wrapper = ECAPAWrapper(classifier)
    wrapper.eval()

    # --- Verify the wrapper produces the same output --------------------------
    with torch.no_grad():
        ref_emb = classifier.encode_batch(dummy_wav, normalize=False)
        ref_emb = ref_emb.squeeze()
        wrapper_emb = wrapper(dummy_wav).squeeze()
        cos_sim = torch.nn.functional.cosine_similarity(
            ref_emb.unsqueeze(0), wrapper_emb.unsqueeze(0)
        ).item()
        print(f"Wrapper vs reference cosine similarity: {cos_sim:.6f}")
        if cos_sim < 0.999:
            print("WARNING: wrapper output diverges from reference — check the export.",
                  file=sys.stderr)

    # --- Export to ONNX -------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to ONNX (opset {args.opset}) ...")
    torch.onnx.export(
        wrapper,
        (dummy_wav,),
        str(output_path),
        input_names=["audio"],
        output_names=["embedding"],
        dynamic_axes={
            "audio": {0: "batch", 1: "num_samples"},
            "embedding": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path}  ({size_mb:.1f} MB)")

    # --- Optional: verify with onnxruntime ------------------------------------
    try:
        import onnxruntime as ort
        import numpy as np

        print("Verifying with onnxruntime ...")
        sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"audio": dummy_wav.numpy()})[0]
        ort_emb = torch.from_numpy(ort_out).squeeze()
        cos_ort = torch.nn.functional.cosine_similarity(
            ref_emb.unsqueeze(0), ort_emb.unsqueeze(0)
        ).item()
        print(f"ONNX vs reference cosine similarity: {cos_ort:.6f}")
        if cos_ort < 0.999:
            print("WARNING: ONNX output diverges — model may need re-export.",
                  file=sys.stderr)
        else:
            print("Verification passed.")
    except ImportError:
        print("onnxruntime not installed — skipping verification. "
              "Install with: pip install onnxruntime")

    print("\nDone. Copy the .onnx file to the target device at:")
    print(f"  ~/.daredevil/models/ecapa/ecapa_tdnn.onnx")


if __name__ == "__main__":
    main()
