# Models & the inference engine

How Daredevil does digital-signal conversion and runs inference — and why these
choices, given three constraints: **open source**, **local-only**, and **WHO first**.

## The engine: one slot-runtime, three backends

The patent's hardware runs int8 models in parallel across DSP cores. The software
analogue is a backend-agnostic **slot runtime** with three tiers:

| Backend | Purpose | Runs on |
|---|---|---|
| **Reference — PyTorch** | quickest path to working models; enrollment | Mac (MPS), Orin (CUDA), Thor (CUDA 13), CPU |
| **Portable — ONNX Runtime** *(next)* | one model file, int8, hardware EPs | Mac (CoreML/ANE), Orin (TensorRT), Thor (TensorRT 10.13), any CPU |
| **Fallback — pure stdlib** | heuristic slots; proves the architecture | literally anywhere |

**Thor compute budget:** On Jetson Thor (128GB LPDDR5X, 2070 TFLOPS Blackwell),
all 4 slots run reference backends simultaneously alongside a 70B LLM, Whisper
large-v3, and Kokoro TTS. This enables daredevil to run **in-process** as a
library import rather than a separate HTTP service — eliminating the 1s polling
delay and enabling real-time awareness (~100ms cycle) at 500-person scale.

**Why ONNX Runtime is the long-term target:** the *same* model artifact runs with
the CoreML execution provider on a MacBook (Apple Neural Engine) and the
TensorRT EP on an Orin, with int8 quantization — directly serving "runs everywhere
+ accelerates on device." We start on torch (ECAPA/PANNs ship as torch) and add an
ONNX export path.

**GIL honesty:** slots run in a `ThreadPoolExecutor`. This parallelizes because
torch/ONNX/numpy release the GIL during native inference. The pure-Python fallback
is GIL-bound; we never pretend otherwise — timing is always measured, and the
speedup widens once real backends are installed.

## Slot A — WHO (speaker identity) · the headline

| Model | License | Notes |
|---|---|---|
| **SpeechBrain ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`) | **Apache-2.0** ✅ | 192-dim, 41M downloads, *not gated* — **default** |
| NVIDIA TitaNet-Large | CC-BY-4.0 | higher accuracy, heavier NeMo dependency |
| pyannote/embedding | MIT but **gated** | install friction (HF login/accept) |

→ **ECAPA wins** for a permissive, frictionless, battle-tested default. Gate it
with **Silero VAD (MIT)** so we only embed actual speech. Enrollment confidence
follows `C(t)=1−e^(−t/3)`; match is cosine similarity, threshold `T=0.70`.

Fallback: a deterministic spectral fingerprint — *not* a real voiceprint, just
enough to make enroll→identify demonstrable with zero dependencies.

## Slot B — WHAT (acoustic events)

| Model | License | Notes |
|---|---|---|
| **PANNs CNN14** | Apache-2.0 / CC | 527 AudioSet classes, ~300MB — accuracy default |
| **YAMNet** | Apache-2.0 | ~4MB, ONNX/TFLite — the **edge** default (int8 story) |
| BEATs / AST | varies | transformer, higher accuracy, heavier |

AudioSet covers the safety-critical classes we care about (baby cry, alarms,
siren, glass breaking, gunshot). Recommendation: PANNs for accuracy, YAMNet for
edge/Orin. Fallback: a light spectral heuristic.

## Slot C — HOW (prosody / emotion) · the licensing snag

This replaces the previous **cloud** Hume API with **local** analysis. But the
obvious local choice has a license catch:

| Option | License | Verdict |
|---|---|---|
| **librosa (`pyin`) + custom jitter/shimmer/HNR** | **ISC** ✅ | **default** — clean for an MIT repo |
| Parselmouth / Praat | GPLv3 | optional; copyleft — opt-in only |
| OpenSMILE (eGeMAPSv02) | audEERING **source-available**, *not* OSI | optional; great features, license cloud |

Same features either way (F0, jitter, shimmer, HNR). Distress heuristic: high F0
variability + high jitter + low HNR ⇒ distressed; the inverse ⇒ calm. Fallback:
stdlib proxies (ZCR, energy, spectral centroid).

## Slot D — user-defined

The extensibility slot: load any model honoring the `Slot` interface (ONNX/torch).
Examples: language identification, keyword spotting, cough/sneeze detection.

## Signal conversion (DSP / Stage 1)

- **Capture:** `sounddevice` (PortAudio, MIT) → multichannel PCM. The hardware
  module presents as a **USB-C UAC2** device, so PDM→PCM happens in firmware and
  the laptop mics and the module hit the **same capture path** — only the channel
  count + coordinate map differ.
- **Resample:** capture rate (44.1/48k) → 16k for the models.
- **Spatial:** `pyroomacoustics` SRP-PHAT (permissive). Beamforming for separation.
- **Ultrasonic (>20kHz):** unreachable on laptop mics — the hardware-only premium.

## License summary (keep the core clean)

ECAPA Apache-2.0 · Silero VAD MIT · PANNs Apache/CC · YAMNet Apache-2.0 ·
librosa ISC · pyroomacoustics MIT · sounddevice MIT. Avoid GPL/“source-available”
in the default path (Parselmouth, OpenSMILE) — keep them strictly opt-in so the
project stays cleanly MIT.
