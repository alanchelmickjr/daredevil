# Portability — one brain, every device

> **Thesis:** Daredevil has to run on *any* device that could run an AI agent, or
> the hardware module is worthless. The open source **is** the brand — so the SDK
> must be the lingua franca for acoustic awareness, not a family of platform forks
> that quietly drift apart.

This document defines the architecture that makes "runs everywhere" a *guarantee*
instead of a marketing line. The companion device-specific plan is
[`IOS_PORT.md`](IOS_PORT.md) (iPhone is target #1, not *the* target).

---

## The trap we are explicitly avoiding

The naive path is a native rewrite per platform: Swift for iOS, Kotlin for
Android, JS for the browser, Python for servers. That is **N codebases**, and the
moment they exist they diverge — the router scores differently here, the tracker
stitches differently there, and "the open source is the brand" becomes three
brands. A protection system that means something different on each device is not a
protection system.

## The shape: one portable core, thin platform shims

The valuable, identity-defining parts are written **once** and every platform
calls into them. What's left per-platform is genuinely thin and unavoidable.

```
        ┌──────────────────────────────────────────────────────┐
        │              daredevil-core  (Rust)                  │
        │  slot runtime · router (priority+overrides) ·         │
        │  tracker · SPRT accumulator · NMF · awareness-map    │
        │  schema · ONNX Runtime inference · heuristic fallback │
        └───────────────┬───────────────┬──────────────┬───────┘
              FFI / C ABI │       FFI    │     WASM     │  FFI
        ┌───────────────┐ │ ┌───────────┐│┌────────────┐│┌──────────────┐
        │ iOS  (Swift)  │ │ │ Android   ││ │ Browser /  ││ │ Server / Orin│
        │ AVAudioEngine │ │ │ (Kotlin)  ││ │ agent      ││ │ / laptop     │
        │ Keychain, HUD │   │ AudioRecord│ │ WebAudio   │  │ (Python/FFI) │
        └───────────────┘   └───────────┘ └────────────┘  └──────────────┘
```

**Core = the 90% (the brain). Shim = the ~10% that is truly different per OS:**
audio capture, mic permission / background mode, secure key storage, and UI.

### Why Rust for the core

- **One canonical codebase** with a C ABI / FFI to every host language
  (Swift, Kotlin, Python via `pyo3`/`cffi`), and it **compiles to WASM** for the
  browser and agent runtimes — the single-codebase story the brand requires.
- Memory-safe and real-time-friendly (no GC pauses in the audio path).
- Mature ONNX inference bindings (`ort` for ONNX Runtime; `tract` as a
  pure-Rust fallback engine), plus `rustfft`/`realfft` for the DSP the prosody
  and spatial stages need.
- The existing Python package stays as the **reference implementation and test
  oracle** (see the parity gate below); it does not get thrown away — it becomes
  the thing the Rust core is validated against, and the easy path on servers.

> The slot **interface** and the **awareness-map JSON schema** in
> [`ARCHITECTURE.md`](ARCHITECTURE.md) are the contract the Rust core implements
> verbatim. The contract is the portable artifact; the language is an
> implementation detail underneath it.

---

## ONNX Runtime is the universal accelerator

[`MODELS.md`](MODELS.md) already names the bet: **one ONNX artifact, a hardware
execution provider per device.** That single idea *is* the "runs everywhere +
accelerates on device" engine. The portable core makes it real:

| Target class | EP (execution provider) | Notes |
|---|---|---|
| Server / cloud | CUDA, TensorRT | the big iron; also where Python-direct is fine |
| Workstation / laptop | CoreML (ANE) · DirectML · CPU | Mac Neural Engine; Windows/Linux x86 |
| Jetson / Orin | TensorRT | int8, the edge-GPU story |
| **Jetson / Thor** | **TensorRT 10.13 (Blackwell, CUDA 13)** | **in-process alongside LLM/STT/TTS; 500-person scale** |
| **Mobile NPU** | **CoreML (iOS/ANE) · NNAPI / QNN-Hexagon (Android)** | target #1 = iPhone |
| Browser / agent | WASM · WebGPU | the in-page / in-agent path |
| **On-module DSP** | int8 kernels | the patent hardware; the reason all of this exists |

**Three tiers, never two** — the pure-Rust/heuristic **fallback is the floor** so
the awareness map still computes on a device with zero acceleration. Reference
(ONNX+EP) → Portable (tract/CPU) → Fallback (heuristic). That floor is the promise
that makes the SDK safe to depend on.

---

## "Tune the model specifically for…" — governed, not ad-hoc

Per-device tuning is necessary (ANE prefers certain op/tensor layouts; Hexagon
wants per-tensor int8; TensorRT wants its own calibration). The danger is that
"runs everywhere" silently becomes "runs everywhere **differently**." So tuning is
a **governed matrix**, gated on accuracy.

### One architecture, many variants

| Slot | Architecture (exported once) | Variants per target class |
|---|---|---|
| WHO | ECAPA-TDNN (192-dim) | fp32 → fp16 → int8 · ANE / Hexagon layouts |
| WHAT | **YAMNet** (~4 MB — the mobile default, *not* PANNs) | fp32 → int8 (TFLite/ONNX) |
| HOW | no model — DSP (F0/jitter/shimmer/HNR) in `realfft` | n/a — pure code, recompiles |
| VAD | Silero (tiny ONNX) | ships as-is |

### The golden-vector parity gate (the keystone)

The existing **synthetic scene generator** (`daredevil/audio/capture.py`) becomes
the **cross-language, cross-platform oracle**:

1. Freeze a corpus of synthetic scenes + the Python reference's awareness maps as
   **golden vectors** (checked into the repo).
2. Every build — Rust core, and every quantized model variant — must reproduce the
   reference within tolerance:
   - WHO: voiceprint cosine-match drift below threshold (e.g. Δ < 0.02).
   - WHAT: top-k class agreement with the reference.
   - Router: identical `priority` ordering and identical override triggers.
3. A variant that fails the gate **does not ship.** This is what lets us tune
   aggressively per-NPU *without* the brand meaning something different on each
   device.

> Parity gate first, optimization second. Without it, every quantization PR is a
> silent regression to the product's core claim.

---

## Where this leaves iPhone

iPhone is simply the **first cell** in the mobile-NPU column — and the one that
proves the portable-core + thin-shim pattern end to end. The Swift work shrinks to
the shim (`AVAudioEngine`, permissions/background, SwiftUI HUD, Keychain); the
brain is the shared Rust core every other target will reuse. Details in
[`IOS_PORT.md`](IOS_PORT.md).

## Build order (strategy level)

- [ ] **Phase 0 — Parity harness.** Golden vectors from the synthetic scenes;
      schema + tolerances frozen. Governs everything after.
- [ ] **Phase 1 — `daredevil-core` (Rust) skeleton.** Slot trait, router, tracker,
      SPRT, awareness-map serializer, heuristic fallback — passing the parity gate
      against the Python reference, CPU-only.
- [ ] **Phase 2 — ONNX inference in core.** ECAPA + YAMNet + Silero via `ort`;
      CPU EP first, then CoreML/NNAPI/TensorRT EPs.
- [ ] **Phase 3 — First shim (iOS).** Prove FFI + capture + on-device run.
- [ ] **Phase 4 — WASM target + second shim (Android).** Prove "write once."
- [ ] **Phase 5 — Per-NPU int8 variants** under the parity gate.
