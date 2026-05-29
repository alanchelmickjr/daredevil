# Performance

Two targets: **< 200 ms** for the awareness pipeline, and **< 3 s** end-to-end
including a local LLM response.

## Measured today (pure-Python + numpy fallback, this machine)

```
daredevil bench
  parallel slots (production path): ~75 ms   ✓ < 200ms
  full listen() incl. demo-only sequential pass: ~350 ms
```

The production path is the **parallel slot pass**. The sequential pass exists only
to show the parallel-vs-sequential comparison in the demo — it is not run in
production. So we are already comfortably under 200 ms before any GPU/ANE help.

## < 200 ms pipeline budget (real models)

Slots run concurrently, so the pipeline cost is `max(slot)`, not the sum.

| Stage | CPU (reference) | Mac CoreML / int8 (target) |
|---|--:|--:|
| capture + framing + resample | ~5 ms | ~5 ms |
| Stage 1 — SRP-PHAT DOA (multi-mic) | 10–30 ms | 10–20 ms |
| Slot A — ECAPA (WHO) | 40–90 ms | 15–35 ms |
| Slot B — events (PANNs→**YAMNet** for edge) | 50–110 ms | 15–40 ms |
| Slot C — prosody (librosa) | 15–30 ms | 15–30 ms |
| **Stage 2 = max(A,B,C)** | **~110 ms** | **~40 ms** |
| Stage 3 — router | < 1 ms | < 1 ms |
| **pipeline total** | **~145 ms** ✓ | **~65 ms** ✓ |

Levers if a slot blows the budget: ONNX Runtime + **int8** quantization, the
CoreML EP (Apple Neural Engine) on Mac / TensorRT on Orin, and **YAMNet** (~4 MB)
instead of PANNs for events. ECAPA is already small.

## < 3 s total with Gemma

```
mic → pipeline(~150ms) → attention gate → STT(surfaced only) → Gemma → reply
```

| Step | Budget |
|---|--:|
| pipeline (above) | ~0.15 s |
| local STT (whisper.cpp tiny/base, short utterance) | 0.1–0.5 s |
| Gemma (E2B/E4B local via Ollama, streamed) — first token | 0.3–1.0 s |
| short reply (streaming) | 0.5–1.5 s |
| **total to first useful token** | **~1–2 s** ✓ |

Key choices that keep it under 3 s: the **attention gate** means we only STT the
*surfaced* sources (not the radio), the LLM is a **small local** model, and we
**stream** the reply so "time to first token" is what the user feels.

## How to measure

```bash
daredevil bench --iters 20      # pipeline latency on this machine
python -m daredevil.demo        # per-run parallel vs sequential timing in the map
```
