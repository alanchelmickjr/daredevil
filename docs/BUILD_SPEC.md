# Build specification

The canonical software spec, distilled from the original handoff. (Third-party
names removed; reframed for the open-source release. The *software* is MIT; the
*hardware module + firmware architecture* are patent-pending.)

## Goal

`pip install daredevil` → run it on a MacBook's built-in 3-mic array and see the
value immediately. First demo target: a MacBook, for a prospective angel investor
/ design partner, to show fast iteration straight off the patent. Must also run
on Jetson / Orin and degrade cleanly to a single mic.

**One-line pitch:** *converting raw environmental audio into structured real-time
context before it reaches the LLM — "OAK-D for ears."*

## What it does

```python
from daredevil import Pipeline

pipeline = Pipeline()                    # auto-detects MacBook / ReSpeaker / ALSA / single
pipeline.enroll("alan", mic_seconds=3)   # 3-second enrollment minimum
context = pipeline.listen(duration=1.0)  # -> structured acoustic context (dict)

response = llm.generate(audio_context=context, user_input=transcript)
```

`context` is the awareness map documented in [`ARCHITECTURE.md`](ARCHITECTURE.md):
enrolled + unknown sources, each with event, prosody, identity, position, and a
fused priority, plus parallel-vs-sequential timing and a privacy block.

## Installation (what a new user types)

```bash
pip install -e .            # today, from source (PyPI publish is on the roadmap)
python -m daredevil.demo    # auto-detects mics; first run downloads models (~500MB)
```

Pure-Python core means the demo runs even with **no** mic, GPU, or models — a
deterministic synthetic scene exercises the whole pipeline.

## Pipeline (refactor + integration, not a ground-up build)

Prior art already working from earlier robot work: speaker embedding/ID, basic
spatial DOA, and prosody via a cloud API. This project **replaces the cloud
prosody with local analysis**, adds **PANNs event classification**, the
**structured output format**, **parallel execution**, the **Gun fleet identity
backbone**, and **pip packaging**.

- **Stage 1 — Spatial:** capture (`sounddevice`), SRP-PHAT DOA
  (`pyroomacoustics`), beamforming. Geometry-agnostic via a coordinate map.
- **Stage 2 — Parallel slots:** A) ECAPA speaker embedding, B) PANNs events with
  safety flags, C) local prosody (librosa/eGeMAPS-style; replaces the cloud API),
  D) user-defined. All concurrent → latency = max(slot), not sum.
- **Stage 3 — Attention router:** identity + position + event + prosody → priority
  + overrides → structured JSON.

See [`MODELS.md`](MODELS.md) for model + license choices and the engine plan.

## Demo modes

```bash
python -m daredevil.demo                 # deterministic synthetic scene (reliable for video)
python -m daredevil.demo --live          # real microphone (enroll + listen live)
python -m daredevil.demo --file scene.wav  # a recorded multi-source scene
python -m daredevil.enroll --name alan --seconds 10   # watch confidence climb
```

## Single-mic fallback

On a laptop with one mic (no array): ✅ speaker ID, ✅ events, ✅ prosody,
❌ spatial DOA, ❌ source separation. Three of four capabilities still give the
LLM structured context it doesn't have today. The spatial awareness is the
premium that justifies the hardware.

## The bridge

> *Everything you just saw runs on a laptop's built-in mics — no cloud, no GPU.
> The hardware module adds a 3D array, an ultrasonic range, and runs the inference
> on-device. Software proves the architecture; hardware proves it scales; the
> patent protects the hardware.*
