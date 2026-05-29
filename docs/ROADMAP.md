# Roadmap

First target: a one-command demo on a MacBook's built-in mics for a prospective
angel investor / design partner — proving fast iteration off the patent. Then
harden into the everywhere-SDK that also runs on the custom hardware.

## Current status (v0.1.0)

- [x] Package scaffold; **pure-Python core (zero required deps)**; pip-installable
- [x] One-command demo (`python -m daredevil.demo`) runs end-to-end **anywhere**
- [x] **WHO**: enrollment + cosine identity match + `C(t)=1−e^(−t/3)` curve
- [x] **WHERE/WHAT/HOW** slots with graceful fallbacks
- [x] Stage 3 router: priority (patent Eq. 2), `SAFETY_CRITICAL` + `DISTRESS`
      overrides, `UNKNOWN-NNN` tracking
- [x] Structured JSON awareness map + terminal radar + matplotlib option
- [x] Local-first identity store + Gun fleet scaffold + Node relay
- [x] Tests green on stdlib alone (`pytest`)

## Next

- [ ] Wire real backends on MacBook: ECAPA (`[speaker]`), PANNs (`[events]`),
      librosa prosody (`[prosody]`); validate accuracy
- [ ] Live multi-mic SRP-PHAT on a real array (`[audio]` + `[spatial]`)
- [ ] ONNX Runtime portable backend (CoreML / TensorRT) + int8 quantization
- [ ] Live Gun P2P voiceprint sync across two devices (encrypted, SEA)
- [ ] Record the demo video (deterministic `--file` scene)
- [ ] Validate on Jetson / Orin
- [ ] Publish `pip install daredevil` to PyPI

## Original 10-day plan (from the build spec)

- **Days 1–2** — Scaffold; mic capture on MacBook + Orin; port speaker embedding.
- **Days 3–4** — Integrate PANNs events; integrate local prosody (replacing the
  cloud API); test independently.
- **Days 5–6** — Stage 3 router: priority, overrides, unknown tracking; wire the
  parallel pipeline.
- **Days 7–8** — Structured JSON output; visualization; CLI; packaging.
- **Day 9** — End-to-end on MacBook **and** Orin; record demo; polish.
- **Day 10** — Demo.

## Success criteria

- [x] `python -m daredevil.demo` runs end-to-end with one command
- [x] Enroll a speaker in 3 seconds and identify correctly
- [x] Classify baby cry / alarm as safety-critical (priority override)
- [x] Prosody runs locally (no cloud)
- [x] Parallel-vs-sequential timing surfaced (measured live with real backends;
      simulated + labeled in pure-Python)
- [x] Clean, useful structured JSON awareness map
- [ ] `pip install daredevil` from PyPI on a MacBook
- [ ] Runs on both MacBook and Orin with real models
