# Roadmap

First target: a one-command demo on a MacBook's built-in mics for a prospective
angel investor / design partner — proving fast iteration off the patent. Then
harden into the everywhere-SDK that also runs on the custom hardware.

## North Star

Daredevil is a **protection system**; the acoustic awareness map is how it keeps
watch. Cameras and LIDAR are **line-of-sight** — they stop at the first wall.
Sound bends around corners, passes through walls, and carries for miles, so ears
reach what eyes can't: the toddler behind the couch, the hiker over the ridge.

The mission spans a spectrum, and the same WHO / WHERE / WHAT primitives serve all
of it:

- **Everyday (most days):** recognize the known mailman and open the door for the
  package; hear the pot boil over and cut the burner — concurrently, while the
  household vacuums and reads to a child, and **without taxing the host GPU**, so it
  can simply stay on.
- **Safety-critical:** a child who stops breathing → trigger care + CPR; a wheel
  about to fail → pull the car over; a child lost in a crowd → identify the voice +
  bearing; a pipe break → catch it before the house floods.
- **The absence:** the scariest signal is the room gone *wrong-quiet*. Detecting
  silence-where-there-should-be-sound is a north-star capability, not an afterthought.

Daredevil is the **perception layer and the trigger** — it notices in time and says
*act now* to whatever can act — on-device, privately, for everyone in the room,
including those the law won't let anyone else identify in the cloud.

### Goals that follow

- [ ] **Absence / cessation detection** — "wrong-quiet" as a first-class safety
      event (breathing rhythm / expected-sound monitoring), not just present-sound classes.
- [ ] **Beyond-line-of-sight localization** — bearing (and range) on sources you cannot see.
- [ ] **Always-on, low-tax** — run continuously alongside the host's real work
      without monopolizing CPU/GPU (parallel slots, int8/ONNX, the on-device module).
- [ ] **Actuation-trigger contract** — a stable, trustworthy "act now" signal a
      robot/agent can bet a life on.
- [ ] **Reliability bar** — measured detection / false-alarm rates per safety class;
      a protection system that silently fails is worse than none.

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
- [ ] **Portable runtime** — Rust core + thin platform shims so one SDK runs on
      every device an agent runs on (the open source *is* the brand). Strategy:
      [`PORTABILITY.md`](PORTABILITY.md); first target (iPhone 13+):
      [`IOS_PORT.md`](IOS_PORT.md)
- [ ] **Library import mode** — expose `Pipeline` as an embeddable component
      (not just HTTP server) so Chloe can run daredevil in-process on Thor
      without HTTP polling overhead. Same awareness map, direct function call.
- [ ] Live Gun P2P voiceprint sync across two devices (encrypted, SEA)
- [ ] Record the demo video (deterministic `--file` scene)
- [ ] Validate on Jetson / Orin
- [ ] Validate on **Jetson Thor** (TensorRT 10.13, CUDA 13, JetPack 7) —
      in-process mode alongside concurrent LLM/STT/TTS. Target: 500-person
      crowd with tiered attention gating.
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
- [x] Parallel-vs-sequential timing surfaced (always measured; pure-Python is
      GIL-bound, so the speedup widens with real backends)
- [x] Clean, useful structured JSON awareness map
- [ ] `pip install daredevil` from PyPI on a MacBook
- [ ] Runs on both MacBook and Orin with real models
