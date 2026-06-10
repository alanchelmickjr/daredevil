# MacBook completion plan — live circles, real bearings, named speakers

The target experience: open the HUD on a MacBook and watch a circle appear for
each person as they speak, at the correct bearing, labeled with their enrolled
name — so the LLM (and the robots behind it) always knows WHO it is talking to
and WHERE they are. This doc is the gap analysis and the ordered plan to close
it, grounded in a full code review (June 2026) and a survey of the lateral
product landscape.

---

## 1. Where the code actually is

Verified against HEAD of `claude/macbook-speaker-identification-lla34j`:

**Working**
- Three-stage pipeline, parallel slots, priority router (Eq. 2), overrides.
- WHO: the sonar-pattern fix (commit `8b6722a`) holds — NMF spectral features
  own track association (WHERE/WHICH), Wald SPRT keyed **per track** owns
  identity (WHO), with hysteresis hold. Identity confirms in 1–4 frames with
  real ECAPA; no phantom tracks; two talkers stay two tracks.
- Web HUD (`daredevil serve`): renders sources as circles at azimuth with
  name, priority color, waveform, and live SPRT progress. The visualization
  layer the demo needs already exists.
- SRP-PHAT DOA (`stage1/spatial.py`) is correctly wired and consumes any
  geometry via the coordinate map (`stage1/mic_arrays.py`).
- Graceful degradation everywhere; tests green on stdlib alone.

**Broken / incomplete**
- 🔴 `audio/capture.py:257` — `MicStream` opens with `channels=1` and its
  callback keeps only channel 0. Live capture is **always mono**, so live
  SRP-PHAT never runs and every live source renders at the same bearing.
  (The `sd.rec()` fallback at `capture.py:304` already uses `arr.n_mics`.)
- 🟠 Calibration output (`calibrate.py`) is captured but never auto-loaded;
  SPRT runs on VoxCeleb-default parameters instead of the user's voice + room.
- 🟠 STT→LLM loop is half-wired: `viz/transcriber.py` buffers and flushes per
  source, the HUD has `transcript` / `llm_response` fields, but the `/probe`
  endpoint returns `implemented: False` and no local LLM call exists in the
  serve path.
- 🟡 Enrollment is a single ~10 s capture; the multi-sample, progress-feedback
  UX in `docs/ENROLLMENT_UX_RESEARCH.md` is designed but not built.
- 🟡 Attention gate (FULL/LIGHT/SKIP, `docs/ATTENTION_GATE_DESIGN.md`) is
  designed but not built — every source pays full slot cost every frame.
- `docs/HANDOFF.md` is referenced by CLAUDE.md but absent from the tree.

## 2. The load-bearing constraint: macOS gives us ONE channel

The research finding that shapes everything: **modern MacBooks do not expose
the raw 3-mic array to applications.** CoreAudio presents the built-in mic as
a single channel that is *already beamformed/processed* by the OS; Apple
developer-forum threads confirm the per-capsule signals are fused before any
API sees them, and Audio MIDI Setup shows 1 input channel on 2019+ MacBook
Pros (some 2017-era machines showed 2).

Consequences, stated plainly:

1. **True DOA from built-in MacBook mics is not achievable.** The `MACBOOK_3`
   geometry can never receive 3 live channels on modern hardware.
2. Faking bearings from a mono stream would violate rule 7 (honest demo) and
   rule 9 (no disguised stubs). We don't do it.
3. The mono-capture bug fix still matters — it is what makes **USB arrays**
   work, and a $25–70 ReSpeaker 4-mic plugged into the MacBook *is* the
   MacBook version: same laptop, same `pip install`, real bearings, and a live
   demonstration of the geometry-agnostic claim (Claim 5) — the array is just
   a coordinate map.

So the product truth is **two tiers, both honest**:

| Tier | Hardware | WHO | WHERE | HUD |
|---|---|---|---|---|
| Built-in | MacBook alone | ✅ full | ❌ (OS beamforms to mono) | non-spatial ring: circles sized/colored by identity+priority, bearing explicitly absent |
| Array | MacBook + USB array (ReSpeaker etc.) | ✅ full | ✅ SRP-PHAT bearings | true radar: circles at measured azimuth |

The awareness map already carries `array.spatial`; the HUD must render the two
modes *visibly differently* so a mono session never looks like a spatial one.

**Step zero (5 minutes, on the actual MacBook):** run `daredevil devices` /
`python -c "import sounddevice as sd; print(sd.query_devices())"` and record
the built-in device's `max_input_channels`. If Alan's machine reports >1, tier
limits relax (coarse 2-channel TDOA becomes defensible, labeled coarse); the
plan below assumes the documented case of 1.

## 3. The plan

Ordered so every milestone is demoable on its own. Estimates are working
sessions, not calendar promises.

### M0 — Truthful multi-channel capture (the unblock) · ~1 session
- `MicStream`: open with `channels = max(1, device_max_input_channels)`
  (clamped to the detected array's `n_mics`), ring-buffer all channels,
  `read_latest()` returns all channels; `capture_live()` returns a
  multi-channel `CaptureResult`.
- Derive `detect_array()` from the **device's actual channel count**, not the
  platform guess — a Mac reporting 1 channel must select `single`, never
  `MACBOOK_3`; a ReSpeaker reporting 4 selects `respeaker-4`.
- Kill the singleton staleness: if the device changes (array plugged in),
  recreate the stream.
- Tests: ring-buffer multi-channel round-trip; array selection from channel
  counts (no hardware needed — these are deterministic).

### M1 — WHERE on a real array · ~2 sessions (needs the USB array in hand)
- Validate live SRP-PHAT end-to-end: ReSpeaker on the MacBook → bearings in
  the awareness map → circles move on the HUD as the speaker walks around.
- Tune the existing alpha-beta bearing filter against real motion; measure and
  record bearing error at known angles (protractor-on-the-desk test — measured,
  never invented).
- Confirm the `load_coordinate_map` path with a hand-written JSON map, proving
  the Claim 5 story live.

### M2 — HUD: the circles tell the truth · ~1–2 sessions
- Two explicit render modes keyed off `array.spatial`: radar (bearing) vs ring
  (non-spatial), labeled on screen.
- "Identifying each speaker as they speak": speaking-state pulse on the active
  circle, name fades from `UNKNOWN-NNN` → enrolled name at the moment SPRT
  confirms (the existing LLR progress bar becomes the pre-confirmation state).
- Click a circle to focus (exists) → show its transcript and LLM reply (M3).

### M3 — Close the loop: AI knows who it's talking to · ~2 sessions
- Wire the local LLM call (Ollama/Gemma, already prototyped in commits
  `5911059`/`3c27d8c`) into `viz/server.py`: focused source → transcript →
  prompt **prefixed with the awareness map** → reply rendered in the HUD.
- Implement `/probe` honestly or delete the stub.
- This is the investor moment: speak, watch your circle light up with your
  name, and the model answers *you* — addressing you as you, because the map
  told it WHO.

### M4 — WHO hardening · ~2 sessions
- Auto-load `calibration.json` when present; surface "calibrated vs default"
  in `daredevil devices` and the map.
- Multi-sample enrollment (3–5 varied utterances, ~20 s total — matches both
  our own UX research and current literature), progressive % feedback, SNR
  gate on enrollment frames, d-prime readiness threshold.
- Record measured identification accuracy for the demo (N trials, same room).

### M5 — Ship & scale (post-demo)
- ONNX Runtime + CoreML EP + int8 (`docs/MODELS.md` plan) → measured, not
  predicted, latency numbers.
- Attention gate FULL/LIGHT/SKIP so crowded rooms stay cheap (the "always-on,
  low-tax" roadmap goal).
- PyPI publish; restore `docs/HANDOFF.md` as the living status page.

## 4. Why this wins (lateral landscape, June 2026)

Nobody ships the assembled product. The field splits cleanly:

- **Diarization without identity or space:** pyannote/pyannoteAI (€8.1M seed,
  Apr 2025 — cloud API), diart (streaming, anonymous SPEAKER_N labels), NVIDIA
  Streaming Sortformer (open weights, ≤4 speakers, anonymous, GPU-assumed),
  AssemblyAI/Speechmatics (cloud).
- **On-device identity, closed:** Picovoice Eagle/Falcon (proprietary,
  no spatial, no events), Apple Sound Recognition + voice ID (OS-locked, no
  developer API), Alexa Voice ID (cloud-matched).
- **Spatial without identity:** Kardome ($10M Series A, Hyundai — embedded OEM
  licensing, not a developer SDK), ODAS (robot DOA, no WHO, dormant), pyroomacoustics (research toolkit — already our backend).
- **Events:** Audio Analytic was *acquired by Meta* (2022) and its SDK pulled
  off the market — the licensable sound-event niche sits vacant.
- **Robots:** no public humanoid audio-perception stack found (Figure/1X/
  Tesla) — academic surveys still treat robot audition as open.

**Open, enrolled, persistent speaker identity on-device — fused with bearing,
events, and prosody into an LLM-facing map — is unoccupied.** The seed rounds
that validate the adjacent slices (pyannoteAI €8.1M, Kardome $10M) are exactly
the size of this thesis. SPRT-style sequential identity decisions appear in no
shipping product we could find; that framing is currently ours.

## 5. The demo script (MacBook + array on the table)

1. `pip install -e . && daredevil serve --live` — HUD opens, ring mode,
   built-in mic. Speak: a circle appears, pulses, SPRT bar fills, your name
   lands. *WHO on any laptop, zero hardware.*
2. Plug in the ReSpeaker. HUD flips to radar mode by itself (geometry from a
   coordinate map — the patent's Claim 5, live). Walk and talk: your named
   circle tracks your bearing.
3. Second person speaks: second circle, `UNKNOWN-001`, tracked and held
   separate. Enroll them in ~20 s; the label becomes their name.
4. Click your circle, ask a question: transcript appears, the local model
   answers *you by name* — because the awareness map reached it before your
   words did.
5. Close the laptop lid halfway and keep talking — sound doesn't need line of
   sight. That's the protection-system pitch in one gesture.

Everything shown is measured and live; mono mode is labeled mono. That's the
credibility the $10M conversation actually buys.
