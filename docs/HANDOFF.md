# HANDOFF — current state, CLI, next steps

Daredevil 0.1.0. Read `../CLAUDE.md` first for the rules; this file is the live
state and the place to start a session.

## What works right now

- **Core runs on the stdlib alone.** `pip install daredevil` with no extras
  imports and runs the full pipeline; every heavy backend is optional, lazy, and
  guarded with a deterministic fallback. `python -m pytest -q` → 32 passing on
  stdlib.
- **The four-stage pipeline** (spatial → parallel slot bank → attention router)
  produces a stable awareness-map dict. Synthetic demo is clean:
  `python -m daredevil.demo`.
- **WHO (identification)** is the headline path and is solid: ECAPA voiceprint
  (fallback fingerprint with no torch), Wald SPRT matching accumulated per track,
  CFAR background adaptation, Welford multi-sample enrollment.
- **Wake word — attention by name.** Daredevil hears its own name and turns to the
  caller; WHO (above) says who that is. Two backends: a stdlib query-by-example
  detector (sub-sequence DTW on a spectral contour, learned from a few spoken
  examples) and optional openWakeWord (local, no key). `daredevil wake --live` to
  teach it; the map carries `wake`, `addressed`, and `attention_reason`
  (`safety|voice|name|owner-speaking|salient`). See `docs/WAKE_WORD.md`. Threshold
  wants on-mic tuning with a real voice.
- **Persistent identity.** Enrolled voiceprints are written to
  `~/.daredevil/voiceprints/` (override with `$DAREDEVIL_HOME`). They survive
  process/session restarts: enroll once, recognized on every later run. Verified
  end-to-end across separate processes.

## MacBook testing — the reality

On Apple hardware the OS **locks beamforming**: you get a single processed mono
stream, not the raw mic array. So on a MacBook:

- **WHERE / DOA is unavailable** — single mic, no direction. This is expected, not
  a bug. `detect()` reports `single` and spatial degrades to mono cleanly.
- **The whole game is WHO** — speaker identification — which is also the project's
  stated priority. Test that.

### How to test WHO on a MacBook

```bash
pip install -e ".[speaker,audio]"          # ECAPA + sounddevice (first run downloads ~500MB)
export DAREDEVIL_KEY="<any-passphrase>"     # encrypts voiceprints at rest (optional but recommended)

daredevil enroll --name <you> --live -s 10  # 10s of your voice → persistent voiceprint
daredevil wake --live -s 2                   # say "Hey Radar" 2-3× → it wakes when called
daredevil listen --live                     # one awareness map to stdout
daredevil serve --live                      # web HUD at http://127.0.0.1:8770 (best for WHO — it accumulates)
daredevil devices                           # what was detected / installed
```

**Set expectations on the SPRT.** Identity accumulates evidence *per track over
time*. A single one-shot `listen` (1s window) usually shows `identifying you
(NN%)` rather than a confirmed match unless a frame clears `immediate_cosine`
(0.80). WHO firms up over a few seconds of continuous audio — use `serve` (or
repeated `listen` on the same persistent track) to see it lock. A slow lock is
the SPRT working, not a failure.

**Persistence model.** The *voiceprint* persists on disk (encrypted when
`DAREDEVIL_KEY` is set; otherwise stored base64 and clearly marked `enc:none`).
The SPRT accumulator is in-memory and re-accumulates each session from the
persisted voiceprint — identity memory persists, recognition evidence restarts.

## Platform notes — why macOS is different (read before laptop testing)

Two things behave unlike a Jetson / USB array / robot base. Neither is a Mac
weakness — it's policy + ecosystem.

**No raw mic channels from the built-in array.** The MacBook has ~3 bezel mics,
but macOS consumes the raw per-mic channels inside CoreAudio's own DSP
(beamforming + Voice Isolation + AEC) and hands every app a single *processed
mono* stream. Reasons: privacy (raw multichannel enables non-consensual spatial
localization), consistent quality, and the array geometry/beamformer are Apple
IP. So **WHERE is unavailable on the built-in mic by design** — not a bug.
It is *not* a Mac limitation: a class-compliant **USB UAC2 multichannel array**
(ReSpeaker, or our hardware module) is passed through raw, all channels intact.
That's the symmetry in `audio/capture.py` — the module presents as a USB-C UAC2
device, so the live path is identical to a laptop and WHERE comes back the moment
real array hardware is plugged in.

**GPU exists, but the ML path has gaps.** There is no CUDA on a Mac (Apple dropped
NVIDIA). The GPU is reachable via Metal / PyTorch **MPS**, but MPS has incomplete
op coverage — SpeechBrain/PANNs kernels can error or silently fall back to CPU.
The Neural Engine (ANE), Apple's best accelerator, is only reachable through
**CoreML**, not raw PyTorch. So "GPU like my other devices" doesn't hold: other
devices are CUDA (the ecosystem's first-class target); the Mac path is
Metal/MPS/CoreML. This is why the acceleration plan (docs/MODELS.md) is
reference-torch → ONNX Runtime with a CoreML execution provider — that's the
route to both the Mac GPU and the ANE. For WHO at 1s windows, CPU is fine, so
none of this blocks identification testing.

## Recent consolidation (June 2026) — four parallel session branches merged

Several `claude/*` web-session branches had each fixed a real piece of the
live-mic path independently; they were never reconciled, so `main` went stale
while the working code sat scattered. They are now consolidated onto one line:

1. **Live-capture rewrite (was PR #10).** `MicStream` is a persistent
   `sounddevice.InputStream` read with a *blocking* `read(frames)` — self-pacing,
   gapless, and **multichannel** (the old callback kept only channel 0, so live
   capture was always mono and SRP-PHAT never ran). `_reconcile_array()` guarantees
   captured channels == `array.n_mics` so the spatial stage never gets a geometry
   it can't fill, and a degenerate-calibration guard (`CALIBRATION_MIN_DPRIME`)
   rejects voiceprints fit on broken audio. (capture.py, spatial.py, config.py)
2. **Fixed the live-mic WHO blocker.** `is_speech_quality` had a hardcoded `0.05`
   RMS floor — too high for a real laptop mic, so normal-distance speech silently
   never reached the matcher and WHO never fired live. The floor and ZCR ceiling
   are now config tunables (`Thresholds.speech_gate_energy = 0.012` ≈ 3× the VAD
   floor, `Thresholds.speech_gate_zcr`); a frame only has to clear "clearly speech"
   to be *eligible*, and the SPRT down-weights quiet frames as designed. Covered by
   `tests/test_core.py::test_speech_gate_passes_quiet_speech`. (utils.py,
   pipeline.py, config.py) *(Chosen over a competing `vad`-floor variant that fixed
   the same bug at 0.004 without a test.)*
3. **Honest timing.** The pipeline previously hardcoded `sequential_ms: 0`, so the
   demo printed `sequential 0ms → 0.0× faster`. Sequential timing is now a real,
   opt-in measured pass (`listen(measure_sequential=True)`, used by the demo);
   otherwise the field is `null` and the renderer shows "not measured" instead of
   inventing a speedup. Pure-Python is GIL-bound, so it honestly reports ~1.0×.
   (pipeline.py, viz/spatial_map.py, demo.py)
4. **Docs:** `MACBOOK_COMPLETION_PLAN.md` (M0–M5 + competitive landscape),
   `IOS_PORT.md` + `PORTABILITY.md` (the hardware-adjacent USB-array / iPhone story),
   `INPUT_PIPELINE_FIX.md`, and a `Makefile` (install / live / listen / devices /
   recalibrate / test). The mono-capture 🔴 blocker called out in the completion
   plan's M0 is **resolved** by item 1 above.

## Known constraints / next steps

- **Backend acceleration is not wired into the slots.** `detect_backend()`
  reports `mps`/`cuda`, but `EmbeddingSlot` (no `run_opts` device) and
  `EventsSlot` (hardcoded `device="cpu"`) run on CPU regardless. For WHO on a
  MacBook (1s windows) CPU latency is fine, so this is not a blocker — but the
  reported backend is cosmetic until the slots honor it. SpeechBrain/PANNs on MPS
  hit unsupported-op fallbacks, so this needs on-device testing, not a blind
  `.to("mps")`.
- **Fleet sync (`GunStore`)** is local-authoritative with a no-op push stub; P2P
  voiceprint sync across the trust chain is the next milestone.

## CLI reference

```
daredevil demo [--live|--file W] [--fallback] [--json]   end-to-end demo
daredevil enroll --name N [--live] [-s SECONDS]          enroll a speaker (persistent)
daredevil wake [--phrase P] [--live] [-s SECONDS]        teach its name (wake word)
daredevil calibrate [--name N] [--live] [--others]       seed the identity model
daredevil listen [--live|--file W] [--json]              emit one awareness map
daredevil serve [--live] [--port 8770]                   local web HUD
daredevil bench [--iters N]                              latency vs crowd size
daredevil devices                                        detected array + backends
daredevil mcp                                            run as an MCP server (stdio)
daredevil version
```
