# HANDOFF — current state, CLI, next steps

Daredevil 0.1.0. Read `../CLAUDE.md` first for the rules; this file is the live
state and the place to start a session.

## What works right now

- **Core runs on the stdlib alone.** `pip install daredevil` with no extras
  imports and runs the full pipeline; every heavy backend is optional, lazy, and
  guarded with a deterministic fallback. `python -m pytest -q` → 25 passing on
  stdlib.
- **The four-stage pipeline** (spatial → parallel slot bank → attention router)
  produces a stable awareness-map dict. Synthetic demo is clean:
  `python -m daredevil.demo`.
- **WHO (identification)** is the headline path and is solid: ECAPA voiceprint
  (fallback fingerprint with no torch), Wald SPRT matching accumulated per track,
  CFAR background adaptation, Welford multi-sample enrollment.
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

## Changes on this branch (`claude/daredevil-macbook-testing-*`)

1. **Fixed the live-mic WHO blocker.** `is_speech_quality` had a hardcoded
   `0.05` RMS floor — too high for a real laptop mic and inconsistent with the
   model's own energy scale (`vad=0.004`, `quality_full_energy=0.02`). Speech that
   didn't clear 0.05 silently never reached the matcher, so WHO never fired live.
   The floor and ZCR ceiling are now config tunables (`Thresholds.vad`,
   `Thresholds.speech_zcr_max`); the floor mirrors the VAD gate and the SPRT
   down-weights quiet frames as designed. (utils.py, pipeline.py, config.py)
2. **Honest timing.** The pipeline previously hardcoded `sequential_ms: 0`, so the
   demo printed `sequential 0ms → 0.0× faster`. Sequential timing is now a real,
   opt-in measured pass (`listen(measure_sequential=True)`, used by the demo);
   otherwise the field is `null` and the renderer shows "not measured" instead of
   inventing a speedup. Pure-Python is GIL-bound, so it honestly reports ~1.0×.
   (pipeline.py, viz/spatial_map.py, demo.py)

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
daredevil calibrate [--name N] [--live] [--others]       seed the identity model
daredevil listen [--live|--file W] [--json]              emit one awareness map
daredevil serve [--live] [--port 8770]                   local web HUD
daredevil bench [--iters N]                              latency vs crowd size
daredevil devices                                        detected array + backends
daredevil mcp                                            run as an MCP server (stdio)
daredevil version
```
