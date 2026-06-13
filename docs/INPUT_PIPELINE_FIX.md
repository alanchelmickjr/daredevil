# Working plan — Input pipeline fix (live capture "disaster")

> Living progress doc. Updated after EACH step so work can resume after any context
> loss. Branch: `claude/brave-gauss-5sG9E`.

## Symptoms (reported, live MacBook)
- "Speak every once in a while, it's not hearing me right now."
- "Stuck in a bad calibration, I think."
- Regression: live mic "once worked, now a disaster." Synthetic path is fine.

## Diagnosis (confirmed by reading code + git history)
The last capture commit `a2cb0ab` re-introduced the exact ring-buffer streaming that
`c9c545a` had just reverted (with the correct reason: a batch consumer can't consume a
streaming source without lag). Four concrete faults:

1. **Audio loss (the disaster).** `MicStream.read_latest()` *peeks the tail* of a
   free-running 2 s ring (`deque(maxlen=20)`); nothing paces reads. Server cadence is
   `POLL_MS=1500ms + pipeline processing` (≈2–4.5 s) > 2 s ring depth → 0.5–2.5 s of every
   cycle is overwritten before it's ever read. Intermittent speech lands in the gap → never
   heard. (Matches "speak once in a while, not hearing me.")
2. **WHERE dead + masked.** `MicStream` hardcodes `channels=1` but `capture_live` tags the
   result with a multi-mic array (`MACBOOK_3`/`RESPEAKER_4`). `spatial.process` trusts the
   *label* (`array.spatial_capable`), calls `_srp_phat` with 1 channel vs 3-mic geometry →
   throws → swallowed by a bare `except` → azimuth always `None`.
3. **No cold-start / no recovery.** Empty ring returns `[0.0]*n` as a real frame (fake
   silence on first poll). A wedged stream keeps `_open=True`, so `get()` never re-opens;
   `read_latest` returns stale-then-silent forever. (Matches "not hearing me *right now*.")
4. **Latent.** `int(seconds/0.1)` truncates sub-second windows (0.3→2, 0.6→5, 0.7→6); `0.1`
   is a duplicated magic literal (CLAUDE.md rule #8).

Plus: **bad calibration self-stuck.** `load_calibration` trusts whatever's on disk. A model
fit while capture was broken (or an enrolled-on-garbage voiceprint) yields no separation and
the matcher rejects the real person — with no self-recovery.

## Root cause
Pull-based batch consumer bolted to a push-based free-running ring with no rate contract,
mono-only, no health/recovery; calibration loader trusts a degenerate model.

## Approach (option #1, done right — researched)
Persistent `sd.InputStream` + **blocking `read(frames)`** (sounddevice docs: blocks until
exactly `frames` are available, returns `(data[frames×ch], overflowed)`). This is
simultaneously self-pacing, gapless, multichannel, and persistent (no mic-flashing) — it
ends the flash-vs-lag oscillation instead of re-picking a side. `blocksize=0` (PortAudio
optimal) removes the `0.1` magic literal and the float-trunc bug by construction.

## Measured facts (don't guess)
- Synthetic calibration session: d′=31.2 (target 1.00±0.03 vs impostor 0.06±0.03). So a
  degenerate-calibration guard floor of **d′ < 0.5** has huge headroom; won't touch synthetic
  test or any "poor-but-usable" real calibration. Textbook defaults are d′≈4.1.
- Tests use `pytest` (not preinstalled here → installing). Hard constraints:
  - `test_calibrate_session_writes_loadable_model`: `load_calibration` must still return a
    model with `target_mean > impostor_mean`.
  - `test_end_to_end_synthetic`: `amap["array"]["spatial"] is True` for synthetic MACBOOK_3
    (synthetic replicates mix across `n_mics` channels → `n_channels == n_mics`, so the new
    spatial gate stays True). ✔ safe.

## Checklist
- [x] 0. Recon tests (constraints captured above)
- [x] 1. `capture.py`: `_pick_input_device()` returns `max_input_channels`
- [x] 2. `capture.py`: rewrite `MicStream` (persistent, blocking read, multichannel, health/re-open, singleton keyed on device+sr+nch)
- [x] 3. `capture.py`: `_reconcile_array()` so captured channels == `array.n_mics`; robust one-shot `sd.rec` fallback
- [x] 4. `spatial.py`: gate DOA on actual `n_channels == n_mics ≥ 2`; `log.debug` the except (stop masking)
- [x] 5. `config.py`: `load_calibration` ignores degenerate model (d′ < 0.5) with a clear warning
- [x] 6. Verify: `pytest -q` + synthetic `listen` smoke
- [x] 7. Commit + push to `claude/brave-gauss-5sG9E` (commit d32ea7c)
- [x] 8. Hand user the calibration reset steps (in this doc + chat)

## Step log
- Step 0 done: read tests, captured constraints; measured synthetic d′=31.2; baseline pytest
  blocked (not installed → installing in background).
- Steps 1–3 done (`capture.py`): `_pick_input_device` now returns max_input_channels;
  `MicStream` rewritten to persistent stream + blocking `read()` (multichannel, health/re-open,
  singleton keyed on channels+device+sr); `_reconcile_array` guarantees channels==n_mics;
  `capture_live` uses the stream with a self-pacing one-shot `sd.rec` fallback. Ring buffer,
  `read_latest`, the `0.1` magic literal and the float-trunc bug are all gone.
- Step 4 done (`spatial.py`): DOA gated on `n_channels>=2 and n_channels==n_mics`; SRP-PHAT
  failure now `log.debug`s instead of silently masking.
- Step 5 done (`config.py`): `load_calibration` computes d′ and ignores a degenerate model
  (`CALIBRATION_MIN_DPRIME=0.5`) with a clear warning; self-recovers from a bad calibration.
- Step 6 done (verify): `py_compile` clean; **pytest 25/25 pass**; synthetic E2E runs; synthetic
  MACBOOK_3 DOA passthrough intact (az 45/0/270); `_reconcile_array` keeps channels==n_mics for
  all cases; degenerate calibration (d′=0.10) rejected→defaults, healthy model kept.

## How to verify on a real mic (cannot be done in this headless container)
1. `python -m daredevil serve --live` (or `python -m daredevil.demo --live`).
2. Speak intermittently — every utterance should register now (no dropped gaps).
3. `daredevil devices` should show the detected array; with a true multi-mic device
   (ReSpeaker / the module) azimuths should populate.

## Calibration reset (for the "stuck" model on the user's machine)
The degenerate-model guard auto-ignores a useless calibration, but a voiceprint enrolled
on broken audio still needs redoing. After confirming the mic works (step 2 above):
- `rm ~/.daredevil/calibration.json`  (or `$DAREDEVIL_HOME/calibration.json`)
- Re-run calibration (it wipes the old voiceprint and re-enrolls): `daredevil calibrate`
  or the HUD onboarding. Do this ONLY after capture is confirmed working.
