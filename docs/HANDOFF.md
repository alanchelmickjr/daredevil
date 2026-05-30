# Daredevil — Handoff & CLI Reference

**As of commit `db1d8f0` · branch `main` · 2026-05-29**

Read `CLAUDE.md` first (goals + guardrails). This doc is the point-in-time
snapshot: how to run everything, what's real vs. stubbed, and what's next.

---

## Pull & build

```bash
git clone https://github.com/alanchelmickjr/daredevil
cd daredevil

# Option A: conda (recommended — all heavy backends + MPS acceleration)
brew install miniforge
conda env create -f environment.yml
conda activate daredevil
pip install -e .

# Option B: pip with extras
pip install -e ".[full]"

# Option C: core only (pure-Python, zero deps — still runs the full pipeline)
pip install -e .
```

The core has **no required dependencies** — `python -m daredevil.demo` works on a
fresh checkout with nothing else installed (pure-Python fallback). Heavy backends
only *accelerate* the same pipeline.

---

## CLI reference

| Command | What it does |
|---|---|
| `daredevil demo` | End-to-end demo on the synthetic scene. |
| `daredevil demo --live` | Same, but uses the real microphone. |
| `daredevil demo --file scene.wav` | Run against a recorded WAV. |
| `daredevil demo --json` | Print only `{enrollment, awareness_map}` JSON. |
| `daredevil serve [--port 8770] [--live]` | Launch the **web HUD** at `http://127.0.0.1:8770`. |
| `daredevil mcp` | Run as MCP server (stdio) for Claude / LLM agents. |
| `daredevil bench [--iters 10]` | Crowd-scaling benchmark. |
| `daredevil enroll --name NAME [-s 3] [--live]` | Enroll a speaker. |
| `daredevil calibrate [--name N] [--live] [--others]` | First-run "get to know each other" session — seeds the identity model from your real voice + room. |
| `daredevil listen [--duration 1.0] [--live] [--json]` | Emit one awareness map. |
| `daredevil devices` | Detected array + installed backends. |

**Quick demo:**
```bash
python -m daredevil.demo          # synthetic scene — works anywhere
python -m daredevil serve --live  # live HUD with real mic + ML backends
```

---

## What's real (verified working on MacBook Air M2, 2026-05-28)

- **ECAPA-TDNN speaker embedding** — SpeechBrain, 192-dim, running on MPS (Apple Metal GPU). Proper audio normalization + embedding normalization per SpeechBrain docs. Match score 0.78+ on same-speaker.
- **PANNs CNN14 event classification** — 527 AudioSet classes, 32kHz resampling, running on CPU. Correctly identifies Speech, Music, Whistling, Vehicle, Animal, etc.
- **librosa prosody** — pYIN F0 extraction + distress heuristic.
- **ConvTasNet source separation** — Asteroid, pretrained on WHAM!, splits mono into 2 source streams before the slots analyze each independently. ~160ms per frame.
- **SPRT identity matching** — Log-likelihood ratio accumulation across frames (Sequential Probability Ratio Test). Identity confidence builds over time like name-that-tune. AS-Norm calibration against enrolled cohort.
- **Multi-sample enrollment** — Welford online mean update. Voiceprint sharpens with each re-enrollment.
- **Tracker** — Slot-based assignment. Single-mic: recency-first (most recent track = current source). Multi-source: cosine + recency. Tracks go ACTIVE → DORMANT (15s) → removed.
- **Web HUD** — Neumorphic-steampunk orbital display. Known speakers on left column (3x linger), unknown active on radar, stale fading to right column (capacity-based eviction). Crash-resistant (handles refresh mid-inference).
- **MCP server** — 4 tools (listen, awareness, enroll_speaker, devices). Stdio transport. Configured for Claude Code in `.claude/settings.local.json`.
- **Pipeline logging** — Capture RMS, separation streams, tracker decisions, LLR state. Logs to stdout (redirect to file with `> /tmp/daredevil.log 2>&1`).
- **Full test suite** — 9 tests passing with all real backends loaded.
- **Privacy** — No cloud, no raw audio stored, non-reversible embeddings only. COPPA compliant.

---

## Backends detected

With conda env `daredevil` active:
```
backend: mps (Apple Metal GPU)
slots: embedding=reference (ECAPA), events=reference (PANNs), prosody=reference (librosa)
separator: reference (ConvTasNet)
```

---

## Architecture (current)

```
capture → spatial (DOA) → SEPARATION (ConvTasNet) → parallel slots → tracker → router/gate → awareness map
                                                    ├─ WHO (ECAPA)                                  ↓
                                                    ├─ WHAT (PANNs)                          → LLM payload
                                                    └─ HOW (librosa)                         (only surfaced sources)
```

---

## Recently fixed (2026-05-30 session)

- **Global SPRT key** — identity accumulates per enrolled speaker name, not per
  tracker contact. One SPRT for "alan" that persists regardless of track thrash.
- **Energy-only accumulator gate** — removed PANNs event-class gating. Music/noise
  no longer blocks identity accumulation; the SPRT decides match, not the gate.
- **Coasting tracks in awareness output** — the HUD now shows all live tracker
  contacts, not just what the current frame detected. Contacts fade to sidebar
  when coasting (lost for >5s), removed at 15s.
- **Track status in awareness map** — `track_status` field (tentative/confirmed/coasting)
  lets the HUD render contacts differently based on confidence.
- **Calibration HUD panel** — neumorphic modal overlay with 3-2-1 countdown, level
  meter, progress bar, d-prime result. Triggered by Calibrate chip or double-click core.
- **Calibration active flag fix** — awareness endpoint was returning stale cached data
  after calibration completed because `active` was never set back to false.
- **Reverted sub-windowing** — 200ms chunks broke PANNs classification (needs 1s).
  Restored full 1s capture with ConvTasNet separation.

## Recently fixed (2026-05-29 session)

- **Mic device selection** — `capture_live()` now picks the first input device with
  native rate ≥ 44.1kHz, skipping dead Bluetooth endpoints (AirPods Max at 24kHz
  returns zeros when not in call mode). MacBook Air mic is the reliable default.
- **ConvTasNet energy normalization** — separated streams were amplified 100-1000x
  (RMS 87 from a 0.07 input). Now normalized so total output energy matches input,
  preserving relative proportions between streams.
- **Separation energy-ratio gate** — ConvTasNet always outputs 2 streams even for a
  single speaker, splitting harmonics 50/50. New `energy_ratio_cap=0.80` check:
  if two streams have nearly equal energy, it's one source split in half, not two
  real sources. Combined with lowered `distinct_cosine=0.60`.
- **Calibrate wipes stale voiceprint** — enrollment during calibration now deletes the
  old record first so you're never measuring against a voiceprint from a different
  mic/session. Single recording used for both enrollment and verification (no double-
  record UX bug). Default bumped to 20s.

## Previously fixed (WHO matching + tracking stitch)

- **Wald SPRT identity classifier** — `enrollment/manager.py` now accumulates a true
  per-frame log-likelihood ratio `logN(s;target) − logN(s;background)` and decides at
  the Wald bound `A = log((1−β)/α)`; confidence is `sigmoid(LLR)` (a real posterior).
  Works with a single enrolled speaker (configured background model; an adaptive
  AS-Norm cohort kicks in at ≥2 speakers). Fixes the old false-UNKNOWN cliff where a
  real speaker below cosine 0.70 was read as unknown (repro: 0/20 → 20/20 at cos 0.55).
- **Identity is per-track** — evidence accumulates per `(track, speaker)`, so two
  simultaneous contacts no longer pool their identity evidence. WHERE (tracking) and
  WHO (identity) now share one state per contact.
- **Multi-target track manager** — `stage3/tracker.py`: gate → associate (embedding +
  bearing) → M-of-N confirm → coast → delete, with an α-β bearing filter. Replaces the
  recency-only mono hack.
- **Separation gating** — `pipeline.py` only splits a stream when there are genuinely
  ≥2 energetic, distinct contacts; a single talker keeps its clean wideband audio so
  the 8 kHz separation residual no longer corrupts the ECAPA voiceprint.
- All tunables live in `config.py` (`IdentityModel`, `TrackerParams`, `SeparationParams`)
  — no magic literals in the logic. 17 tests pass on stdlib alone.

## In progress (2026-05-30)

1. **LLR progress bar on UNKNOWN cards** — When a voiceprint is enrolled, UNKNOWN
   contacts that are being identified show a progress bar: the SPRT's LLR building
   toward the Wald bound. "Identifying... 40%... 70%... ALAN." Real-time Name That
   Tune feedback so the user sees the system working, not a sudden flip.

2. **Persist SPRT state across restarts** — Save the LLR accumulators to disk so a
   server restart doesn't mean cold start. Known voices re-lock in 1 frame instead
   of rebuilding from zero.

## Known issues / next work (prioritized)

1. **Quality-gated identity accumulator (THE BLOCKER)** — The tracker spawns new
   contacts frame-to-frame instead of collecting good embeddings into one identity.
   Fix: decouple identity accumulation from the tracker's frame-to-frame association.
   Collect *only* high-quality frames (Speech-classified, high energy, above VAD) into
   a per-source centroid that persists across tracker thrash. Match against enrolled DB
   progressively — confidence grows as good chunks coalesce, doesn't require continuous
   signal. This is how Shazam, Content ID, pyannote, and every production system works:
   quality gate → per-source accumulator → progressive match → decay.
   The SPRT math is correct; it's being starved of evidence because the tracker resets it.

2. **Calibration + enrollment UX in the web HUD** — Move the calibrate flow out of CLI
   into the neumorphic HUD. 20s enrollment with guided prompts ("SPEAK" / "SHUT UP" /
   "THE WORLD"), real-time level feedback, d-prime readiness display, and a countdown
   so the user knows when recording starts. The CLI version works but the timing is
   invisible — user can't tell when to talk.

3. **Feed-forward attention filter** — The gate should feed state back to the beginning
   of the next loop. Ambient sources get lightweight checks instead of full processing.
   Design doc: `docs/ATTENTION_GATE_DESIGN.md`.

4. **Gemma LLM loop** — Wire surfaced sources (via `llm_payload(amap)`) to local Gemma (Ollama). Target < 3s to first token.

5. **Live wake word** — openWakeWord ("Hey Radar") to steer focus.

6. **ONNX Runtime** — Replace PyTorch inference with ONNX + CoreML EP for production speed.

---

## Design docs

- [`docs/ATTENTION_GATE_DESIGN.md`](ATTENTION_GATE_DESIGN.md) — Feed-forward filter: gate output steers next frame's processing budget
- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — Full system architecture
- [`docs/MODELS.md`](MODELS.md) — Model selection, licensing, engine plan
- [`docs/ROADMAP.md`](ROADMAP.md) — Feature roadmap
- [`docs/PRIVACY.md`](PRIVACY.md) — Privacy guarantees
- [`docs/PERFORMANCE.md`](PERFORMANCE.md) — Latency analysis

---

## Environment

```
conda env: daredevil (Python 3.11, miniforge)
torch: 2.12.0 (MPS)
speechbrain: 1.1.0
asteroid: 0.7.0
librosa: 0.11.0
panns-inference: 0.1.1
mcp: 1.27.1
Platform: macOS Darwin 24.6.0, Apple Silicon
```

---

## Expected output (sanity check)

```
python -m pytest -q                    # 9 passed
python -m daredevil.demo --json        # awareness map with reference backends
python -m daredevil serve --live       # HUD at :8770, 2 sources per frame, ~160ms
tail /tmp/daredevil.log                # per-frame: capture, separation, tracker, LLR
```
