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

## Recently fixed (WHO matching + tracking stitch)

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

## Known issues / next work (prioritized)

1. **Feed-forward attention filter** — The gate should feed state back to the beginning of the next loop. Ambient sources get lightweight checks instead of full processing. Design doc: `docs/ATTENTION_GATE_DESIGN.md`.

2. **Calibrate the SPRT score models on real ECAPA** — `IdentityModel` defaults
   (`target_mean=0.65`, `impostor_mean=0.18`, …) are from published VoxCeleb stats;
   measure your own enrolled cohort and tune them (and `alpha`/`beta`) to the live mic.

3. **Gemma LLM loop** — Wire surfaced sources (via `llm_payload(amap)`) to local Gemma (Ollama). Target < 3s to first token.

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
