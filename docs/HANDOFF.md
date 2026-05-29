# Daredevil — Handoff & CLI Reference

**As of commit `5711eb4` · branch `claude/fervent-faraday-rx9Ns` · 2026-05-29**

Read `CLAUDE.md` first (goals + guardrails). This doc is the point-in-time
snapshot: how to run everything, what's real vs. stubbed, and what's next.

---

## Pull & build

```bash
git fetch origin
git checkout claude/fervent-faraday-rx9Ns
git pull

pip install -e .            # core: pure-Python, zero heavy deps — runs immediately
# optional accelerators (install what you want; slots auto-upgrade fallback -> real):
pip install -e ".[speaker]" #  WHO   — SpeechBrain ECAPA (downloads ~weights on first run)
pip install -e ".[events]"  #  WHAT  — PANNs CNN14
pip install -e ".[prosody]" #  HOW   — librosa
pip install -e ".[spatial]" #  WHERE — pyroomacoustics (+ numpy/scipy)
pip install -e ".[audio]"   #  live mic (sounddevice)
pip install -e ".[viz]"     #  matplotlib radar/spectrogram PNGs
pip install -e ".[full]"    #  everything
```

The core has **no required dependencies** — `python -m daredevil.demo` works on a
fresh checkout with nothing else installed (pure-Python fallback). Heavy backends
only *accelerate* the same pipeline.

---

## CLI reference

Two entry points: the installed `daredevil` script and `python -m daredevil`.

| Command | What it does |
|---|---|
| `daredevil demo` | End-to-end demo on the synthetic scene (enroll → listen → awareness map → radar). |
| `daredevil demo --live` | Same, but uses the real microphone (needs `[audio]`). |
| `daredevil demo --file scene.wav` | Run against a recorded multi-source WAV. |
| `daredevil demo --spectrogram out.png` | Also render the spectrogram + overlay PNG (needs `[viz]`). |
| `daredevil demo --save-png radar.png` | Also render the polar radar PNG. |
| `daredevil demo --json` | Print only `{enrollment, awareness_map}` JSON. |
| `daredevil serve [--port 8770] [--live]` | Launch the **web HUD** at `http://127.0.0.1:8770`. |
| `daredevil bench [--iters 10]` | Crowd-scaling benchmark: pipeline latency vs. source count; shows → LLM stays flat. |
| `daredevil enroll --name NAME [-s 3] [--live]` | Enroll a speaker; prints the `C(t)=1-e^(-t/3)` confidence. |
| `daredevil listen [--duration 1.0] [--live] [--file W] [--json]` | Emit one awareness map (radar or JSON). |
| `daredevil devices` | Detected mic array + which backends are installed. |
| `daredevil version` | Version. |
| `python -m daredevil.enroll --name alan --seconds 3` | Standalone enrollment entry point. |

**Quick demo for someone new:**
```bash
python -m daredevil.demo      # see the awareness map + radar (synthetic, labeled)
python -m daredevil serve     # open http://127.0.0.1:8770  — the orbital HUD
python -m daredevil bench     # the "flat in a crowd" proof
```

---

## What's real vs. stubbed (be honest in the demo)

**Verified working (pure-Python, tested here):**
- Three-stage pipeline; parallel slot execution + timing.
- WHO: 3s enrollment, cosine match, clip-length-invariant fallback fingerprint.
- Attention gate: `attention: surface|ambient` + `routed_to_llm`; the radio is gated out.
- Stage-3 priority (patent Eq. 2) + `SAFETY_CRITICAL` / `DISTRESS` overrides; `UNKNOWN-NNN` tracking.
- Structured JSON awareness map; ASCII radar; matplotlib HUD + spectrogram.
- Local-first identity store + Gun fleet scaffold (encrypted-at-rest hook).
- Web HUD + `serve` (verified end-to-end), `bench`, `9` tests passing.

**Stubbed / not yet validated (next work):**
- Real ML backends (ECAPA / PANNs / librosa): integration code is written but **not run**
  here (no torch / can't download weights). They auto-upgrade `fallback → reference`
  when the extra is installed; first real run is unverified.
- STT (`/probe` in the HUD): placeholder text — local whisper.cpp to wire in.
- LLM loop (Gemma via Ollama): not wired yet.
- Wake word: `config.wake_word` + HUD chip only; openWakeWord not yet listening.
- Live multi-mic SRP-PHAT: code present, not validated on a real array.

**Known perf note:** the → LLM payload stays **flat** as sources grow (the gate), but the
pure-Python perception front-end climbs (~48 ms @ 2 sources → ~480 ms @ 20) and crosses
200 ms around ~10 simultaneous sources. Fix = cheap pre-gate + per-source parallelism +
ONNX int8. See `docs/PERFORMANCE.md`.

---

## Expected output after pulling (sanity check)

```
python -m pytest -q          # 9 passed
python -m daredevil bench    # sources 2/5/10/20 -> pipeline_ms rises, "→ LLM" stays 2
python -m daredevil.demo     # awareness map with alan (enrolled), UNKNOWN baby (SAFETY), radio (ambient/gated)
```

---

## Next tasks (prioritized)

1. **Wire ECAPA live** — `pip install -e ".[speaker]"`, run `daredevil demo --live`, confirm
   it identifies a real enrolled voice (replaces the fallback fingerprint). Headline capability.
2. **Gemma loop** — surfaced sources (`llm_payload(amap)`) → local Gemma (Ollama, E2B/E4B) →
   streamed reply; target < 3 s to first token. Keep it local.
3. **Front-end speed** — per-source parallelism + ONNX-Runtime int8 (CoreML on Mac) to hold
   < 200 ms even in a crowd.
4. **Live wake word** — openWakeWord ("Hey Radar") to grab/steer focus; click → whisper STT in the HUD.
5. **Live spatial** — validate SRP-PHAT on a real multi-mic array.

---

## Map of the code

```
daredevil/
  config.py        thresholds/weights (patent Eq.2), safety classes, wake_word, backend detect
  pipeline.py      Stage1 -> parallel Stage2 -> Stage3; listen()/enroll(); timing; scene= for crowds
  audio/           capture.py (live/file/synthetic + crowds), utils.py (stdlib DSP, fingerprint)
  stage1/          mic_arrays.py (geometries + coordinate-map loader), spatial.py (SRP-PHAT)
  stage2/          base.py (Slot), embedding.py (WHO), events.py (WHAT), prosody.py (HOW)
  stage3/          tracker.py (UNKNOWN-NNN), router.py (priority + gate + llm_payload)
  enrollment/      manager.py (enroll/match; C(t)=1-exp(-t/tau))
  fleet/           store.py (Local/Gun), crypto.py (Fernet at-rest), gun-relay/ (Node peer)
  viz/             spatial_map.py (ascii/radar/spectrogram), server.py + web/index.html (HUD)
  demo.py cli.py enroll.py __main__.py
docs/              ARCHITECTURE · MODELS · ROADMAP · PRIVACY · PERFORMANCE · BUILD_SPEC · HANDOFF
tests/             test_core.py (stdlib-only)
```

Guardrails (full list in `CLAUDE.md`): stdlib-only core, no cloud ever, persist only
non-reversible embeddings, no hardware IP in the repo, no third-party names, honest demos.
