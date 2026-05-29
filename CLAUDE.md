# CLAUDE.md — engineering guide for Daredevil

Guidance for Claude Code (and humans) working in this repo. Read this first.

> **Current state, CLI reference, and next steps:** see [`docs/HANDOFF.md`](docs/HANDOFF.md).

## What this is

Daredevil is an open-source SDK that converts raw microphone audio into a
**structured acoustic awareness map** for LLMs: **WHO** is speaking (speaker ID),
**WHERE** they are (spatial DOA), **WHAT** is happening (acoustic events), and
**HOW** they sound (prosody/emotion). All inference is **local**. The awareness
map (a JSON dict) *is* the product.

Thesis: *"To be scalable, it has to run where the AI agents are."* So the SDK
must run on any laptop with zero special hardware, and accelerate on device.

## Non-negotiable rules

1. **The core stays pure-Python (stdlib only).** `pip install daredevil` with no
   extras must import and run the full pipeline. Heavy libraries (`torch`,
   `speechbrain`, `panns_inference`, `librosa`, `pyroomacoustics`, `sounddevice`,
   `numpy`, `matplotlib`, `cryptography`) are **optional** — import them *lazily,
   inside functions, wrapped in try/except*, and always provide a deterministic
   fallback. Never add a hard dependency to the core.
2. **No cloud, ever.** `Config.allow_cloud` is `False` and there must be no code
   path that sends audio or embeddings off-device. Local inference only. (This is
   why the old Hume prosody API was removed.)
3. **Privacy by construction.** Persist only non-reversible embedding vectors —
   never raw audio. Encrypt voiceprints at rest. Keep `privacy` truthful in the
   output.
4. **IP boundary (this repo is PUBLIC).** The *software* is MIT and open. The
   *hardware module + firmware architecture* are patent-pending and confidential.
   **Do NOT commit** the bill of materials, part numbers, exact mic-array
   coordinates, schematics, or pin assignments. Hardware references stay
   high-level (the "bridge statement" level). Arbitrary array geometries are
   supported via a user-provided coordinate map (`mic_arrays.load_coordinate_map`)
   — that's the geometry-agnostic hook; we don't ship the module's geometry.
5. **WHO is the priority.** Speaker identity is the headline capability. When in
   doubt, make identification solid first.
6. **No personal names of third parties** (e.g. the demo recipient) anywhere in
   committed files.
7. **Be honest in the demo.** The synthetic scene is labeled SYNTHETIC. Simulated
   latencies are labeled `"simulated"`. Don't present fabricated numbers as live.

## Scope fences

Goals: **WHO-first, local, honest, open-source software.** Within that, these are
load-bearing — a future session should treat them as hard boundaries.

**Always**
- Keep the core importable + runnable on the **stdlib alone**; every heavy backend
  is optional, lazy, guarded, with a deterministic fallback.
- Run the tests **and** the demo before pushing; if something is unverified, say so.
- New tunables go in `config.py`; a new capability is a new `Slot`; keep the
  awareness-map schema stable.

**Never**
- Never add a required/heavy dependency to the core.
- Never add a code path that sends audio or embeddings off-device — no cloud, no
  telemetry. `allow_cloud` stays `False`.
- Never persist or transmit raw audio. Embeddings only, encrypted at rest.
- Never commit hardware IP: BOM, part numbers, exact mic coordinates, schematics,
  pin assignments, firmware. High-level "bridge statement" only.
- Never put third-party personal names in committed files.
- Never present synthetic/simulated numbers as live.
- Never open a PR, or push to a branch other than the working branch, without an
  explicit ask.

**In scope (this repo)**
- The open-source SDK: capture, spatial, the four slots, router, enrollment, the
  Gun fleet layer, viz, demo, CLI, docs — and wiring real OSS models as optional
  backends.

**Out of scope (here)**
- Hardware/firmware design artifacts and the module's array geometry — confidential,
  they live elsewhere. Cloud services or accounts of any kind.

## The graceful-degradation pattern (used everywhere)

```python
class XSlot(Slot):
    def warmup(self):
        try:
            import heavy_lib            # lazy + guarded
            self._model = heavy_lib.load(...)
            self._backend_name = "reference"
        except Exception:
            self._backend_name = "fallback"
    def run(self, audio, sr, ctx=None):
        if self._model is not None:
            try: return real_inference(...)
            except Exception: pass
        return heuristic_fallback(...)   # always works, deterministic
```

In synthetic mode, fallback slots may read `ctx["truth"]` (ground truth from the
scene) so the demo is crisp; real audio paths ignore it and use models/features.

## Layout

```
daredevil/
  config.py            thresholds, weights (patent Eq. 2), safety classes, backend detect
  pipeline.py          orchestrator: Stage1 -> parallel Stage2 -> Stage3; timing
  audio/
    capture.py         CaptureResult; live (sounddevice) / file (wav) / synthetic scene
    utils.py           stdlib DSP: resample, rms, cosine, spectral fingerprint (fallback embedding)
  stage1/
    mic_arrays.py      MicArray geometries (single/macbook/respeaker) + coordinate-map loader
    spatial.py         SRP-PHAT DOA (pyroomacoustics) -> SpatialSource list; degrades to mono
  stage2/
    base.py            Slot interface (Slot D = user-loadable)
    embedding.py       Slot A — WHO  (ECAPA / fingerprint fallback)
    events.py          Slot B — WHAT (PANNs / heuristic fallback)
    prosody.py         Slot C — HOW  (librosa|opensmile / stdlib proxies)
  stage3/
    tracker.py         persistent UNKNOWN-NNN identifiers
    router.py          priority scoring + SAFETY_CRITICAL / DISTRESS overrides -> awareness map
  enrollment/manager.py   enroll/match; C(t)=1-exp(-t/tau); cosine match (T=0.70)
  fleet/
    store.py           IdentityStore: LocalStore (default) + GunStore (P2P sync)
    crypto.py          encrypt-at-rest (Fernet when DAREDEVIL_KEY set; SEA on JS side)
    gun-relay/         runnable Node.js Gun peer (the fleet backbone)
  viz/spatial_map.py   ASCII radar (always) + matplotlib polar (optional)
  demo.py / cli.py / enroll.py / __main__.py   entry points
tests/                 pure-stdlib tests of the deterministic plumbing
docs/                  ARCHITECTURE, MODELS, ROADMAP, PRIVACY, BUILD_SPEC
```

## Run / test

```bash
pip install -e ".[dev]"        # or nothing — core needs no deps
python -m daredevil.demo       # end-to-end synthetic demo
python -m daredevil.demo --live           # real mic
python -m daredevil.demo --simulate-latency
python -m pytest -q            # tests run on stdlib alone
daredevil devices              # detected array + installed backends
```

## Wiring real backends (next milestone)

Install the relevant extra and the slot's `warmup()` upgrades itself from
`fallback` → `reference` automatically — no other code changes:
`[speaker]` ECAPA, `[events]` PANNs, `[prosody]` librosa, `[spatial]`
pyroomacoustics, `[audio]` sounddevice. First run downloads pretrained weights
(~500MB) into `~/.daredevil/models`. See `docs/MODELS.md` for the engine plan
(reference torch → portable ONNX Runtime with CoreML/TensorRT EPs + int8).

## Conventions

- Match the surrounding style; keep comments purposeful and at the level of *why*.
- New tunables go in `config.py`. New event→safety mappings go in
  `SAFETY_CRITICAL_CLASSES`.
- A new analysis capability is a new `Slot` (Slot D). Keep the contract.
- Keep the awareness-map schema stable; it's the public contract with the LLM.
