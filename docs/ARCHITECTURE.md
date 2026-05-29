# Architecture

Daredevil is a **three-stage parallel pipeline**. The design mirrors the
patent-pending firmware architecture so the software and the hardware module
speak the same language — but everything here runs on commodity machines.

```
Mic array (laptop / USB / 3D module — any geometry)
        │
        ▼
┌─ STAGE 1 ── Spatial signal processing (continuous, deterministic) ──┐
│  capture → resample → SRP-PHAT DOA → beamform / separate            │
│  geometry-agnostic: driven by a coordinate map, adapts to any array │
└─────────────────────────────────────────────────────────────────────┘
        │  one (mono) or many (separated) sources
        ▼
┌─ STAGE 2 ── Parallel inference bank (all slots run concurrently) ───┐
│  Slot A · WHO    speaker embedding (192-dim) ──┐                    │
│  Slot B · WHAT   acoustic event + safety flag ─┤  latency =         │
│  Slot C · HOW    prosody → emotional state ────┤  max(slot),        │
│  Slot D · user   pluggable int8/ONNX/torch ────┘  NOT sum(slots)    │
└─────────────────────────────────────────────────────────────────────┘
        │  per-source slot results
        ▼
┌─ STAGE 3 ── Attention router (fusion) ──────────────────────────────┐
│  identity match (cosine) + position + event + prosody               │
│  → priority score → overrides → structured JSON awareness map       │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
   LLM-ready awareness map (the product)
```

## Stage 1 — Spatial (`stage1/`)

- **Geometry-agnostic** (patent Claim 5): the array is described by a coordinate
  map (`mic_arrays.py`); SRP-PHAT consumes the coordinates and adapts. We ship
  `single`, `macbook-3`, `respeaker-4`, and a JSON coordinate-map loader for
  arbitrary arrays (including the hardware module, whose geometry is not shipped).
- **DOA** via `pyroomacoustics` SRP-PHAT (GCC-PHAT). **Separation** via
  delay-and-sum beamforming per direction.
- **Degrades gracefully:** single mic → one undirected source; no
  `pyroomacoustics` → one source. WHO/WHAT/HOW still work.
- **Synthetic mode:** uses scene ground-truth as perfect separation so the
  downstream architecture is exercised identically to a live capture.

## Stage 2 — Parallel inference bank (`stage2/`)

Each slot implements the same `Slot` contract (`base.py`): `warmup()` then
`run(audio, sr, ctx) -> dict`. Slots run in a `ThreadPoolExecutor`. This is
correct because real backends (torch / ONNX / numpy) **release the GIL** during
native compute, so wall-clock latency approaches the slowest slot rather than the
sum (the patent's core claim). The pure-Python fallback is GIL-bound — we say so,
and the speedup widens once real backends are installed. Timing is always
measured, never simulated.

| Slot | Question | Output keys |
|---|---|---|
| A `embedding` | **WHO** | `vector`, `dim`, `backend` |
| B `events` | **WHAT** | `class`, `confidence`, `safety_critical`, `topk` |
| C `prosody` | **HOW** | `state`, `distress`, `features` |
| D user | anything | (your schema) |

Slot D is the extensibility point (patent Claims 4 & 13): load any model that
honors the interface — language ID, keyword spotting, cough detection, etc.

## Stage 3 — Attention router (`stage3/`)

**Priority** (patent Eq. 2):

```
P = w_id·S_identity + w_event·S_event + w_prosody·S_prosody + w_temporal·S_temporal
```

with `S_identity = cosine_match · enrollment_confidence`, `S_event = confidence`
for safety-critical events, `S_prosody = distress`, `S_temporal = recency`.
Default weights `(0.35, 0.30, 0.20, 0.15)` in `config.py`.

**Overrides:**
- `SAFETY_CRITICAL` — a safety-critical event above `T_safety` pins priority to
  `1.0` (baby cry, alarm, siren, glass, gunshot, …; see `SAFETY_CRITICAL_CLASSES`).
- `DISTRESS` — an enrolled speaker whose distress exceeds `T_distress` is
  escalated regardless of event class (patent Claims 3 & 14).

**Tracking** (`tracker.py`, patent Claim 6): unknown sources get a persistent
`UNKNOWN-NNN` id by matching embeddings across frames, enabling temporal tracking
without enrollment.

## The engine & backends

A backend-agnostic slot runtime — the software analogue of the patent's
multi-core slot bank. Three tiers:

1. **Reference (PyTorch)** — fastest path to working models; enrollment.
2. **Portable (ONNX Runtime)** — one model file, int8, hardware EPs (CoreML on
   Mac, CUDA/TensorRT on Orin). *Next milestone.*
3. **Fallback (pure stdlib)** — heuristic slots so the map computes anywhere.

`detect_backend()` picks CUDA / MPS / CPU / fallback. See `docs/MODELS.md`.

## Awareness-map schema (the LLM contract)

```jsonc
{
  "timestamp": "ISO-8601",
  "backend": "fallback|cpu|mps|cuda",
  "array": { "name": "macbook-3", "n_mics": 3, "spatial": true },
  "sources": [
    {
      "id": "alan" | "UNKNOWN-001",
      "type": "enrolled" | "unknown",
      "event":   { "class": "...", "confidence": 0.0, "safety_critical": true },
      "prosody": { "state": "calm|stressed|confused|distressed", "distress": 0.0 },
      "identity": { "confidence": 0.0, "match_score": 0.0, "enrollment_confidence": 0.0 },
      "position": { "azimuth": 0.0, "elevation": 0.0 },   // omitted if no spatial
      "priority": 0.0,
      "priority_override": "SAFETY_CRITICAL|DISTRESS"      // only when triggered
    }
  ],
  "timing": { "parallel_ms": 0.0, "sequential_ms": 0.0 },
  "privacy": { "cloud_used": false, "raw_audio_stored": false, "embeddings": "non-reversible" }
}
```

Sources are sorted by `priority` (highest first) so the LLM gets the attention
order for free.
