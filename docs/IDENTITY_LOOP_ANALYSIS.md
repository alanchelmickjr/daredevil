# Identity Loop Analysis — 2026-05-31

## Symptom

Hundreds of HUD boxes accumulate for a single speaker. Identity never confirms.
Captions build up for minutes, then dump as one huge block with no utterance
separation. Worked for one person. Broke when multi-target support was added.

## What worked (single target)

The SPRT accumulated LLR frame-over-frame against one enrolled speaker. The
tracker had one track. The key was stable. Identity confirmed in 2-3 strong
frames. Captions grouped naturally because there was only one source ID.

## What broke (multi-target addition)

Three independent systems now process the same frame embedding with no
coordination. Each makes its own "same source?" decision. They fight.

---

## Full frame-by-frame data flow

```
Frame arrives (1s audio chunk)
  |
  +-- Stage 1: spatial -> SpatialSource (mono audio, azimuth)
  |
  +-- Separation: ConvTasNet splits -> 2 streams
  |     _active_streams filters: keeps 1 (or 2 if genuinely distinct)
  |
  +-- Stage 2: EmbeddingSlot -> 192-dim ECAPA vector (one per source)
  |
  +-- tracker.assign(emb, position, event_class) -> track_id "UNKNOWN-NNN"
  |     cosine(emb, track["vector"]) where vector is UNNORMALIZED SUM
  |     if cos > 0.55 -> match -> _update (vector += emb, no renorm)
  |     if no match -> _new() -> UNKNOWN-{counter++}
  |
  +-- accumulator.ingest(emb, quality) -> acc_id (integer)
  |     cosine(emb, source.centroid) where centroid is NORMALIZED EMA
  |     if cos > 0.40 -> update existing source
  |     else -> new source
  |     OUTPUT: centroid_confidence (reported but not consumed)
  |
  +-- enrollment.match(emb, energy, key="global") -> match result
  |     cosine(emb, enrolled_voiceprint) -> raw score s
  |     frame_llr = log_gaussian(s|H1) - log_gaussian(s|H0)
  |     acc = (1 - leak) * prev_llr + quality * frame_llr
  |     KEY is ("global", speaker_name)
  |     if acc >= A -> decided = True -> identity confirmed
  |
  +-- enrollment.retain(tracker.live_ids())
  |     live_ids() returns ["UNKNOWN-001", "UNKNOWN-002", ...]
  |     retain keeps entries where key[0] is in that set
  |     "global" is NEVER in that set
  |     *** LLR IS WIPED EVERY SINGLE FRAME ***
  |
  +-- Router -> awareness map -> HUD
```

---

## Three bugs, one cause

### Bug 1: SPRT never accumulates

`pipeline.py:242` calls `self.enrollment.retain(self.tracker.live_ids())`.
`pipeline.py:196` passes `key="global"` to the SPRT.

`retain()` keeps only keys where `key[0]` is in `live_ids()`. The live IDs are
`["UNKNOWN-001", "UNKNOWN-002", ...]`. The string `"global"` is never in that
set. The LLR accumulator is wiped to zero every frame. The SPRT can never cross
the Wald bound. Identity never confirms.

### Bug 2: Tracker centroid blurs -> track proliferation

`tracker.py:78-79` — `_update` sums the new embedding onto the track vector
without normalizing. `cosine()` handles magnitude implicitly so the direction
is what matters. But after 20+ frames of slightly varying vocal quality, the
summed vector converges toward a generic "average speech" direction. Vocal
shifts, breaths, or pauses produce embeddings that fall below the 0.55 gate
against this blurred centroid. Result: new track opens. Each track gets a HUD
box.

With one speaker this was tolerable — the centroid stayed close enough because
every frame was the same person. With multiple speakers or longer sessions, the
centroid drifts toward the room mean and discrimination collapses.

### Bug 3: Transcription accumulates without boundaries

`server.py:298` feeds audio to `transcriber.feed(self._focus_id, audio, sr,
is_speaking)`. The transcriber buffers by source_id and flushes only on pause
(`check_pauses`: active goes False while buffer is non-empty).

Since identity never confirms (Bug 1), the focus is set by raw track ID. But
track IDs keep changing (Bug 2). Two failure modes:

- If focus tracks a stale ID: audio buffers indefinitely because no new frames
  arrive for that ID, so `is_speaking` stays True forever on that key. Buffer
  grows. Eventually dumps as one huge caption.
- If focus gets re-pointed: the old buffer orphans (never flushes), the new one
  starts fresh. Text fragments have no continuity.

---

## The circular dependency

```
Identity can't confirm  -->  because LLR is wiped
LLR is wiped            -->  because retain() prunes "global"
retain() prunes         -->  because it's keyed to tracker IDs
tracker IDs proliferate -->  because centroid blurs / identity can't consolidate tracks
identity can't consolidate -> because identity can't confirm
```

---

## The accumulator: wired but inert

`accumulator.py` was designed to break this cycle by building quality-gated
centroids independent of the tracker. It outputs `centroid_confidence` which is
reported in the awareness map but consumed by nothing. The SPRT doesn't use the
accumulated centroid. The tracker doesn't consult it. The HUD doesn't gate on
it (display threshold exists in config but the JS doesn't filter by it).

---

## How this was already solved — 80 years of prior art

### Passive sonar contact management (1940s-1970s)

SOSUS tracked dozens of submarine contacts simultaneously using analog hardware
(filter banks + paper LOFARgrams). The architecture was:

```
Hydrophones -> Beamformer (delay lines) -> Narrowband filter bank -> LOFARgram display
                                                                          |
                                                              Operator assigns designators
                                                              (Sierra-1, Sierra-2, ...)
                                                                          |
                                                              Cross-bearing fixes -> track
```

**The key insight: a contact IS its trajectory through observation space
(bearing × time × frequency). Identity is never an averaged embedding.**

- Tonal lines (machinery harmonics at specific frequencies) form a spectral
  barcode unique to each vessel
- The track is kinematic (position, velocity), updated by each new bearing fix
  via a filter (Kalman or alpha-beta)
- The signature (tonal frequencies) is used for *classification* and
  *re-association*, NOT for state estimation
- These are separate concerns: WHERE (track state) vs WHAT (classifier)

**This avoids centroid drift entirely** because:
- Track state is a position/velocity, not an averaged signature
- The signature is *observed*, not accumulated
- Re-association uses gating (does this new detection fall within the
  uncertainty region of an extrapolated track?) not cosine-on-averaged-vectors

### Frequency-domain ICA for source separation

The paper "Separation of passive sonar target signals using frequency domain
ICA" (Parhizkar et al., 2016) and the Independent Vector Analysis literature
(Hiroe 2006, Kim et al. 2006, Ono 2011) show:

```
Multichannel input -> STFT -> Per-bin ICA separation -> Permutation alignment -> ISTFT
```

The critical problem is **permutation ambiguity**: at each frequency bin, ICA
separates the sources but assigns them to output channels arbitrarily. Solutions:

1. **Envelope correlation** (Murata 2001): sources that belong together have
   correlated temporal envelopes across frequency bins
2. **Mixing vector continuity** (Sawada 2004): transfer functions are smooth
   across frequency, so the mixing vectors' directions align smoothly
3. **DOA-based alignment**: the steering vector at each bin encodes direction;
   group outputs pointing the same way
4. **Independent Vector Analysis (IVA)**: treat all frequency bins of one source
   as a single multidimensional variable — solves permutation by construction

**The fix to permutation is structural coupling**, not averaging. The DAW analog:
each source gets its own channel from the start and they never mix. The sonar
analog: each contact has a bearing-time trajectory and tonal signature that
lives in its own lane.

### SPRT in sonar detection — how accumulators are keyed

In operational sonar:
- Each **resolution cell** (beam index × frequency bin) has its own accumulator
- The cell is defined by spatial-spectral position, not by embedding similarity
- Energy in a cell either builds the accumulator toward detection threshold or
  doesn't — the cell is the stable address
- When a cell's accumulator crosses threshold: a **detection** is declared
- Above that: the **tracker** forms tracks from sequences of detections
- Above that: the **classifier** identifies what the track IS

The hierarchy is: detect → track → classify. Each layer has one job.

### DAW signal flow — why a guitarist separates in seconds

A DAW never has the identity problem because it prevents it architecturally:

```
Input (multi-track) -> Channel strips (independent, parallel)
                           -> Insert effects (serial per channel)
                           -> Sends (parallel taps)
                           -> Fader/Pan
                           -> Bus assignment
                           -> Master
```

Each source has its own channel from recording time. They are never blended
into a single centroid. The architecture maintains independence by construction.

For AI separation (Demucs/HTDemucs): fixed output heads trained with labeled
data. Output 0 is always vocals. Output 1 is always drums. The architecture
assigns, not a similarity score. **Permutation Invariant Training (PIT)** solves
the training problem; at inference, overlap-based stitching maintains identity
across segments by comparing actual signal overlap — NOT by averaging embeddings.

### The leaky integrator SPRT (operational sonar variant)

```
S_i = gamma * S_{i-1} + log_LR_i,    0 < gamma < 1
```

- Persistent signals build up and stay detected
- Fading signals gradually lose evidence (but don't reset to zero)
- A gap followed by reappearance doesn't need to rebuild from scratch if the
  gap is shorter than the time constant
- The accumulator lives at a **stable address** (the cell/track), not at a
  key that gets pruned

---

## Complete file map

```
daredevil/
├── __init__.py              version string
├── __main__.py              entry point: delegates to cli.main()
├── config.py                ALL tunables: PriorityWeights, Thresholds, IdentityModel,
│                            TrackerParams, SeparationParams, NMFParams, Config,
│                            SAFETY_CRITICAL_CLASSES, detect_backend(), load_calibration()
├── pipeline.py              ORCHESTRATOR: Stage1 -> separation -> Stage2 (parallel) ->
│                            tracker -> accumulator -> SPRT match -> router -> awareness map
│                            *** THE BUGS LIVE HERE (key="global", retain mismatch) ***
├── calibrate.py             First-run onboarding: captures voice+room, fits IdentityModel,
│                            saves calibration.json. Measures d-prime.
├── demo.py                  Synthetic end-to-end demo
├── cli.py                   CLI dispatcher: demo, enroll, listen, serve, calibrate, bench, devices
├── enroll.py                Enrollment CLI helper
├── mcp_server.py            MCP tool interface for Claude/agents
│
├── audio/
│   ├── __init__.py
│   ├── capture.py           MicStream (ring buffer, persistent), capture_live, capture_file,
│   │                        synthetic_scene, synthetic_voice. DEFAULT_SCENE definition.
│   └── utils.py             STDLIB DSP: rms, cosine, resample, fingerprint, spectral_centroid,
│                            zero_crossing_rate, is_speech_quality, to_mono. numpy optional.
│
├── stage1/                  SPATIAL DECOMPOSITION (WHERE)
│   ├── __init__.py
│   ├── mic_arrays.py        MicArray dataclass, SINGLE/MACBOOK_3/RESPEAKER_4 geometries,
│   │                        detect(), load_coordinate_map() — geometry-agnostic hook
│   ├── spatial.py           Stage1: SRP-PHAT DOA via pyroomacoustics, degrades to mono.
│   │                        Synthetic path: scene_truth as perfect separation.
│   ├── separation.py        ConvTasNet source separator (Asteroid). Fallback: pass-through.
│   │                        Splits mixed audio into N streams before slots analyze.
│   └── nmf.py               SpectralLibrary: NMF decomposition for frame-stable tracking
│                            features. Fixed triangular basis (numpy) / band-group (stdlib).
│                            learn_basis() ready but not wired to online learning.
│
├── stage2/                  PARALLEL SLOT BANK (WHO / WHAT / HOW)
│   ├── __init__.py
│   ├── base.py              Slot ABC: warmup(), run(audio, sr, ctx) -> dict
│   ├── embedding.py         Slot A — WHO: ECAPA-TDNN (speechbrain) / fingerprint fallback.
│   │                        192-dim L2-normalized voiceprint. THE HEADLINE CAPABILITY.
│   ├── events.py            Slot B — WHAT: PANNs CNN14 (527 AudioSet classes) / heuristic.
│   │                        Safety-critical flag (is_safety_critical).
│   └── prosody.py           Slot C — HOW: librosa pyin / opensmile eGeMAPS / stdlib proxies.
│                            Distress scalar [0,1] + state label (calm/stressed/distressed).
│
├── stage3/                  TRACKING + ROUTING (the broken layer)
│   ├── __init__.py
│   ├── tracker.py           UnknownTracker: M-of-N confirmation, bearing alpha-beta filter,
│   │                        coast/delete lifecycle. Associates by cosine on ACCUMULATED SUM.
│   │                        *** BUG: centroid drifts, cosine degrades over time ***
│   ├── accumulator.py       IdentityAccumulator: quality-gated EMA centroids, independent
│   │                        of tracker. Outputs centroid_confidence. INERT — nothing consumes.
│   └── router.py            AttentionRouter: priority scoring (patent Eq. 2), safety/distress
│                            overrides, attention gate (surface vs ambient). Builds the
│                            awareness map JSON — the product.
│
├── enrollment/
│   ├── __init__.py
│   └── manager.py           EnrollmentManager: Wald SPRT per (key, speaker). Welford multi-
│                            sample enrollment. CFAR background adaptation. Hysteresis hold.
│                            *** BUG: key="global" gets wiped by retain() every frame ***
│
├── fleet/                   P2P SYNC (identity portability)
│   ├── __init__.py
│   ├── store.py             LocalStore (JSON + encrypt-at-rest), GunStore (+ peer sync).
│   │                        make_store() factory.
│   ├── crypto.py            Fernet AES when DAREDEVIL_KEY set; base64 fallback. Never raw audio.
│   └── gun-relay/           Node.js Gun peer relay (the fleet backbone)
│       ├── package.json
│       ├── relay.js
│       └── README.md
│
├── viz/                     DISPLAY LAYER
│   ├── __init__.py
│   ├── server.py            Web HUD server (http://127.0.0.1:8770). _State holds pipeline.
│   │                        awareness() -> listen() + transcript/LLM routing.
│   │                        _CalibrationSession for in-browser onboarding.
│   │                        *** BUG: transcriber.feed uses focus_id that changes constantly ***
│   ├── transcriber.py       Whisper-cli STT. Buffers per source_id, flushes on pause.
│   │                        *** BUG: never flushes because focus_id is unstable ***
│   ├── spatial_map.py       render_ascii (terminal radar), render_matplotlib (polar),
│   │                        render_spectrogram (chaos-in/structure-out), render_radar_hud.
│   └── web/
│       └── index.html       Neumorphic-steampunk HUD. Orbital radar + cards + captions.
│                            *** Renders every source as a card. Track proliferation = UI flood ***
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── ATTENTION_GATE_DESIGN.md
│   ├── BUILD_SPEC.md
│   ├── ENROLLMENT_UX_RESEARCH.md
│   ├── IDENTITY_LOOP_ANALYSIS.md    (this document)
│   ├── MODELS.md
│   ├── NMF_TRACKING_DESIGN.md       (NMF for frame-stable association — partly wired)
│   ├── PERFORMANCE.md
│   ├── PRIVACY.md
│   ├── ROADMAP.md
│   └── TESTING_SPRT.md              (test guide for the SPRT, written before the bug)
│
├── tests/
│   ├── test_calibrate.py
│   ├── test_core.py
│   ├── test_matching.py
│   └── test_nmf.py
│
├── pyproject.toml
├── README.md
└── CLAUDE.md
```

---

## The sonar lesson applied to Daredevil

### What sonar does (and Daredevil should)

```
DETECT:   energy in a resolution cell exceeds threshold (per-cell SPRT)
TRACK:    sequence of detections at consistent bearing form a trajectory
CLASSIFY: tonal signature / embedding identifies WHAT the track IS
```

Three layers. Each has one job. No layer tries to do the other's work.

### What Daredevil currently does (broken)

```
DETECT + TRACK + CLASSIFY all in one pass:
  tracker.assign() tries to answer WHICH by cosine-on-accumulated-sum
  accumulator.ingest() tries to answer WHICH by cosine-on-EMA-centroid
  enrollment.match() tries to answer WHO but its key gets wiped
  router.build() tries to display everything, gets 100 boxes
```

### The architectural error

The tracker was asked to do two jobs:
1. Maintain spatial/temporal continuity (WHERE/WHEN — its real job)
2. Build identity evidence by accumulating embeddings (WHO — not its job)

It does neither well because the approach to (2) — summing embeddings —
degrades the signal needed for (1). And the system designed for (2) — the
IdentityAccumulator — was wired in but never connected to anything that
makes decisions.

### What "cake on a Tuesday" looks like

The sonar architecture, translated:

```
1. DETECT:  is_speech_quality() — energy + ZCR gate → this frame has speech
2. TRACK:   associate by bearing + NMF spectral features (frame-stable)
            → stable track ID per physical source
            Track state: position + velocity (alpha-beta filter, already built)
            Track does NOT accumulate identity embeddings
3. CLASSIFY: SPRT runs per track, keyed by track_id
            SPRT key persists as long as the track lives
            retain() prunes only when the track dies (which is correct)
            Identity confirms → track upgrades from UNKNOWN-NNN to "alan"
```

The track is the **stable address** that the SPRT accumulates against. The
track's job is only spatial/temporal continuity. Identity is a label that
gets attached to the track once the SPRT fires. Exactly how a sonar operator
writes "Sierra-1 = Akula-class" on the plot once classification is confident.

The NMF spectral features (already built in `stage1/nmf.py`, already designed
in `docs/NMF_TRACKING_DESIGN.md`) are the correct association signal for the
tracker — they answer "same ongoing sound?" which is frame-stable, unlike
ECAPA embeddings which answer "same speaker?" over utterances.

---

## What needs to be decided (not prescribed)

The original single-target design had one path: frame -> SPRT -> identity. It
worked. The multi-target addition introduced two extra systems (tracker,
accumulator) but didn't decide who owns what:

- **WHO** (identity): SPRT against enrollment — needs a stable key that persists
  across frames for the same physical speaker.
- **WHICH** (association): some system must answer "is this frame from the same
  physical source as the last frame?" — that's what gives the SPRT its key.
- **WHERE** (spatial): azimuth tracking, bearing filter — currently entangled
  with WHO in the tracker.

The three systems each try to answer WHICH independently and disagree. The fix
is: pick one owner for WHICH, feed its output as the SPRT key, and remove the
redundant association logic.
