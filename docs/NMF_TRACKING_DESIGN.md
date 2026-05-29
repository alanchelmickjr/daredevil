# NMF Spectral Basis Tracking — "Name That Tune in 3 Notes"

## The Problem

ECAPA embeddings have low cosine similarity between consecutive 1-second frames
of the same source because they encode *speaker identity* from full utterances,
not *spectral continuity* across short windows. The tracker needs a different
signal: one that says "this slice belongs to the same ongoing sound" even when
the content changes (different words, different notes).

## The Insight

Every sound source has a spectral signature — a characteristic energy
distribution across frequency bands that persists even as the content varies.
Alan's vocal tract shapes every phoneme the same way. The coffee shop music has
a consistent spectral envelope. The espresso machine has its own basis.

**Non-negative Matrix Factorization (NMF)** decomposes a spectrogram into:
- **W** (basis matrix): the learned spectral templates (each column is one "source type")
- **H** (activation matrix): how much of each basis is present at each time frame

```
V ≈ W × H
[freq × time] ≈ [freq × k] × [k × time]
```

Once you learn the bases, any new frame is a weighted sum of known sources.
Tracking becomes: "which basis is dominant?" — not "does this embedding match?"

## How It Solves Our Problem

### Current flow (broken for tracking):
```
1s audio → ECAPA → 192-dim embedding → cosine vs last frame → 0.1 (fail)
```

### Proposed flow:
```
1s audio → STFT → NMF decompose against learned bases → activation vector
                                                         ↓
                                          [0.6 alan, 0.3 music, 0.1 noise]
                                                         ↓
                                          tracker matches on activation pattern
```

The activation vector IS the "3 notes" — a compact representation of which
sources are present in this frame. Two consecutive frames of alan talking will
have similar activation patterns (high weight on alan's basis) even if the
ECAPA embeddings differ wildly.

## Algorithm

### Learning bases (enrollment / calibration):

```python
import numpy as np

def learn_basis(spectrogram, n_components=8, n_iter=200):
    """NMF: V ≈ W @ H, minimize ||V - WH||^2 with multiplicative updates."""
    V = np.abs(spectrogram) + 1e-10
    freq_bins, time_frames = V.shape
    W = np.random.rand(freq_bins, n_components) + 0.1
    H = np.random.rand(n_components, time_frames) + 0.1
    for _ in range(n_iter):
        H *= (W.T @ V) / (W.T @ W @ H + 1e-10)
        W *= (V @ H.T) / (W @ H @ H.T + 1e-10)
    return W, H
```

### Online tracking (per frame):

Given a new frame's spectrum `v` (one column of the STFT) and the learned
basis `W`, solve for activations:

```python
def decompose_frame(v, W, n_iter=50):
    """Find h such that v ≈ W @ h, h >= 0."""
    h = np.random.rand(W.shape[1]) + 0.1
    for _ in range(n_iter):
        h *= (W.T @ v) / (W.T @ W @ h + 1e-10)
    return h / (h.sum() + 1e-10)  # normalize to proportions
```

The resulting `h` is a vector of length k — the "fingerprint" of this frame
in basis-space. Track association uses cosine on `h` vectors, which will be
high for the same source across frames.

## Integration into Daredevil

### Where it lives:

```
capture → spatial → [NMF DECOMPOSE] → separation (conditional) → slots → tracker → router
                         ↓
                    activation vector per frame
                         ↓
                    tracker associates on activations (not ECAPA embeddings)
```

### The bases library:

- **Enrollment**: when you enroll "alan", also learn alan's spectral basis (2-3
  components capturing vocal tract resonances). Store alongside the ECAPA vector.
- **Online learning**: unknown sources accumulate spectral statistics. After N
  frames of stable tracking, their basis is "learned" and added to the library.
- **Fixed bases**: common sounds (broadband noise, tonal music, percussive
  transients) can be pre-loaded as universal bases.

### Separation gating:

NMF decomposition tells you *before* running ConvTasNet whether there are
actually multiple distinct sources. If only one basis is active (h = [0.9, 0.05,
0.05]), skip separation — single source. If two+ bases are active
(h = [0.4, 0.4, 0.2]), separation is warranted.

### Computational cost:

- STFT: ~1ms for 1s at 16kHz (stdlib-possible with DFT, instant with numpy)
- NMF decompose (fixed W, 50 iterations): ~2ms with numpy on a 128-bin spectrum
- **Total: ~3ms** — negligible compared to the 160ms separation or 100ms ECAPA

This is the pre-filter. Fast, cheap, runs on stdlib if needed, tells the tracker
and separator what they need to know before the expensive models run.

## The "3 Notes" Principle

Your dad's mental model was a basis library of song openings. Each entry is a
short spectral/rhythmic pattern. Matching is: project the new 3 notes into the
library and find the closest basis. He didn't need to hear the whole song because
the *attack pattern* is unique.

For Daredevil:
- 3 notes = ~300ms of audio = ~15 STFT frames
- Each source's basis captures its attack pattern (onset shape, spectral tilt,
  formant positions for speech, harmonic structure for music)
- Matching is projection: `h = solve(v ≈ W @ h)` → which basis fires hardest?
- Tracking is continuity: same basis firing across frames = same source

## Relationship to Existing Work

- **NMF for source separation** (Lee & Seung 1999, Smaragdis & Brown 2003) —
  original work on using NMF to separate audio sources
- **Online NMF** (Lefèvre et al. 2011) — streaming updates to W as new data arrives
- **Supervised NMF** (Grais & Plumbley 2017) — pre-trained bases per source class
- **NMF + spatial cues** (Ozerov & Févotte 2010) — combine spectral bases with
  DOA for multichannel separation

## Implementation Plan — status

1. ✅ `daredevil/stage1/nmf.py` — basis decomposer + `SpectralLibrary`
   (numpy NMF activations; stdlib band-group fallback, same dimension).
2. ⏳ Per-source basis fields on enrollment records — *deferred* (see note).
3. ✅ Tracker associates on the frame-stable NMF activation: `pipeline.py` feeds
   `SpectralLibrary.feature(...)` into `tracker.assign`; ECAPA stays for identity
   (SPRT, keyed per track). Tunables in `config.py` → `NMFParams`.
4. ⏳ Separation gating via NMF (today: energy + spectral-distinctness gate).
5. ⏳ Online basis learning after N confirmed frames.
6. ✅ `daredevil calibrate` exists (seeds the identity model); learning background
   *bases* there is a natural extension of step 5.

### Note — why v1 uses a fixed basis (and defers online learning)

The tracker feature must keep a **constant dimension** for a contact across its
life. If the representation switched from a 48-bin envelope to k-dim activations
mid-track (the moment an online basis is first learned), every existing track's
stored feature would silently change meaning and association would break exactly
at the switch. So v1 ships a **fixed overlapping-triangular (universal) basis** —
the "fixed/supervised bases" option named in this doc — which is deterministic,
frame-stable, and dimensionally stable. `learn_basis()` is implemented and ready;
wiring per-source online bases (steps 2 & 5) needs a dimension-stable scheme
(fixed k, append a per-source column, recompute affected tracks) and is the clear
next step — tracked honestly rather than shipped as a latent bug.

## Why This Works When ECAPA Doesn't (for tracking)

ECAPA-TDNN was trained with AAM-softmax on VoxCeleb — it learns to push
*different speakers* apart and pull *same speaker* together over utterances of
3-10 seconds. It was never optimized for frame-to-frame consistency on 1-second
windows. Its embedding space is designed for verification, not tracking.

NMF activations are a frame-level representation of *what spectral components
are present right now*. Two consecutive frames of the same source will have
nearly identical activations because the physical source didn't change. It's
measuring the physics, not the identity — and that's what the tracker needs.

Identity (ECAPA) answers WHO. Activations (NMF) answer "is this the same
ongoing sound?" Those are different questions and need different representations.
