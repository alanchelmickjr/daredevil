# Attention Gate — Feed-Forward Filtering Design

## The Problem

The pipeline currently flows one direction:

```
capture → spatial → separation → slots → tracker → router/gate → LLM
```

The gate makes a decision at the END (surface vs ambient), but that decision
is never fed back. Every frame, every source gets full processing regardless
of whether it mattered last time. That's wasteful and, worse, it means the
system can't learn what to ignore.

## The Insight

The gate should feed FORWARD to the beginning of the next loop:

```
capture → [FILTER] → spatial → separation → slots → tracker → router/gate
              ↑                                                      │
              └──────────── priority state from last frame ──────────┘
```

**If a source was ambient/low-priority last frame, don't waste cycles
separating and classifying it again — unless something changes.**

The filter is not a hard block. It's an attention budget. Sources that were:
- **surface** last frame: full processing, every slot, every frame
- **ambient** last frame: lightweight check only (energy + event class stability)
- **new/changed**: full processing (the gate doesn't block a surprise)

## What Triggers Full Processing

A previously-ambient source gets promoted to full processing when:
1. **Energy spike** — sudden loud event (baby cry, alarm, glass break)
2. **Event class change** — was "music", now "speech" (someone started talking)
3. **Enrolled speaker detected** — even partial match, escalate
4. **Safety-critical class** — always full processing, no gating
5. **Timeout** — haven't fully processed in N frames, do a full check

## What "Lightweight Check" Means

For ambient sources on subsequent frames:
- Compute energy (< 0.1ms)
- Check if energy changed significantly (> 2x or < 0.5x previous)
- Optionally: quick event classifier (not full PANNs — just speech/not-speech)
- If stable: reuse last frame's classification, don't run separation or embedding
- If changed: promote to full processing

## The State Machine

Each tracked source maintains:
```python
{
    "id": "UNKNOWN-003",
    "processing_level": "full" | "light" | "skip",
    "last_full_frame": 42,      # frame number of last full analysis
    "last_event_class": "music",
    "last_energy": 0.023,
    "priority_history": [0.15, 0.16, 0.14],  # recent priorities
    "attention": "ambient",
}
```

Transitions:
```
        ┌─ energy spike / class change / safety / timeout ──┐
        │                                                    ↓
    [SKIP] ←── stable 3+ frames ── [LIGHT] ←── ambient ── [FULL]
                                                    ↑          │
                                                    └── surface┘
```

## Latency Budget Impact

Current: separation (~160ms) + slots (~100ms) = 260ms per source per frame.
With 2 separated sources: 520ms per frame.

With feed-forward filtering:
- 1 surface source (full): 260ms
- 1 ambient source (light): 1ms
- **Total: 261ms** — nearly 2x faster with 2 sources

With 5 sources in a crowd:
- 1 surface (you): 260ms
- 4 ambient (TV, fan, fridge, street): 4ms
- **Total: 264ms** vs 1300ms without filtering

## Separation Budget

The separator itself is expensive (160ms). The filter should decide BEFORE
separation whether it's needed:
- If the overall energy profile hasn't changed, and no new spatial direction
  appeared, skip separation entirely — reuse last frame's source split
- If energy changed in a specific frequency band, re-separate
- If a new DOA appears (multi-mic), always re-separate

## Connection to the LLM Loop

When Gemma (or whatever local LLM) is wired in, the feed-forward filter
becomes the attention mechanism for the entire agent:

```
LLM context window:
  - Only sources that passed the gate
  - Historical context from tracked sources
  - Priority-ordered (safety first, enrolled speaker second, etc.)

LLM output (future):
  - "Focus on the speaker at 45 degrees" → adjusts gate thresholds
  - "Ignore the music" → demotes that source's processing level
  - "Alert me if baby cries" → safety class always at full processing
```

The LLM can steer the attention gate, which steers the filter, which steers
what gets processed. The system becomes a closed loop: perception → attention →
action → perception, with the LLM as the decision layer.

## Implementation Priority

1. **Source state machine** — track processing_level per source in the tracker
2. **Energy-based pre-filter** — before separation, check if re-separation needed
3. **Slot skip for ambient** — don't run ECAPA/PANNs on sources that haven't changed
4. **Timeout re-check** — ambient sources get full processing every N frames
5. **LLM steering** — future: LLM adjusts thresholds and focus

## Papers/References

- SORT (Simple Online Realtime Tracking, Bewley et al. 2016) — predict-detect-associate pattern
- pyannote 3.0 (Plaquet & Bredin 2023) — ACTIVE/TENTATIVE/DORMANT state machine for speakers
- VBx (Landini 2022) — HMM transition probabilities encode "speaker probably didn't change"
- Attention Is All You Need (Vaswani 2017) — the gate IS a form of attention (hard attention over sources)
- Personal VAD (Ding et al. 2020, Google) — speaker-conditioned filtering, only process what matters
