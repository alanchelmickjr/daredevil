# Wake word — attention by name (the second channel)

> "It should be natural, like a human. I build these for robots — and when we
> can't get someone's attention by the *sound* of our voice, we use their name."

That sentence is the whole design. Daredevil grabs (and gives) attention the way
people do, on two channels:

1. **By the sound of the voice** — urgency, distress, loudness. This is the
   prosody slot (HOW) feeding the router's `DISTRESS` escalation, and
   safety-critical events (`SAFETY_CRITICAL`). The room's *sound* earns attention.
2. **By name** — when the sound alone isn't enough, you say the name. The wake
   word *is* the system's name. Hearing it is an explicit bid for attention, so it
   escalates too (`ADDRESSED`), just under genuine distress.

Wake word and identification go **hand in hand**: the **name** grabs attention,
**WHO** says who is calling. The awareness map carries both, so the LLM/robot can
answer the right person by name — and, symmetrically, address a human by name when
its own voice won't carry.

## How it works

`daredevil/stage2/wake.py` — `WakeWordDetector`, the project's
graceful-degradation pattern:

- **template backend (always works, stdlib only).** *Query-by-example keyword
  spotting.* You say the name a few times; we keep a short **band-energy spectral
  contour** of it (a per-frame, loudness-invariant spectral-shape sequence) and
  detect by **sub-sequence Dynamic Time Warping** of the live window against that
  contour. DTW handles the natural time-warp of speech; the sub-sequence variant
  lets the name occur anywhere in the window. This is a real, classic KWS method —
  not a stub — and it is inherently **speaker-personal**: the system wakes for the
  people who taught it its name.
- **reference backend (optional).** [openWakeWord](https://github.com/dscripka/openWakeWord)
  (Apache-2.0) — fully on-device ONNX, **no AccessKey, no cloud, no metering**.
  Set `WakeWordParams.oww_model` to a model name and `pip install openwakeword`;
  `warmup()` upgrades the backend automatically. (We deliberately do *not* use
  Picovoice Porcupine here: it requires a cloud-validated AccessKey and meters
  usage — antithetical to `allow_cloud = False`.)

**Privacy.** Only the lossy band-energy contour is persisted
(`~/.daredevil/wakewords/*.json`) — never raw audio. It cannot be played back as
speech.

## Use it

```bash
daredevil wake --name-phrase            # (alias) teach the name
daredevil wake --live -s 2              # say "Hey Radar" a couple of times
daredevil wake --live -s 2              #   repeat for robustness (multi-sample)
daredevil serve --live                  # HUD: say the name → your circle pulses,
                                        #   your name lands, focus turns to you
```

Programmatically:

```python
pipe.enroll_wake(phrase="Hey Radar", source="live")   # teach the name
amap = pipe.listen(source="live")
amap["wake"]      # {name, enrolled, backend, detected: <source id or None>, score, phrase}
# surfaced sources gain: "addressed": true, "attention_reason": "name"
```

## What it adds to the awareness map (schema is additive — nothing removed)

- top level: `amap["wake"] = {name, enrolled, backend, detected, score, phrase}`.
  `detected` is the source id that called (or `null`), and (if `grab_focus`) that
  id becomes `config.focus`.
- per source (when named): `"addressed": true`, `"wake": {phrase, score}`, and
  `"attention_reason"` — one of `safety | voice | name | owner-speaking | salient`.
  That reason is the natural cue for how to respond, and how the attention was
  earned in the first place.

## Tunables (`config.WakeWordParams`)

`enabled`, `analysis_sr` (8 kHz — phonetic detail lives < 4 kHz), `frame_ms`/`hop_ms`,
`n_bands`, `threshold` (DTW similarity to fire, default 0.72), `min_phrase_s`,
`grab_focus`, `oww_model`, `oww_threshold`.

## Honesty / what still needs the real Mac

The detector is verified deterministically (`tests/test_wake.py`): it matches a
phrase against a time-warped, noisy copy of itself, discriminates a reversed
contour, rejects broadband noise, persists across processes, and wires into the
pipeline. **The 0.72 threshold wants tuning with a real voice in the real room** —
say the name, watch `amap["wake"]["score"]`, and set the threshold a little below
your reliable scores. That tuning is the one step that needs the laptop and a mic.
