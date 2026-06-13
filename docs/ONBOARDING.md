# Onboarding — "pick me out of the crowd"

The first run has one job: make someone *feel* what WHO-first acoustic awareness is,
in ninety seconds, on whatever machine they have. The shape is human and simple —
**you talk to it, it learns your voice and your name, then it plays a room full of
other people and picks you out of the crowd, by name.**

```bash
daredevil onboard            # anywhere — synthetic walkthrough, real recognition
daredevil onboard --live     # your mic + the crowd played out the speakers
daredevil onboard --live --name Huan --crowd 6 --windows 8 -s 12
```

## The arc

1. **Learn your voice.** A short enrollment (~8 s live; instant in synthetic). Reports
   the measured enrollment confidence and which voiceprint backend answered
   (heuristic fingerprint, or ECAPA with `[speaker]` installed).
2. **Learn its name.** Teaches the wake word (default "Hey Radar") so it wakes when
   called — the *name* channel of attention (see `docs/WAKE_WORD.md`).
3. **The reveal — find YOU in a crowd.** It puts other voices in the room and, over a
   few listen windows, lights you up against them: *"window 3/5: ✓ that's Huan
   (conf 92%) — 4 other voices gated out."* The crowd is heard and tracked, and
   deliberately kept out of the conversation — that gate is the whole point.

It ends with measured results (*"picked you out in 5/5 windows"*) and nudges to try
it in the wild: a café, the TV on, a friend talking over you, walking away
mid-sentence.

## Live vs synthetic — both honest

- **Live** (`--live`, needs `[audio]`): real enrollment, the crowd is **played out
  your speakers** (`audio/crowd.CrowdPlayer`), and your mic genuinely hears you
  *over* it. The match is earned by the live cosine SPRT.
- **Synthetic** (default, zero deps): the same flow on a **labeled SYNTHETIC** scene —
  your enrolled voice plus a generated crowd (`audio/crowd.crowd_scene_sources`).
  Recognition is still **real** (cosine SPRT on the voiceprint); only the audio is
  synthesized. Nothing is faked, and it never claims a mic/crowd is live when it
  isn't (it probes the mic first and says so).

## The crowd is generated, never recorded

`audio/crowd.py` synthesizes a murmur of *distinct synthetic speakers* with the same
speech-like DSP the rest of the demo uses. No recordings, no real voices, no real
names (rules #6/#7). It's deterministic per seed — same crowd every time, which is
also what makes the result reproducible.

## Knobs

`--name` (you), `--crowd` (how many other voices, default 4), `--windows` (reveal
length, default 5), `-s/--seconds` (enrollment length, default 8). Recognition
tuning lives in `config.IdentityModel`; the wake threshold in
`config.WakeWordParams` (see `docs/WAKE_WORD.md`).

## For a demo (e.g. to a technical friend)

`daredevil onboard --live` is the whole pitch end to end: enroll in seconds, then
get picked out of a crowd by name — open, on-device, no key. Follow it with
`daredevil serve --live` for the visual HUD, and plug a USB/RasPi mic array to add
live bearings (WHERE). What still wants the real machine: tuning the wake threshold
and the crowd volume to the room — both called out on screen.
