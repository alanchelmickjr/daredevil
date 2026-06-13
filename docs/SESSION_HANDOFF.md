# Session handoff — live HUD test on the MacBook (2026-06-13)

Pick up here on the **local Claude Code CLI on the MacBook**. Everything below is
already pushed to branch **`claude/intelligent-gates-mpyqe1`**, so a `git pull` gets it all.

## The immediate goal

Get the **live HUD** running on the Mac and watch it pick the user out against the
room (TV / AirPods Max session):

```bash
cd ~/<path>/daredevil
git fetch origin && git checkout claude/intelligent-gates-mpyqe1 && git pull
pip install -e ".[audio]"          # adds sounddevice — REQUIRED for a real mic
python -m daredevil serve --live   # → http://127.0.0.1:8770
```

Then open **http://127.0.0.1:8770**. Tail/host it for the user and say when it's ready.

> Why this must run on the Mac (not the prior cloud session): the web runner is a
> Linux VM with no `/dev/snd` and no `sounddevice`, and its `127.0.0.1` isn't the
> user's `127.0.0.1`. Mic + page have to live on the same machine as the user. The
> local CLI fixes that — same assistant, real hardware.

## Hardware gotchas (AirPods Max)

- **Mic:** `audio/capture.py::_pick_input_device` deliberately skips AirPods as input
  (in headset/SCO mode they return zeros at low sample rate) and falls through to the
  **built-in MacBook mic array**. That's correct — keep AirPods as **output**.
- **The "crowd" needs to be out loud in the room.** If TV/crowd audio is playing *into
  the AirPods*, the built-in mic hears silence and it'll just track the user (no crowd
  effect). For the cocktail-party demo: either put the **TV on the room speakers**
  (real other-voices → click the TV source in the HUD for captions, watch the user stay
  surfaced while the TV is gated), or for `onboard --live` set system **output to the
  laptop speakers** so the synthetic crowd is in the room, not in the ears.

## What's on this branch (this session's work)

1. **Wake word — attention by name** (commit `28977f2`)
   - `daredevil wake --live` learns "Hey Radar"; `amap["wake"]` carries score/addressed.
   - One tuning knob: `config.WakeWordParams.threshold` (default 0.72). If it's too eager
     or too deaf, watch `amap["wake"]["score"]` live and set threshold just under the
     scores you reliably hit.
2. **Crowd / babble generator** (commit `d5a12bf`) — `daredevil/audio/crowd.py`
   - `babble()`, `crowd_scene_sources()`, `CrowdPlayer` (loops synthetic babble out the
     speakers for the live reveal; honest no-op with no audio device). Generated DSP,
     never a recording — labeled SYNTHETIC.
3. **Onboarding — "pick me out of the crowd"** (commits `c875178`, `109794b`)
   - `python -m daredevil onboard` — anywhere, zero deps, real recognition on a labeled
     SYNTHETIC scene. `--live` does it for real (mic + crowd out the speakers).
   - Flags: `--name N`, `--live`, `--crowd K` (default 4), `--windows W` (default 5),
     `-s SEC` (enroll length, default 8).
   - Verified here: synthetic onboard picks "huan" out of a 4-voice crowd **5/5 windows
     at 93%**, crowd heard-but-gated.
   - Docs: `docs/ONBOARDING.md`. README front door updated ("Try it in 60 seconds").

## How to test each thing locally

```bash
python -m pytest -q                 # 38 passing on stdlib alone
python -m daredevil onboard         # synthetic arc end-to-end (no mic needed)
python -m daredevil onboard --live  # real: enroll, then crowd out the speakers
python -m daredevil enroll --name <you> --live -s 10
python -m daredevil wake --live -s 2
python -m daredevil serve --live    # the HUD
```

## State / open items

- Branch is **4 commits ahead of `main`** (wake + 3 onboarding). **NOT merged to main** —
  the user wants to test live first. Don't merge without an explicit ask.
- Optional upgrades: `pip install -e ".[speaker]"` swaps the heuristic voiceprint for
  real ECAPA (better WHO); `[events]`, `[prosody]`, `[spatial]` upgrade the other slots;
  README's macOS path adds whisper-cpp/llama-cpp for live captions in the HUD.
- Tuning that needs the real room: wake threshold and crowd volume (both surfaced on
  screen during `onboard`/`wake`).
