# SPRINT — CalHack Saturday June 21

**Goal:** Dexter walks CalHack, enrolls people by voice, remembers them,
connects everyone to Alan on LinkedIn, and uses his dual SO-101 arms +
OAK-D Pro (+ RealSense) to physically act out demos while he talks.

**Platforms:** Jetson (Dexter's brain) + MacBook (dev/fallback)
**Contract:** `GET http://127.0.0.1:8770/awareness` → JSON awareness map
**Face:** Dexter's display is his face by default. On voice command
("show me who you see" / "show the radar"), Dexter surrenders his face to
the live spatial awareness radar — a map of every tracked source around Alan.

---

## Track 1 — MINIMUM VIABLE (must ship Saturday)

Everything in Track 1 is on the critical path. Nothing else matters until
these are green.

### T1.1 — Jetson bring-up (Wed June 18)
- [ ] `pip install -e .` on Jetson, fallback mode runs clean
- [ ] `daredevil demo` completes on Jetson (pure-Python, no deps)
- [ ] `daredevil devices` reports array + slot status
- [ ] Identify USB mic model, confirm sounddevice detects it
- [ ] `pip install sounddevice soundfile` → live mic capture works
- [ ] `daredevil serve --live` stable on Jetson (no model backends yet)
- [ ] Dexter can poll `/awareness` from its own process and parse JSON

### T1.2 — Real backends on ARM (Wed–Thu)
- [ ] Try `pip install torch torchaudio speechbrain` on Jetson ARM
- [x] ONNX fallback plan (code ready, needs model export + Jetson test):
  - [x] ONNX inference path wired in `embedding.py` (reference → onnx → fallback)
  - [x] Export script at `scripts/export_ecapa_onnx.py`
  - [ ] Run export on MacBook: `python scripts/export_ecapa_onnx.py`
  - [ ] `pip install onnxruntime` on Jetson (ARM wheel exists)
  - [ ] scp `~/.daredevil/models/ecapa/ecapa_tdnn.onnx` to Jetson
- [ ] Verify `daredevil devices` shows `embedding: reference` (not fallback)
- [ ] Try `pip install librosa panns-inference` for WHAT/HOW slots
- [ ] `daredevil serve --live` with real backends — identity recognition working

### T1.3 — Enrollment + recognition (Thu June 19)
- [ ] `daredevil calibrate --name Alan --live` on Jetson with venue-like noise
- [ ] Enroll Alan → walk away 30s → come back → re-identified as "alan"
- [ ] Enroll a second person → both recognized simultaneously
- [ ] Noisy-room test: play music from phone, enroll over it — quality gate rejects bad frames
- [ ] Enrollment via HTTP: `POST /calibrate/start` from Dexter's flow works

### T1.4 — Stability soak (Thu)
- [ ] 30-min continuous `daredevil serve --live`, no crash, no memory leak
- [ ] Monitor RSS memory over time (should be flat)
- [ ] SPRT accumulators pruned when tracks die (no unbounded growth)
- [ ] Tracker coast/delete lifecycle verified (tracks don't pile up)

### T1.5 — Dexter integration contract (Thu–Fri)
- [ ] Dexter polls `GET /awareness` every 1s
- [ ] On `sources[].type == "enrolled"`: Dexter greets by name
- [ ] On `sources[].type == "unknown"` + speech: Dexter asks name, triggers enrollment
- [ ] On re-recognition after enrollment: "Hey [name], good to see you again!"
- [ ] On safety_critical event: Dexter reacts ("Whoa, what was that?")

---

## Track 2 — THE RADAR FACE (finish if T1 green by Thu night)

The spatial awareness map rendered live on Dexter's face display. Switchable
by voice command — Dexter's face is default, radar is on-demand.

### T2.1 — Live radar endpoint
- [ ] New `GET /radar` endpoint returns a rendered radar image (PNG)
  - Reuse `render_radar_hud()` from `viz/spatial_map.py`
  - Render to in-memory PNG buffer, serve with correct content-type
  - Include enrolled names, bearing arrows, priority halos, waveforms
  - Refresh on each poll (Dexter's face display polls at ~2 FPS)
- [ ] Or: serve the web HUD at `/` and Dexter opens it in a chromium webview
  - Lower effort, richer animation, but needs chromium on Jetson

### T2.2 — Face switchover
- [ ] Dexter voice command: "show me who you see" / "show the radar"
- [ ] Dexter switches face display from avatar → radar webview/image
- [ ] Voice command: "show your face" / "be yourself" → switches back
- [ ] Radar shows: each tracked source as an orbital node with:
  - Name (enrolled) or "UNKNOWN-NNN"
  - Bearing arrow (azimuth from DOA)
  - Event class icon (speech / music / baby_cry)
  - Priority halo (bigger = higher priority)
  - Waveform snippet
  - Attention status: "→ LLM" vs "ambient" (dimmed)

### T2.3 — "Imagine we see a map of everyone I can track"
- [ ] Center of radar = Dexter/Alan (the listener)
- [ ] Each person in the crowd appears as a node at their bearing
- [ ] Enrolled people glow green with their name
- [ ] Unknown speakers glow blue, pulsing when speaking
- [ ] Safety events pulse red with the event class
- [ ] Ambient/gated sources are dimmed but visible
- [ ] Live: nodes appear, move, coast, and fade as people come and go

---

## Track 3 — LINKEDIN CONNECTOR (finish if T1 green by Fri morning)

Dexter becomes Alan's networking agent. After meeting someone, Dexter
offers to connect them with Alan on LinkedIn.

### T3.1 — The flow
1. Dexter enrolls someone by voice ("Hi, I'm Dexter! What's your name?")
2. After enrollment, Dexter says: "Want me to connect you with Alan on LinkedIn?"
3. On yes: Dexter shows a QR code on his face linking to Alan's LinkedIn profile
4. Or: Dexter collects their LinkedIn handle/name and queues a connection request
5. Alan's LinkedIn profile URL is hardcoded (or in config)

### T3.2 — Implementation
- [ ] QR code approach (simplest, no API needed):
  - Generate QR code for `https://linkedin.com/in/alanhelmick` (or correct URL)
  - Render on Dexter's face display when triggered
  - "Scan this to connect with Alan!"
- [ ] Or: collect name → save to a "connections queue" file → Alan batch-sends later
- [ ] Dexter tracks who he's already offered this to (don't repeat)
- [ ] Count: "You're person #17 I've connected with Alan today!"

---

## Track 4 — PHYSICAL DEMO CHOREOGRAPHY (dual SO-101 arms + OAK-D Pro)

Dexter doesn't just talk — he acts. The arms and RealSense make the demo
physical and memorable.

### T4.1 — Arm gestures mapped to awareness events
| Awareness event | Arm gesture |
|---|---|
| New person detected (unknown speaker) | Both arms wave hello |
| Person enrolled successfully | Fist bump / handshake offer |
| Person re-recognized | Point at them + thumbs up |
| Safety-critical event (alarm, etc.) | Arms cover ears / duck |
| Showing radar on face | Arms gesture outward ("look at this!") |
| Offering LinkedIn QR | One arm presents display, other points |
| Idle / ambient music | Gentle sway / conducting gesture |
| Multiple people talking | Head swivels between bearings |

### T4.2 — RealSense integration
- [ ] RealSense depth camera confirms speaker direction (visual + audio bearing fusion)
- [ ] Dexter turns to face the speaker with highest priority
- [ ] Depth data helps Dexter gauge proximity ("you're getting closer!")
- [ ] Visual person detection cross-references audio identity
  (audio says "alan" at 45° NE, camera sees a person at 45° NE — confidence boost)

### T4.3 — Demo choreography script (the show)

**ACT 1 — "Meet Dexter" (first 30s with each person)**
1. Person approaches. Daredevil detects new audio source at their bearing.
2. Dexter turns toward them. Arms wave. "Hey! I don't think we've met. I'm Dexter."
3. "What's your name?" → Person says name.
4. Dexter enrolls: "Hold on, let me learn your voice... [3-2-1 countdown]
   ...Say something natural for a few seconds."
5. Enrollment completes: "Got it! I'll remember you now, [Name]."
6. Arms do a little celebration gesture.

**ACT 2 — "I remember you" (re-encounter)**
1. Person walks away, comes back later.
2. Daredevil SPRT matches their voice within 2-3 frames.
3. Dexter turns toward them: "Hey [Name]! Good to see you again."
4. Arms do a wave or point.
5. "Want to see something cool? Say 'show me the radar.'"

**ACT 3 — "The radar" (spatial awareness demo)**
1. Person says "show me the radar" (or Alan triggers it).
2. Dexter: "Check this out — this is everything I can hear right now."
3. Face switches to live radar HUD.
4. Arms gesture outward presenting the display.
5. Every tracked source visible: enrolled names, unknowns, music, ambient.
6. "See? You're right there at [bearing]°. And there's music coming from [bearing]°."
7. "I can track everyone in this room. All local, no cloud, just sound."
8. Person says "show your face" → face switches back.

**ACT 4 — "Connect with Alan" (LinkedIn)**
1. After the demo: "My creator Alan is here somewhere. Want me to connect you?"
2. On yes: QR code appears on face. "Scan this!"
3. Arms present the display.
4. "You're person #N that I've connected with Alan today. He's going to be busy."
5. Transition back to face mode.

**ACT 5 — "The crowd" (scaling demo for judges)**
1. Multiple enrolled people in the room.
2. Show radar: every person tracked by name, bearing, and priority.
3. Dexter: "Right now I'm tracking [N] people. [Name1] is talking at [bearing],
   [Name2] is quiet over at [bearing]."
4. Someone triggers a safety event (phone alarm).
5. Radar flashes red. Dexter: "Whoa — alarm at [bearing]°! Priority override."
6. Arms duck/cover-ears gesture.
7. "That's why this matters. Daredevil hears what cameras can't."

---

## Track 5 — POLISH & EDGE HARDENING (Fri June 20)

### T5.1 — Performance
- [ ] ONNX Runtime with TensorRT EP on Jetson (if wheels available)
- [ ] Profile: target < 500ms per awareness cycle with real backends
- [ ] Batch enrollment: 3s minimum, but accept 5-10s for higher confidence

### T5.2 — HUD polish
- [ ] Identity cards in web HUD show: name, confidence bar, bearing, waveform
- [ ] Clean transitions between face mode and radar mode
- [ ] Enrollment progress visible on face during calibration (countdown, level meter)

### T5.3 — Storage persistence
- [ ] LocalStore voiceprints survive Jetson reboot
- [ ] Verify: power cycle → enrolled people still recognized
- [ ] Calibration model persists at `~/.daredevil/calibration.json`

### T5.4 — Edge cases
- [ ] Two people talking at same bearing → NMF separates by spectral features
- [ ] Person enrolled on MacBook → recognized on Jetson (same voiceprint format)
- [ ] Enrollment in very noisy environment → quality gate rejects, asks to retry
- [ ] Rapid-fire enrollment: 5 people in 5 minutes → all recognized

---

## Track 6 — "HEAR FOR THOSE WHO CAN'T" (the mic drop)

Daredevil gives LLMs ears. But what about people who don't have ears?
Extend the perception layer bidirectionally:
- **Hearing → Deaf:** Daredevil's awareness map becomes real-time captions +
  spatial alerts (visual/haptic) for deaf users. Sound becomes sight.
- **Deaf → Hearing:** RealSense watches sign language, Dexter translates to
  speech. Sign becomes sound.

Dexter's dual SO-100 arms can *sign back*. The robot becomes a bridge.

### T6.1 — Sign language recognition (RealSense + MediaPipe)
- [ ] MediaPipe Hands → 21 hand landmarks per hand, 30 FPS from RealSense
- [ ] ASL alphabet classifier (fingerspelling) — simplest starting point
  - MediaPipe hand landmarks → normalize → classify with a small model
  - Libraries: `mediapipe`, `opencv-python`, or a pretrained ASL model
- [ ] Dexter speaks the signed letters/words aloud (TTS)
- [ ] Stretch: full ASL phrase recognition (not just fingerspelling)
  - Google's ASL dataset or sign-language-processing/datasets on HuggingFace

### T6.2 — Sound → visual bridge for deaf users
- [ ] Awareness map rendered as visual alerts on Dexter's face:
  - Directional indicator: "SPEECH from your LEFT"
  - Safety events: big red flash + vibration pattern description
  - Identity: "ALAN is speaking behind you"
- [ ] Subtitle mode: if STT is wired, real-time captions on the face display
- [ ] Haptic option: awareness map → vibration patterns (left/right buzzer for direction)

### T6.3 — Dexter signs back (the moment)
- [ ] Map simple phrases to SO-100 arm sign language gestures
  - "Hello" / "Thank you" / "My name is Dexter" / "Nice to meet you"
  - Pre-choreographed joint trajectories for each sign
- [ ] When Dexter recognizes sign language input, he signs + speaks the response
- [ ] The bridge: deaf person signs → Dexter translates to speech for hearing people,
  hearing person speaks → Daredevil transcribes → Dexter shows captions for deaf person

### T6.4 — Demo choreography: Act 6 — "The bridge"

**ACT 6 — "Everyone deserves to be heard" (the closer)**
1. Alan finds CalHack participants who know ASL (or teaches a few signs).
2. "Dexter, can you understand sign language?" → Dexter: "Let's find out."
3. Face switches to camera feed. Person signs "hello."
4. Dexter recognizes it: speaks "Hello!" + signs "hello" back with arms.
5. "Daredevil gives AI ears. But some people don't have ears. So we made it
   work both ways — sound becomes sight, and sign becomes sound."
6. Show split-screen on face: left = radar (what Dexter hears),
   right = camera (what Dexter sees being signed).
7. "On the day it matters, this isn't a demo. It's a kid in a crowd who
   can't hear the fire alarm. It's a parent who signs and needs the robot
   to translate for the doctor. Daredevil is the bridge."
8. Arms lower. Quiet beat. That's the show.

---

## Sat June 21 — DEMO DAY CHECKLIST

- [ ] Final smoke test on Jetson hardware
- [ ] All enrolled voiceprints backed up
- [ ] Battery/power sorted for mobile Dexter
- [ ] Backup plan: MacBook can run `daredevil serve --live` if Jetson fails
- [ ] Demo script rehearsed: Acts 1-5 above
- [ ] QR code tested with real phone
- [ ] Tag `v0.2.0`

---

## Next week — iPhone keyboard + Realm sync
- [ ] iOS keyboard extension with local Whisper STT
- [ ] Daredevil WHO identifies typist by voice
- [ ] Realm (MongoDB) replaces Gun for fleet identity sync
- [ ] Enroll on Dexter → recognized on phone (and vice versa)

---

## Session resume notes (for Claude)

### Blockers FIXED (merged `brave-gauss-5sG9E` → main)
- [x] **d-prime guard**: `load_calibration()` rejects degenerate models (d' < 0.5) —
  fixes "live tracking broken after bad calibration." Falls back to textbook defaults.
- [x] **Blocking mic stream**: `InputStream.read()` replaces ring buffer — no more
  dropped audio. Self-pacing, gapless, multichannel.
- [x] **Spatial channel reconciliation**: captured channels must match array geometry
  or it degrades to SINGLE (no fake spatial).

### Other completed work
- [x] ONNX backend tier in `embedding.py` (reference → onnx → fallback)
- [x] Export script: `scripts/export_ecapa_onnx.py`
- [x] Web HUD responsive at 1024x600 (7" display media queries)
- [x] `active_speaker` field in awareness map
- [x] Enrollment HTTP flow fixed (new person appears in synthetic scene post-calibration)
- [x] Test: `tests/test_enrollment_http.py`
- [x] Integration contract: `docs/DEXTER_INTEGRATION.md` (also copied to Chloe repo)

### Committed (66284dc)
- [x] All daredevil sprint work committed and tested (28 passing)
- [x] ONNX backend, active_speaker, HUD responsive, enrollment fix, integration doc
- [x] energy_floor parameter in utils.py (0.05→0.015)

### Still pending (daredevil side)
- Export ONNX model on MacBook (see Jetson recipe below)
- Test `daredevil serve --live` on real mic post-merge
- Jetson bring-up once SD card recovered
- Gun fleet code stays in repo but OFF critical path — LocalStore only for Saturday
- STT/Gemma loop is NOT functional — not critical path

### Jetson bring-up recipe (for next Claude or Alan)

**Step 0 — SD card recovery** (separate Linux box with SDK Manager):
  - 3rd Claude instance working on this now (June 17)
  - If reflash needed: JetPack 6.x for AGX Orin 64GB

**Step 1 — Export ONNX model on MacBook** (needs conda env with torch+speechbrain):
```bash
# On MacBook, in the daredevil conda env:
conda activate daredevil          # or whatever the env is called
python scripts/export_ecapa_onnx.py
# Creates: ~/.daredevil/models/ecapa/ecapa_tdnn.onnx (~20MB)
```

**Step 2 — Copy model + repo to Jetson:**
```bash
scp -r ~/.daredevil/models/ecapa/ jetson:~/.daredevil/models/ecapa/
# Or: git clone the daredevil repo on Jetson if not already there
```

**Step 3 — Install daredevil on Jetson:**
```bash
# On Jetson:
cd ~/daredevil                    # or wherever the repo lives
pip install -e ".[onnx,audio]"    # onnxruntime + numpy + sounddevice + soundfile
# onnxruntime ARM wheel exists on PyPI for aarch64
```

**Step 4 — Verify:**
```bash
daredevil devices                 # should show: embedding: onnx (not fallback)
daredevil demo                    # synthetic — should complete clean
daredevil serve --live            # real mic — open http://jetson:8770 in browser
```

**Step 5 — Enroll Alan:**
```bash
daredevil calibrate --name alan --live
# Or via HTTP from Dexter: POST /calibrate/start {"name":"alan","seconds":10}
```

**Step 6 — Dexter integration:**
```bash
# On Jetson, in a separate terminal:
curl http://127.0.0.1:8770/awareness | python -m json.tool
# Should show sources with type, event, position, identity, active_speaker
# Dexter polls this endpoint every 1s via dexter_awareness_poller.py
```

### Context for future sessions
- Dexter consumes daredevil via HTTP GET, not import — decoupled by network boundary
- Alan's conda env has all backends (torch MPS, ECAPA, librosa) on MacBook already
- Dexter (Chloe) has: dual SO-101 arms (6-DOF, Feetech STS3215), OAK-D Pro depth camera,
  ReSpeaker v2.0 USB mic (XMOS AEC), XLE head gantry (pan/tilt), HDMI face display,
  omni-wheel base, Jetson AGX Orin 64GB (JP6). Hume EVI3 for voice. MongoDB for memory.
- Dexter repo: github.com/alanchelmickjr/Chloe-a-Johhny5-robot — separate from daredevil
- IPC: pub/sub bus in chloe/ipc/bus.py — all modules communicate via Topics
- Fast dispatch: regex voice commands in chloe/voice/fast_dispatch.py
- LinkedIn connector: QR code on face display — Dexter side already wired (Day 2 done)
- Arm choreography lives in Dexter repo, daredevil just provides the trigger data
  via the awareness map fields (type, event, priority_override, attention)
- Dexter Claude finished Day 1+2 ahead of schedule (commit f5e39f3), starting Day 3
