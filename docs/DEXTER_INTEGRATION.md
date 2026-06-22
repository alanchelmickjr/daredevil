# Daredevil ↔ Dexter Integration Contract

Both repos reference this doc. Daredevil is the perception layer (this repo).
Dexter is the robot (Chloe-a-Johhny5-robot). They are decoupled by HTTP.

## Status (updated by daredevil agent)

**Merged `brave-gauss-5sG9E` → main.** The fixes the Chloe agent flagged are now live:
- d-prime guard in `load_calibration()` — degenerate calibration models (d' < 0.5)
  are ignored with a warning, falls back to textbook defaults. This is the fix for
  "live tracking broken after bad calibration."
- Blocking `InputStream.read()` replaces the ring buffer — no more dropped audio.
- Multichannel DOA with spatial channel reconciliation.
- ONNX Runtime backend added (reference → onnx → fallback). Export script at
  `scripts/export_ecapa_onnx.py` — run on MacBook, scp `.onnx` to Jetson.
- `active_speaker` field in awareness map — shortcut for Dexter to know who's
  talking without parsing full sources array.
- Web HUD responsive at 1024x600 for 7" face display.
- Enrollment HTTP flow tested and fixed (new person appears in synthetic awareness
  map after calibration).
- STT/Gemma loop is NOT functional — code exists but never wired end-to-end.
  Not on critical path for Saturday.

## The boundary

```
Dexter (Chloe codebase)                    Daredevil (this repo)
───────────────────────                    ──────────────────────
orchestrator.py                            pipeline.py
  └─ dexter_awareness_poller.py  ─HTTP──▶  viz/server.py :8770
       polls GET /awareness                  └─ returns JSON awareness map
       polls GET /calibrate/status
       posts POST /calibrate/start
       posts POST /calibrate/phase
       posts POST /focus
```

Dexter polls. Daredevil serves. No shared memory, no imports, no coupling.
If daredevil crashes, Dexter keeps talking — it just loses acoustic awareness.
If Dexter crashes, daredevil keeps running — the awareness map is still there.

## Awareness map JSON contract

This is the schema Dexter relies on. Changes here break Dexter.

```json
{
  "timestamp": "2026-06-17T13:49:50.485519",
  "backend": "reference|fallback",
  "array": {
    "name": "macbook-3|single|respeaker-4",
    "n_mics": 3,
    "spatial": true
  },
  "wake_word": "Hey Radar",
  "focus": "alan|null",
  "top_llr": {"alan": 5.3, "emerson": -1.2},

  "routed_to_llm": ["alan", "UNKNOWN-002"],

  "sources": [
    {
      "id": "alan",
      "type": "enrolled|unknown",
      "attention": "surface|ambient",
      "event": {
        "class": "speech|baby_cry|alarm|music|...",
        "confidence": 0.95,
        "safety_critical": false
      },
      "prosody": {
        "state": "calm|distressed|excited",
        "distress": 0.12
      },
      "priority": 0.395,
      "identity": {
        "confidence": 0.632,
        "match_score": 1.0,
        "enrollment_confidence": 0.632
      },
      "position": {
        "azimuth": 0.0,
        "elevation": 0.0
      },
      "priority_override": "SAFETY_CRITICAL|DISTRESS|null",
      "track_status": "tentative|confirmed|coasting"
    }
  ],

  "timing": {
    "parallel_ms": 381.5,
    "sequential_ms": 0
  },
  "privacy": {
    "cloud_used": false,
    "raw_audio_stored": false,
    "embeddings": "non-reversible"
  },

  "active_speaker": {
    "id": "alan",
    "confidence": 0.632,
    "azimuth": 0.0,
    "priority": 0.395
  },

  "transcript": {"source": "alan", "text": "hello dexter"},
  "llm_response": "Hi Alan!"
}
```

### Fields Dexter cares about most

| Field | Dexter uses it for |
|-------|---------------------|
| `sources[].id` | Greeting by name, display, tracking |
| `sources[].type` | `enrolled` → greet; `unknown` + speech → ask name |
| `sources[].attention` | `surface` → interact; `ambient` → ignore |
| `sources[].event.class` | React to safety events, music, etc. |
| `sources[].event.safety_critical` | Emergency gesture + alert |
| `sources[].prosody.state` | Emotional response, distress escalation |
| `sources[].priority` | Who to face, who to talk to first |
| `sources[].position.azimuth` | Turn gantry toward speaker |
| `sources[].identity.confidence` | How sure we are (display confidence) |
| `sources[].track_status` | `confirmed` = real; `tentative` = maybe |
| `routed_to_llm` | Which sources crossed the attention gate |
| `focus` | Currently focused speaker for STT |
| `transcript` | What the focused speaker said (when STT wired) |
| `active_speaker` | Highest-priority enrolled speaker currently talking (shortcut) |
| `top_llr` | SPRT log-likelihood ratios per enrolled speaker |

### Enrollment endpoints

**Start calibration:**
```
POST /calibrate/start
Body: {"name": "Jamie", "seconds": 10.0, "others": false}
```

**Advance phase:**
```
POST /calibrate/phase
```

**Poll status:**
```
GET /calibrate/status
Response: {
  "active": true,
  "phase": "voice|countdown|background|world|fitting|done|idle",
  "phase_index": 1,
  "countdown": 3,
  "elapsed": 5.2,
  "duration": 10.0,
  "level": 0.7,
  "prompt": "Got you! 42 voice frames.",
  "voice_frames": 42,
  "bg_frames": 0,
  "dprime": null,
  "error_pct": null,
  "model": null,
  "quality": null
}
```

**Simplified enrollment flow for Dexter:**
1. POST `/calibrate/start` with name + seconds
2. POST `/calibrate/phase` → starts voice capture (3-2-1 countdown)
3. Poll `/calibrate/status` until `phase == "voice_done"`
4. POST `/calibrate/phase` → starts background capture
5. Poll until `phase == "background_done"`
6. POST `/calibrate/phase` → fitting
7. Poll until `phase == "done"` — read `quality` field
8. If quality == "poor", prompt retry; if "good"/"fair", done

**Set focus (for STT steering):**
```
POST /focus
Body: {"id": "alan"}     # or {"id": null} to clear
```

## What Dexter needs to build (Chloe repo)

### 1. `chloe/perception/dexter_awareness_poller.py` (new file)

```python
"""Polls daredevil /awareness and publishes to IPC bus."""

import asyncio, aiohttp, logging
from chloe.ipc.bus import MessageBus, Topic

DAREDEVIL_URL = os.environ.get("DAREDEVIL_URL", "http://127.0.0.1:8770")
POLL_INTERVAL = 1.0  # seconds

class DaredevilPoller:
    def __init__(self, bus: MessageBus):
        self.bus = bus
        self._last_ids = set()
        self._enrolled = {}      # id -> name from prior awareness maps
        self._meeting_count = 0  # LinkedIn counter

    async def run(self):
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get(f"{DAREDEVIL_URL}/awareness") as resp:
                        amap = await resp.json()
                    await self._process(amap)
                except Exception as e:
                    logging.warning(f"daredevil poll failed: {e}")
                await asyncio.sleep(POLL_INTERVAL)

    async def _process(self, amap):
        current_ids = set()
        for s in amap.get("sources", []):
            sid = s["id"]
            current_ids.add(sid)

            if s["type"] == "enrolled" and s["attention"] == "surface":
                if sid not in self._enrolled:
                    # NEW enrolled person — first recognition
                    self._enrolled[sid] = s
                    await self.bus.emit_async(
                        Topic.VISION_FACE_RECOGNIZED,
                        source="daredevil", name=sid,
                        confidence=s.get("identity", {}).get("confidence", 0),
                        azimuth=s.get("position", {}).get("azimuth"))

            if s["type"] == "unknown" and s["attention"] == "surface":
                if sid not in self._last_ids:
                    # NEW unknown speaker
                    await self.bus.emit_async(
                        Topic.VISION_PERSON_DETECTED,
                        source="daredevil", track_id=sid,
                        azimuth=s.get("position", {}).get("azimuth"))

            if s.get("event", {}).get("safety_critical"):
                await self.bus.emit_async(
                    Topic.ACTUATOR_MOTOR,
                    source="daredevil", action="safety_alert",
                    event_class=s["event"]["class"],
                    azimuth=s.get("position", {}).get("azimuth"))

        self._last_ids = current_ids

    async def enroll(self, name: str, seconds: float = 10.0):
        """Trigger daredevil enrollment from Dexter's conversation flow."""
        async with aiohttp.ClientSession() as session:
            await session.post(f"{DAREDEVIL_URL}/calibrate/start",
                json={"name": name, "seconds": seconds, "others": False})
            await session.post(f"{DAREDEVIL_URL}/calibrate/phase")
            # Poll until done
            while True:
                async with session.get(f"{DAREDEVIL_URL}/calibrate/status") as r:
                    status = await r.json()
                if status["phase"] == "done":
                    return status
                if not status["active"]:
                    return status
                await asyncio.sleep(0.5)
```

### 2. IPC Topics to add (`chloe/ipc/bus.py`)

The poller reuses existing Topics where possible:
- `VISION_PERSON_DETECTED` — unknown speaker surfaced
- `VISION_FACE_RECOGNIZED` — enrolled speaker matched (reusing face topic)
- `ACTUATOR_MOTOR` — safety alert gesture trigger

New topic needed only for radar mode:
```python
ACTUATOR_DISPLAY_MODE = "actuator.display.mode"  # {mode: "face"|"radar"|"qr"}
```

### 3. Fast dispatch additions (`chloe/voice/fast_dispatch.py`)

```python
# Daredevil voice commands
(re.compile(r"\bshow.*radar\b|\bshow.*awareness\b|\bwho.*you.*(?:hear|track|see)", re.I),
 "display_mode", {"mode": "radar"}),
(re.compile(r"\bshow.*(?:your)?.*face\b|\bbe yourself\b", re.I),
 "display_mode", {"mode": "face"}),
(re.compile(r"\bconnect.*(?:with|to)?\s*alan\b|\blinkedin\b|\bwho.*(?:made|built|created).*you", re.I),
 "display_mode", {"mode": "qr"}),
```

### 4. Gesture additions (`chloe/motion/gestures.py`)

Map awareness events to existing gestures:
```python
AWARENESS_GESTURE_MAP = {
    "new_person":      "WAVE_HELLO",
    "re_recognized":   "WAVE_HELLO",
    "enrolled":        "ARMS_UP",      # celebration
    "safety_critical": "DUCK",         # new: arms cover head
    "show_radar":      "PRESENT",      # new: arms gesture outward
    "show_qr":         "PRESENT",
}
```

### 5. Display/kiosk integration (`chloe/dashboard/display.py`)

Three modes:
- **face**: Normal animated face (default)
- **radar**: Daredevil web HUD at `http://127.0.0.1:8770/` in a webview, OR
  poll `/awareness` and render locally with CV2
- **qr**: LinkedIn QR code overlay (10s timeout, auto-dismiss)

Simplest radar approach: open `http://127.0.0.1:8770/` in the GTK WebKit
kiosk that already exists. No new rendering code needed.

## What Daredevil needs to add (this repo)

### 1. ReSpeaker mic array support

The ReSpeaker v2.0 USB has 4 mics in a circular array. Add to `mic_arrays.py`:

```python
RESPEAKER_4 = MicArray(
    name="respeaker-4",
    n_mics=4,
    spatial=True,
    coordinates=None,  # load via coordinate map or hardcode
    layout="circular",
)
```

Daredevil's `_pick_input_device()` in `capture.py` should detect the ReSpeaker
by USB product name and auto-select it + the matching array geometry.

### 2. Radar HUD improvements for face display

The web HUD at `/` already works. For the 7" touchscreen (1024x600):
- Ensure the HUD is responsive at that resolution
- Add enrolled name labels that are readable at arm's length
- Pulse animation on speaking sources
- Safety event flash (red overlay)

### 3. Enrollment quality feedback

The `/calibrate/status` response already has `quality` and `dprime`. Dexter
can use these to decide whether to retry:
- `quality == "good"` (d' >= 2.5): "Got it! I'll remember you."
- `quality == "fair"` (d' >= 1.5): "I think I got it, but it's noisy in here."
- `quality == "poor"` (d' < 1.5): "Hmm, too noisy. Let's try again somewhere quieter."

## Coordination protocol

Both Claude agents should:
1. Reference this doc for the JSON contract — don't reinvent it
2. Test against each other with `daredevil serve` (synthetic mode works fine)
3. The daredevil agent (this repo) does NOT modify the Chloe repo
4. The Chloe agent does NOT modify the daredevil repo
5. Integration testing: run both processes, verify the polling loop works

## Hardware notes

| Part | Detail | Risk |
|------|--------|------|
| Jetson | CLAUDE.md says AGX Orin 64GB (JP6). Sprint says Nano 8GB. **Confirm which.** | Orin handles everything; Nano needs ONNX. |
| Mic | ReSpeaker v2.0 USB (XMOS XVF-3000 AEC, 4-mic circular array) | Daredevil needs ReSpeaker array geometry |
| Camera | Intel RealSense (head) | Face detection is Chloe's job, not daredevil's |
| Arms | Dual SO-101, 6-DOF, Feetech STS3215, 3 serial buses | Gesture keyframes are Chloe's job |
| Display | 7" touchscreen via HDMI, GTK WebKit kiosk | Daredevil HUD must render at 1024x600 |
| Base | iMRP omni-wheel (backwards mount, inverted controls) | |
| Gantry | XLE pan/tilt (pan ±180°, tilt ±45°) | Daredevil azimuth → gantry pan for head tracking |
