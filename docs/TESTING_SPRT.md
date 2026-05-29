# Testing the SPRT WHO fix — pull, verify, tune

A follow-along guide for the Wald-SPRT identity + tracking change on branch
`claude/daredevil-project-review-Y7CRT`. Three levels: smoke (anywhere, 2 min),
live (MacBook + ECAPA, the real proof), then tuning if needed.

---

## Step 0 — Pull

```bash
cd daredevil
git fetch origin claude/daredevil-project-review-Y7CRT
git checkout claude/daredevil-project-review-Y7CRT
git pull origin claude/daredevil-project-review-Y7CRT
```

---

## Step 1 — Smoke test (no mic, no GPU — runs anywhere)  ⏱ ~2 min

```bash
python -m pytest -q
python -m daredevil.demo
```

**Expect:**
- [ ] `17 passed`
- [ ] Demo: `alan` shown as **enrolled** (`id_conf≈0.632`), `baby_cry` **surfaced**
      with `SAFETY_CRITICAL`, `music` **ambient** (gated out of the LLM).

If both hold, the decision math and the stitch are correct on the fallback path.

---

## Step 2 — Live test (MacBook, conda env, real ECAPA on MPS)  ← the real proof

```bash
conda activate daredevil
daredevil devices                 # confirm: embedding=reference, backend=mps

# First-run: the "get to know each other" session. Seeds the identity model
# from YOUR real voice + room (the human-tuned first copy). ~1 minute, hands-free.
daredevil calibrate --name alan --live --others
#   1/3 talk to it   2/3 stay quiet (room)   3/3 let a TV/2nd voice play
#   -> writes ~/.daredevil/calibration.json; every Pipeline loads it automatically.
#   From there it auto-tunes the background in real time (CFAR).

daredevil enroll --name alan -s 10 --live   # (calibrate already enrolls; re-enroll only to refresh)
daredevil serve --live            # HUD at http://127.0.0.1:8770
# in another shell, watch the reasoning:
#   daredevil serve --live > /tmp/dd.log 2>&1   &&   tail -f /tmp/dd.log
```

**What GOOD looks like** (talk for a few seconds):
- [ ] Within ~1–4 frames your track flips to **MATCHED alan** in the log
      (`raw=…  llr=…  score=…`), and `llr` climbs frame to frame, then stays.
- [ ] **One** track per real person — no phantom `UNKNOWN-NNN` appearing
      alongside you when you're the only one talking.
- [ ] Go quiet, then a *different* voice → a new `UNKNOWN` track, not "alan".

**What BAD would look like** (and which knob to reach for — Step 3):
- Real you keeps reading `UNKNOWN` → matcher too shy (lower the bar).
- TV / another person reads as `alan` → matcher too eager (raise the bar).
- You fragment into two tracks → tracker association too tight.

---

## Step 3 — Tune (only if Step 2 shows a problem). One knob at a time.

All knobs live in `daredevil/config.py`. The identity decision is a Wald SPRT:
per frame it adds `log p(s|you) − log p(s|background)` and fires when the running
total crosses `A = log((1−β)/α)`.

### Identity (`IdentityModel`)

| Symptom | Knob | Move | Why |
|---|---|---|---|
| Real you read as UNKNOWN | `target_mean` | **lower** (e.g. 0.65 → 0.55) | tells it your real same-voice cosine is lower than assumed |
| Real you read as UNKNOWN | `beta` (miss rate) | **raise** (0.05 → 0.10) | lowers bound A → matches sooner |
| Impostor/TV reads as you | `alpha` (false-accept) | **lower** (0.01 → 0.001) | raises bound A → needs more evidence |
| Matches feel "instant/cheaty" | `immediate_cosine` | **raise** (0.80 → 0.88) | a single frame must be very clean to short-circuit the SPRT |
| Quiet speech ignored | `quality_full_energy` | **lower** (0.02 → 0.01) | counts softer frames as full evidence |

> Rule of thumb: **shy matcher → raise `beta` or lower `target_mean`. Eager matcher → lower `alpha`.** Change one, re-run Step 2.

### Tracking (`TrackerParams`)

| Symptom | Knob | Move |
|---|---|---|
| One person → two tracks | `assoc_cosine` | **lower** (0.55 → 0.45) |
| Two people → merged into one | `assoc_cosine` | **raise** (0.55 → 0.65) |
| Track drops during a pause | `coast_s` / `delete_s` | **raise** |
| Bearings jumpy | `bearing_alpha` | **lower** (0.5 → 0.3) |

### Separation (`SeparationParams`)
Only matters with the ConvTasNet backend. If a single talker still spawns a
phantom contact, **raise `dominance_ratio`** (0.35 → 0.5) so a weak second stream
is discarded.

---

## Step 4 — Calibrate automatically (next feature, not yet built)

Instead of hand-tuning `target_mean`/`impostor_mean`, a `daredevil calibrate`
command can record a few seconds of you + a few of "anyone else / TV", measure
the actual cosine distributions, and write the fitted `IdentityModel` to a local
config. **Say the word and I'll build it** — it's the cleanest way to lock
accuracy to *your* mic and room. See `docs/HANDOFF.md` → "next work" #2.

---

## One-line mental model

> Track the contact (WHERE), then accumulate identity evidence on that track
> until it crosses the SPRT bound (WHO) — exactly how passive sonar calls a
> target. The awareness-map JSON schema did not change.
