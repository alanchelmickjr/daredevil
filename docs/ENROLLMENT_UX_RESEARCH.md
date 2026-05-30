# Enrollment & Calibration UX — Prior Art Research

How production speaker recognition systems prime identity matching.
Researched 2026-05-29 to inform Daredevil's onboarding flow.

---

## Audio Duration (Industry Consensus)

| System | Minimum | Sweet Spot | Notes |
|--------|---------|------------|-------|
| Microsoft Azure (text-independent) | 20s net speech | 30-120s across multiple calls | Strips silence before counting |
| Microsoft Azure (text-dependent) | 3 passphrase repetitions | 3 phrases | Each ~2-4s |
| Apple Hey Siri | 5 utterances | ~15-20s total speech | Short session |
| Picovoice Eagle | Progressive % | Until `enroll()` returns 100% | Multiple distinct utterances |
| SpeechBrain ECAPA | 1 utterance works | 5-30s | Longer is better |
| Resemblyzer | 5-30s | 10-15s | Single or multiple clips averaged |
| Kaldi/VoxSRC | Multi-session | Conversation-length per session | Different days/conditions |

**Bottom line:** 10-20s of net speech (silence excluded) is the floor for text-independent. 3-5 distinct utterances is the universal UX pattern.

---

## Diversity Requirements

- **Apple:** 5 different prompt phrases — not the same sentence repeated. Captures pitch/rate variation.
- **Microsoft text-independent:** Explicitly requires "diverse" audio — different sentences, captured in the deployment environment. Same device/room as production use.
- **VoxSRC/NIST SRE protocols:** Multi-session enrollment (different days) dramatically improves robustness.

**Bottom line:** Phrase diversity matters more than raw duration. Different prosody (question, statement, casual) covers the embedding space better than repeating one phrase 5x.

---

## Calibration (Score Distribution Measurement)

- **WeSpeaker AS-Norm:** Cosine score mean/std from a top-N cohort. Normalize: `(score - mu) / sigma` from both sides.
- **SpeechBrain PLDA:** Models between-class and within-class variance. S-norm gives ~0.10% EER improvement.
- **VoxCeleb challenge winners:** Quality-aware calibration — incorporates SNR and duration so scores are consistent across conditions.
- **Universal pattern:** Measure real target + real impostor distributions in-situ, not from published stats.

---

## Quality Gating (What Systems Reject)

| Check | Who Does It | Implementation |
|-------|-------------|----------------|
| Silence / too quiet | All | RMS below threshold -> reject frame |
| Too short (< min duration) | Azure, Eagle | Return "insufficient audio" error |
| SNR too low | Azure, Eagle | "quiet environment, no other speakers" |
| Clipping / too loud | Best practice | Reject frames above saturation |
| Multiple speakers | Eagle | "Single speaker only in the audio" |
| Progress feedback | Eagle, Azure | Return % complete or remaining seconds |

---

## UX Patterns That Work

1. **Progressive feedback (Eagle):** Each `enroll()` call returns a percentage. User talks until 100%. System decides when it has enough — not a magic number of utterances.

2. **Conversational prompts (Apple/Daredevil):** Give something natural to say. 2-3 different prompts ensure prosodic diversity without feeling like a test.

3. **Level check before enrollment (universal):** Show audio level before committing. Immediate feedback: too quiet / good / too loud.

4. **Separation feedback (d-prime):** After enrollment, tell how well-separated their voice is from background. Honest, measured number.

5. **Background measurement as explicit step:** Voice -> room -> others. You need both target and impostor distributions measured in-situ.

---

## Re-enrollment / Sharpening

- **Microsoft:** No auto-sharpening. Requires explicit re-enrollment API call.
- **Research consensus:** Adaptive enrollment is powerful but poisonable. Guard with: high confidence threshold + quality gate + margin past decision bound.
- **Multi-sample mean update (SpeechBrain, Kaldi):** Each new sample nudges the mean embedding (Welford / first-order statistics).

---

## What Daredevil Already Has vs. What's Missing

### Already implemented (verified in code)
- 3-phase calibration session: voice -> room -> others (`calibrate.py`)
- Cosine distribution measurement + Gaussian fit (`Calibrator.fit()`)
- d-prime separation metric + human-readable error rate
- Level feedback during capture (`_level_note()`)
- Multi-sample Welford mean for re-enrollment
- Guarded online voiceprint adaptation (`_maybe_refine()`)
- CFAR background adaptation (live impostor drift tracking)
- VAD energy gate on enrollment frames

### Missing (blocking real matching)
- **No calibration.json exists** — SPRT uses VoxCeleb textbook defaults instead of measured distributions
- **Single enrollment sample** — only 1 capture of ~10s. Need 3-5 diverse utterances.
- **No diversity detection** — no check that enrollment chunks cover different prosody
- **No progressive enrollment %** — user doesn't know when "enough" is enough
- **No SNR gate** — noisy frames accepted without penalty
- **No "readiness" threshold** — no minimum d-prime to declare enrollment complete

### Design decisions to make
- d-prime threshold for "ready" (d' >= 3.0 is < 0.1% error; d' < 1.5 means "talk more")
- Whether to require re-calibration on environment change (CFAR drift detection)
- Whether the HUD should surface enrollment status / prompt for more samples
