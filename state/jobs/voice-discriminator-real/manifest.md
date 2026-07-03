# voice-discriminator-real — context-proof checklist

> A fresh session (post-compaction, restart, migration) resumes from THIS file + the gap doc.
> Nothing here depends on conversation memory. Read these, in order, before touching code:
> 1. `docs/GAP_VOICE_DISCRIMINATOR.md` — the verified gap analysis (50 gaps, file:line cited,
>    adversarially checked 2026-07-01/02). Steps below reference its tables and path.
> 2. `state/jobs/voice-discriminator-real/meta` + `steps.txt` — the live cursor.
> 3. `chronicle/` today + yesterday — what already happened.
> Env: conda env `daredevil` (`/opt/homebrew/Caskroom/miniforge/base/envs/daredevil/bin/python`).
> Tests: `python -m pytest -q` (37 passing baseline). HUD: `python -m daredevil serve --live`.
> Store: `~/.daredevil/` (voiceprints: Alan + Emerson; quarantine/ holds synthetic impostors).

## Non-negotiables (from the working contract + Alan)

- Rule 1: no guessing — every claim traces to file:line or a log artifact.
- Rule 2: the AI runs all commands (mic sessions: narrate plainly, run at sensible moments).
- Chronicle at every externalization: step popped/closed, doc written, decision made, fix landed.
- Verify each step's CHECK by observation before `job.sh pop`; a step without its check observed
  is NOT done. Auto-test hook runs pytest on every source edit — keep it green.
- HUD honesty rules (daredevil CLAUDE.md 7/9): no fabricated numbers, no disguised stubs.

## Checklist (mirrors steps.txt — [ ] flips only when the CHECK is observed)

- [ ] 1. B3 capture loop — ≥55 windows/60s ×10min headless; /awareness <50ms
- [ ] 2. B1 calibration fit — held-out H1, voices-in-H0, floor 0.18±0.11; impostor_mean ≥0.15;
        60s foreign voice → zero MATCHED
- [ ] 3. M1 hot-apply — Wald bounds change in-process after HUD fit; test pinned
- [ ] 4. B2 revocable hold — TV takeover demotes ≤10s; irrevocable-hold test pins REPLACED
- [ ] 5. M3 speech-class casefold — live active_speaker == owner while speaking
- [ ] 6. M4+M11 store hygiene — bare serve writes nothing; silence unenrollable; names casefold
- [ ] 7. M5+M6 H0 learning + cohort — bg mean rises on guest voices; cohort at 2 enrolled; no flap
- [ ] 8. M2+M12+m8 calibration robustness — mid-phase mic failure surfaces, map resumes ≤60s
- [ ] 9. M7+M8+M9+m7+m11 HUD truthfulness — dimmed named card on pause; zero mono bearings;
        no misattributed captions; audio-age shown
- [ ] 10. M10+M13+m2+m3+m9+m10 robustness — fallback watermark; first observed STT→LLM output;
        key-mismatch surfaced; anti-alias resample
- [ ] 11. M14+M15+m4+m6 overlap — owner named / TV UNKNOWN under scripted overlap

## DONE_WHEN (extrinsic, all 8 from the doc's Definition of done)

Continuous listening · owner ≤4s acquisition 9/10 · revocable hold ≤10s · 10-min impostor
rejection zero-MATCHED · honest calibration governing the RUNNING matcher · restart survival
with key-mismatch surfacing · truthful HUD · store hygiene. Each observed live, evidence in
the chronicle, then `job.sh close voice-discriminator-real`.
