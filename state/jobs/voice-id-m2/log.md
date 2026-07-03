# Job log — voice-id-m2

- 2026-07-01T19:29:34-0700 created: Daredevil knows Alan's voice on this M2 MacBook — live speaker ID, persistent across restarts
- 2026-07-01T19:29:34-0700 done-when set: A live run on this M2 identifies enrolled speaker 'alan' by name in the awareness map, in a NEW process after the enrolling one exited
- 2026-07-01T19:29:34-0700 step queued: step zero: conda run -n daredevil python -m daredevil devices — record built-in mic max_input_channels and installed backends
- 2026-07-01T19:29:34-0700 step queued: baseline: conda run -n daredevil python -m pytest -q in the daredevil repo — expect ~32 passing; record actual
- 2026-07-01T19:29:34-0700 step queued: verify real ECAPA loads on M2 (torch CPU/MPS) and computes an embedding from a wav — no cloud, no fallback fingerprint
- 2026-07-01T19:29:34-0700 step queued: enroll Alan: daredevil enroll (Alan speaks — multi-sample if available); verify voiceprint file lands in ~/.daredevil/voiceprints/
- 2026-07-01T19:29:34-0700 step queued: live WHO: daredevil listen / serve --live while Alan talks; verify the awareness map names 'alan' with SPRT confirm
- 2026-07-01T19:29:34-0700 step queued: restart persistence: kill the process, start a fresh one, verify 'alan' is still recognized (this is the DONE_WHEN observation)
- 2026-07-01T19:29:34-0700 step queued: if confirm is slow/flaky: auto-load calibration into SPRT (the known gap in calibrate.py) and re-verify
- 2026-07-01T19:29:34-0700 activated
- 2026-07-01T19:32:52-0700 popped (dispatched): step zero: conda run -n daredevil python -m daredevil devices — record built-in mic max_input_channels and installed backends
- 2026-07-01T19:33:38-0700 popped (dispatched): baseline: conda run -n daredevil python -m pytest -q in the daredevil repo — expect ~32 passing; record actual
- 2026-07-01T19:36:53-0700 popped (dispatched): verify real ECAPA loads on M2 (torch CPU/MPS) and computes an embedding from a wav — no cloud, no fallback fingerprint
- 2026-07-01T19:38:45-0700 popped (dispatched): enroll Alan: daredevil enroll (Alan speaks — multi-sample if available); verify voiceprint file lands in ~/.daredevil/voiceprints/
