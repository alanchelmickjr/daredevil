# Collaboration brief

This brief is the working map for a new collaborator or agent session. It is
intentionally shorter than the full design docs and points to the files that
matter before changing code.

## Start here

1. Read `CLAUDE.md` for project rules, privacy constraints, and the stdlib-only
   core contract.
2. Read `docs/HANDOFF.md` for current live-mic reality, MacBook expectations,
   and the next implementation queue.
3. Run the smoke checks before changing behavior:
   - `python -m pytest -q`
   - `python -m daredevil.demo`
4. Keep all new heavy backends optional, lazy, and guarded with deterministic
   fallbacks. The package with no extras must still import and run.

## Current review snapshot

- Core packaging is intentionally dependency-free; optional extras are grouped by
  capability in `pyproject.toml`.
- The main pipeline is the product boundary: capture/spatial sources enter Stage
  2 slots in parallel, then Stage 3 returns a structured awareness map for agents.
- The docs are already explicit that WHO is the priority and that MacBook built-in
  mics are mono from the OS, so live MacBook testing should focus on persistent
  identity and wake/name attention rather than DOA.
- The fallback path is healthy in this environment: the test suite passes and the
  synthetic demo produces a safety-prioritized awareness map with no cloud use.

## Collaboration lanes

### Lane A — WHO reliability

Best first lane for contributors. Work on enrollment quality, SPRT tuning,
identity accumulation, and live-mic diagnostics. Keep tuning values in
`daredevil/config.py`, and add tests around decision behavior before adjusting
thresholds.

### Lane B — Backend portability

Wire ECAPA/PANNs/prosody acceleration without changing the core install contract.
Prefer ONNX Runtime paths that can later use CoreML/TensorRT execution providers.
Do not make torch, numpy, or audio libraries mandatory imports.

### Lane C — Live operations

Improve `daredevil serve --live`, `/awareness`, the web HUD, and calibration UX.
Changes here should preserve the awareness-map schema because downstream agents
consume it as the contract.

### Lane D — Safety semantics

Add honest detection paths for safety-critical events and absence/wrong-quiet
monitoring. Avoid fake success: if a detector is not implemented, return a clear
not-implemented or low-confidence result rather than a fabricated alert.

## Guardrails for parallel work

- Avoid overlapping edits to `daredevil/config.py` unless the branch owner is
  coordinating threshold changes.
- Keep docs and code in sync when changing CLI commands, awareness-map fields, or
  privacy/storage behavior.
- Do not commit hardware IP: exact mic geometry for confidential modules, BOMs,
  schematics, pin assignments, or firmware details.
- Do not add cloud calls, telemetry, or raw-audio persistence. The privacy fields
  in the output must remain truthful.
- Do not add third-party personal names to committed files.

## Ready-to-pick tasks

1. Add a diagnostic command or log mode that explains why a live speech frame was
   accepted/rejected by the speech quality gate.
2. Add tests for awareness-map schema stability, especially `timing`, `privacy`,
   `wake`, and `attention_reason` fields.
3. Implement a low-risk `/health` endpoint for the live server with backend,
   array, store, and slot readiness.
4. Add a measured live-soak script that records uptime, RSS, track count, and
   recognition decisions without storing raw audio.
5. Update stale docs that still mention older test counts or branch names after
   confirming the current expected behavior.

## Definition of ready for review

A branch is ready when it:

- passes `python -m pytest -q`,
- runs `python -m daredevil.demo`,
- documents any live/hardware checks that could not be performed,
- preserves pure-stdlib fallback behavior, and
- clearly states whether awareness-map fields or CLI behavior changed.
