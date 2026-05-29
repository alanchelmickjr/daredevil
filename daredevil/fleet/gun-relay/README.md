# Daredevil fleet relay (Gun.js)

The decentralized identity backbone. A Gun **peer** (not a cloud server) that lets
your trusted devices reconcile their encrypted voiceprint graph — enroll once,
recognized across the fleet (patent Claim 9), with no account and no server.

## Run

```bash
cd daredevil/fleet/gun-relay
npm install
node relay.js        # http://<host>:8765/gun   (DAREDEVIL_GUN_PORT to change)
```

Point the Python side at it:

```python
from daredevil import Pipeline
from daredevil.config import Config

cfg = Config(fleet_backend="gun", gun_peers=("http://127.0.0.1:8765/gun",))
Pipeline(config=cfg).enroll("alan", mic_seconds=3)   # local write + best-effort peer push
```

Set `DAREDEVIL_KEY` (with `pip install -e ".[crypto]"`) so records are encrypted
before they ever leave the device.

## Data model

```
daredevil/voiceprints/<name> -> { "enc": "fernet" | "none", "data": "<ciphertext>" }
```

- Only **encrypted, non-reversible embedding vectors** are stored/synced. Never raw audio.
- **Offline-first:** the local store is authoritative offline; peers merge (CRDT) when connected.
- **Status:** the local cache + best-effort push is wired; full bidirectional SEA
  sync + trust-chain key management is the next milestone (see `docs/ROADMAP.md`).
