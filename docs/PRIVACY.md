# Privacy

Daredevil exists so an LLM can know **who** is speaking **without surveilling
anyone**. Privacy is a design constraint, not a feature.

## Guarantees

1. **On-device inference. Cloud never.** `Config.allow_cloud` defaults to `False`
   and there is no code path that transmits audio or embeddings off the device.
   (The prior cloud prosody API was removed precisely for this.)
2. **Raw audio is never stored or transmitted.** We persist only computed
   embedding vectors and the structured awareness map.
3. **Voiceprints are non-reversible.** A 192-dim ECAPA embedding is a
   mathematical representation; you cannot reconstruct speech from it.
4. **Encrypted at rest.** With `DAREDEVIL_KEY` set and `cryptography` installed,
   voiceprints are encrypted (Fernet / AES) on disk; decryption happens only in
   memory. Target per the hardware spec: AES-256.
5. **Truthful output.** Every awareness map carries
   `"privacy": {"cloud_used": false, "raw_audio_stored": false, "embeddings": "non-reversible"}`.

## Fleet-shared identity over Gun (no cloud)

The identity layer is built on **[Gun](https://gun.eco/)** — a decentralized,
offline-first, peer-to-peer graph database. This realizes the patent's
"fleet-shared encrypted identity across a trust chain": enroll once, and your
*own* trusted devices recognize you — synced directly between peers, with **no
server and no account**.

- **What syncs:** only encrypted, non-reversible embedding records. Never audio.
- **Encryption:** **SEA** (Gun's built-in ECDSA/AES) on the JS side; Fernet on the
  Python side. Same principle — only ciphertext leaves a device.
- **Wire format:** the graph node `daredevil/voiceprints/<name>` holds the same
  encrypted blob the local store writes.
- **Offline-first:** the local store is always authoritative offline; peers
  reconcile when connected (CRDT merge). Enrollment never blocks on the network.

Run a peer with the relay in [`../daredevil/fleet/gun-relay/`](../daredevil/fleet/gun-relay/).
Default deployments use the local store and need nothing running.

## Threat-model notes (current scaffold)

- Without `DAREDEVIL_KEY`, local records are base64 JSON (clearly marked
  `"enc": "none"`) — convenient for a single-device demo, **not** for fleet sync.
  Set a key before enabling Gun.
- The fallback "fingerprint" embedding is not a secure biometric; it exists only
  so the demo runs without ML deps. Real deployments use ECAPA.
- Trust-chain key management (who may join the fleet) is the next milestone.
