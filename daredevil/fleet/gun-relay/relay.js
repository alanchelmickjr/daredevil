/*
 * Daredevil fleet relay — a minimal Gun.js peer (the decentralized identity backbone).
 *
 * This is NOT a cloud server: it's just another peer that helps your devices find
 * each other and reconcile the encrypted voiceprint graph. Run it on any machine
 * on your trust chain (or skip it entirely — Gun is peer-to-peer and offline-first).
 *
 *   npm install
 *   node relay.js            # listens on http://<host>:8765/gun
 *
 * Data model:
 *   daredevil/voiceprints/<name> -> { enc: "fernet"|"none", data: "<ciphertext>" }
 * Only encrypted, non-reversible embedding vectors are ever stored or synced —
 * never raw audio. Encryption/signing is handled by SEA on the JS side and Fernet
 * on the Python side (see ../crypto.py).
 */
const Gun = require("gun");
const http = require("http");

const PORT = process.env.DAREDEVIL_GUN_PORT || 8765;

const server = http.createServer((req, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });
  res.end("Daredevil Gun relay — peer endpoint at /gun\n");
});

const gun = Gun({ web: server, file: process.env.DAREDEVIL_GUN_DATA || "fleet-data" });

server.listen(PORT, () => {
  console.log(`Daredevil Gun relay (peer) on http://0.0.0.0:${PORT}/gun`);
  console.log("offline-first · peer-to-peer · encrypted (SEA) · no cloud");
});

// Optional: log voiceprint graph activity (souls only — never plaintext).
gun.get("daredevil").get("voiceprints").map().on((_data, key) => {
  console.log(`[fleet] voiceprint updated: daredevil/voiceprints/${key}`);
});
