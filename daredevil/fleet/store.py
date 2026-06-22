"""Identity store — local-first, with a Gun.js fleet backbone.

`LocalStore` (default) persists encrypted voiceprint records as JSON under the
data dir — this is also exactly how Gun behaves offline. `GunStore` extends it
with peer sync to a Gun relay (the Node sidecar in fleet/gun-relay/). Enroll
once, recognized across the trust chain (patent Claim 9), with only non-reversible
embedding vectors shared — never raw audio.
"""
from __future__ import annotations

import abc
import json
import logging
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..config import Config, default_data_dir
from . import crypto

log = logging.getLogger("daredevil.store")

_SAFE = re.compile(r"[^a-zA-Z0-9._-]")


class IdentityStore(abc.ABC):
    @abc.abstractmethod
    def put(self, name: str, record: dict) -> None: ...

    @abc.abstractmethod
    def get(self, name: str) -> Optional[dict]: ...

    @abc.abstractmethod
    def all(self) -> List[dict]: ...

    @abc.abstractmethod
    def delete(self, name: str) -> None: ...

    def names(self) -> List[str]:
        return [r["name"] for r in self.all()]


class LocalStore(IdentityStore):
    def __init__(self, data_dir: Optional[Path] = None, key: Optional[bytes] = None):
        base = Path(data_dir) if data_dir else default_data_dir()
        self.dir = base / "voiceprints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.key = key

    def _path(self, name: str) -> Path:
        return self.dir / f"{_SAFE.sub('_', name)}.json"

    def put(self, name: str, record: dict) -> None:
        blob = crypto.encrypt(record, self.key)
        self._path(name).write_text(json.dumps(blob))

    def get(self, name: str) -> Optional[dict]:
        p = self._path(name)
        if not p.exists():
            return None
        return crypto.decrypt(json.loads(p.read_text()), self.key)

    def all(self) -> List[dict]:
        out = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                out.append(crypto.decrypt(json.loads(p.read_text()), self.key))
            except Exception as e:
                log.warning("skipping corrupt voiceprint %s: %s", p, e)
                continue
        return out

    def delete(self, name: str) -> None:
        p = self._path(name)
        if p.exists():
            p.unlink()


class GunStore(LocalStore):
    """Offline-first local cache + Gun peer sync.

    The local cache (LocalStore) is authoritative offline; sync to/from Gun peers
    happens opportunistically. Full P2P sync is the next milestone — the wire
    format is: graph `daredevil/voiceprints/<name>` -> the same encrypted blob the
    LocalStore writes. SEA handles encryption + signing on the JS side.
    """

    def __init__(self, data_dir: Optional[Path] = None, key: Optional[bytes] = None,
                 peers: tuple = (), timeout: float = 2.0):
        super().__init__(data_dir, key)
        self.peers = tuple(peers)
        self.timeout = timeout

    def put(self, name: str, record: dict) -> None:
        super().put(name, record)
        self._sync_push(name, crypto.encrypt(record, self.key))

    def _sync_push(self, name: str, blob: dict) -> None:
        # TODO: push to Gun peers (WS/HTTP) — see fleet/gun-relay/relay.js.
        # Intentionally a no-op when no peers configured; LocalStore stays correct.
        if not self.peers:
            return
        try:  # best-effort; never block enrollment on the network
            import urllib.request
            data = json.dumps({"soul": f"daredevil/voiceprints/{name}", "blob": blob}).encode()
            for peer in self.peers:
                req = urllib.request.Request(peer, data=data,
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=self.timeout)
        except Exception as e:
            log.debug("gun sync failed (offline-first, local write OK): %s", e)


def make_store(config: Config) -> IdentityStore:
    key = crypto.key_from_env()
    data_dir = config.resolved_data_dir()
    if config.fleet_backend == "gun":
        return GunStore(data_dir=data_dir, key=key, peers=config.gun_peers,
                        timeout=config.fleet_timeout_s)
    return LocalStore(data_dir=data_dir, key=key)
