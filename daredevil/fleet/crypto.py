"""Encryption-at-rest for voiceprints.

Target (patent Claim 8): AES-256 at rest, decrypt only in runtime memory. We use
`cryptography` (Fernet, AES-128-CBC + HMAC) when available + a key is provided via
$DAREDEVIL_KEY. With no key we store base64 JSON ("enc":"none") and mark it clearly
as unencrypted — local-only, never the default for fleet sync.

On the JS/Gun side, the equivalent is SEA (Gun's built-in ECDSA/AES). Only
computed embedding vectors are ever stored or synced — never raw audio.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from typing import Optional


def have_crypto() -> bool:
    try:
        import cryptography.fernet  # noqa: F401
        return True
    except Exception:
        return False


def derive_key(passphrase: str) -> bytes:
    """Derive a urlsafe-base64 32-byte Fernet key from a passphrase."""
    digest = hashlib.sha256(passphrase.encode()).digest()
    return base64.urlsafe_b64encode(digest)


def key_from_env() -> Optional[bytes]:
    pw = os.environ.get("DAREDEVIL_KEY")
    return derive_key(pw) if pw else None


def encrypt(obj: dict, key: Optional[bytes]) -> dict:
    data = json.dumps(obj).encode()
    if key and have_crypto():
        from cryptography.fernet import Fernet
        token = Fernet(key).encrypt(data).decode()
        return {"enc": "fernet", "data": token}
    return {"enc": "none", "data": base64.b64encode(data).decode()}


def decrypt(blob: dict, key: Optional[bytes]) -> dict:
    enc = blob.get("enc", "none")
    if enc == "fernet":
        if not (key and have_crypto()):
            raise ValueError("Encrypted record requires DAREDEVIL_KEY + cryptography")
        from cryptography.fernet import Fernet
        data = Fernet(key).decrypt(blob["data"].encode())
        return json.loads(data)
    return json.loads(base64.b64decode(blob["data"]))
