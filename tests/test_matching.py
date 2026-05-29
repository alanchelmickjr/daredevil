"""SPRT identity matching + track-manager tests (pure stdlib, deterministic).

These pin the *decision math* independently of any model: we feed the matcher
embedding vectors at controlled cosine similarities (the same code path ECAPA
drives) and assert the Wald SPRT accepts true speakers quickly, rejects
impostors, and that the tracker keeps one contact as one track while separating
two distinct contacts.

`_VectorSlot` is a test double standing in for ECAPA so the matcher can be tested
in isolation — it is NOT part of the product.
"""
import math
import random

import pytest

from daredevil.config import Config
from daredevil.enrollment.manager import EnrollmentManager
from daredevil.fleet.store import LocalStore
from daredevil.stage3.tracker import UnknownTracker

DIM = 128


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _rand_unit(rng):
    return _unit([rng.gauss(0, 1) for _ in range(DIM)])


def _at_cosine(target, base, rng):
    """A unit vector whose cosine to `base` is approximately `target`."""
    g = _rand_unit(rng)
    proj = sum(g[i] * base[i] for i in range(DIM))
    g = _unit([g[i] - proj * base[i] for i in range(DIM)])   # orthogonal to base
    s = math.sqrt(max(0.0, 1.0 - target * target))
    return [target * base[i] + s * g[i] for i in range(DIM)]


class _VectorSlot:
    backend = "reference"

    def __init__(self):
        self.q = []

    def warmup(self):
        pass

    def run(self, audio, sr, ctx=None):
        return {"vector": self.q.pop(0) if self.q else [0.0] * DIM}


def _manager(tmp):
    cfg = Config(data_dir=str(tmp))
    slot = _VectorSlot()
    store = LocalStore(data_dir=str(tmp))
    return EnrollmentManager(cfg, slot, store), slot, store


def _enroll(mgr, slot, vec, name="alan"):
    slot.q = [list(vec)]
    return mgr.enroll(audio=[0.0] * 16000, sr=16000, name=name, seconds=10)


# --- SPRT identity decisions ------------------------------------------------

def test_single_realistic_frame_matches_one_shot(tmp_path):
    rng = random.Random(1)
    base = _rand_unit(rng)
    mgr, slot, store = _manager(tmp_path)
    _enroll(mgr, slot, base)
    enrolled = store.get("alan")["vector"]
    m = mgr.match(_at_cosine(0.55, enrolled, rng), energy=0.05)
    assert mgr.is_match(m) and m["name"] == "alan"   # 0.55 is squarely same-speaker


def test_impostor_never_matches(tmp_path):
    rng = random.Random(2)
    base = _rand_unit(rng)
    mgr, slot, store = _manager(tmp_path)
    _enroll(mgr, slot, base)
    enrolled = store.get("alan")["vector"]
    false = 0
    for _ in range(20):
        false += mgr.is_match(mgr.match(_at_cosine(0.15, enrolled, rng), energy=0.05))
    assert false == 0


def test_one_shot_match_rate_is_high(tmp_path):
    rng = random.Random(4)
    base = _rand_unit(rng)
    mgr, slot, store = _manager(tmp_path)
    _enroll(mgr, slot, base)
    enrolled = store.get("alan")["vector"]
    hits, N = 0, 30
    for _ in range(N):
        om = EnrollmentManager(mgr.config, slot, store)   # fresh = stateless one-shot
        hits += om.is_match(om.match(_at_cosine(0.55, enrolled, rng), energy=0.05))
    assert hits >= 0.9 * N   # the old matcher scored 0/30 here


def test_weak_frames_accumulate_then_lock(tmp_path):
    rng = random.Random(3)
    base = _rand_unit(rng)
    mgr, slot, store = _manager(tmp_path)
    _enroll(mgr, slot, base)
    enrolled = store.get("alan")["vector"]
    # A borderline frame shouldn't decide on frame 1, but evidence should build.
    assert not mgr.is_match(mgr.match(_at_cosine(0.45, enrolled, rng), energy=0.05))
    locked = None
    for f in range(2, 9):
        if mgr.is_match(mgr.match(_at_cosine(0.45, enrolled, rng), energy=0.05)):
            locked = f
            break
    assert locked is not None   # name-that-tune: locks within a few frames


def test_per_track_evidence_is_independent(tmp_path):
    rng = random.Random(8)
    base = _rand_unit(rng)
    mgr, slot, store = _manager(tmp_path)
    _enroll(mgr, slot, base)
    enrolled = store.get("alan")["vector"]
    # An impostor on its own track must not benefit from the real speaker's
    # accumulated evidence on a different track.
    for _ in range(5):
        mgr.match(_at_cosine(0.6, enrolled, rng), energy=0.05, key="UNKNOWN-001")
    m_imp = mgr.match(_at_cosine(0.12, enrolled, rng), energy=0.05, key="UNKNOWN-002")
    assert not mgr.is_match(m_imp)


# --- track manager ----------------------------------------------------------

def test_tracker_one_contact_stays_one_track():
    rng = random.Random(5)
    v = _rand_unit(rng)
    t = UnknownTracker()
    ids = {t.assign([x + rng.gauss(0, 0.02) for x in v]) for _ in range(6)}
    assert len(ids) == 1


def test_tracker_two_contacts_two_tracks():
    rng = random.Random(6)
    a, b = _rand_unit(rng), _rand_unit(rng)
    t = UnknownTracker()
    assert t.assign(list(a)) != t.assign(list(b))


def test_tracker_confirms_after_m_hits():
    rng = random.Random(7)
    v = _rand_unit(rng)
    t = UnknownTracker()
    sid = t.assign(list(v))
    assert t.status_of(sid) == "tentative"
    t.assign(list(v))                       # second hit -> M-of-N confirm
    assert t.status_of(sid) == "confirmed"
