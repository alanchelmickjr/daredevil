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
    return mgr.enroll(audio=[0.1] * 16000, sr=16000, name=name, seconds=10)


# --- SPRT identity decisions ------------------------------------------------

def test_realistic_frames_match_in_two(tmp_path):
    """Contract since the frame-LLR clip: NO single frame can decide alone (blip/
    replay robustness) — two squarely-same-speaker frames decide. At the live
    0.5s hop that is ~1s to acquisition."""
    rng = random.Random(1)
    base = _rand_unit(rng)
    mgr, slot, store = _manager(tmp_path)
    _enroll(mgr, slot, base)
    enrolled = store.get("alan")["vector"]
    m = mgr.match(_at_cosine(0.55, enrolled, rng), energy=0.05)
    assert not mgr.is_match(m), "a single frame decided alone (clip broken)"
    m = mgr.match(_at_cosine(0.55, enrolled, rng), energy=0.05)
    assert mgr.is_match(m) and m["name"] == "alan"


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


def test_two_frame_match_rate_is_high(tmp_path):
    """Fresh manager per trial; two same-speaker frames must acquire ≥90% of the
    time (was one-shot before the anti-blip frame-LLR clip)."""
    rng = random.Random(4)
    base = _rand_unit(rng)
    mgr, slot, store = _manager(tmp_path)
    _enroll(mgr, slot, base)
    enrolled = store.get("alan")["vector"]
    hits, N = 0, 30
    for _ in range(N):
        om = EnrollmentManager(mgr.config, slot, store)   # fresh accumulators
        om.match(_at_cosine(0.55, enrolled, rng), energy=0.05)
        hits += om.is_match(om.match(_at_cosine(0.55, enrolled, rng), energy=0.05))
    assert hits >= 0.9 * N


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


# --- regression: decided candidate must win best-selection (2026-07-01) ------

def test_immediate_accept_beats_earlier_nonmatch_in_cohort(tmp_path):
    """With two enrolled speakers, an immediate (high-cosine) accept for the
    SECOND speaker must not lose the best-candidate slot to the first speaker's
    non-match on an llr tie. Regression: live 'alan' scored cos=0.806 (decided)
    but the map surfaced Emerson's cos=0.156 with identity null."""
    rng = random.Random(7)
    a, b = _rand_unit(rng), _rand_unit(rng)
    mgr, slot, store = _manager(tmp_path)
    _enroll(mgr, slot, a, name="Emerson")
    _enroll(mgr, slot, b, name="alan")
    enrolled = store.get("alan")["vector"]
    mgr.match(_at_cosine(0.85, enrolled, rng), energy=0.05, key="t1")      # arms
    m = mgr.match(_at_cosine(0.85, enrolled, rng), energy=0.05, key="t1")  # engages
    assert mgr.is_match(m), "decided candidate was dropped from best-selection"
    assert m["name"] == "alan"


def test_immediate_accept_needs_two_consecutive_strong_frames(tmp_path):
    """One loud blip (or one frame of a replayed recording) must NOT latch an
    identity: the first strong frame arms; only a consecutive strong frame
    engages the hold (gaps B2/M16). A weak frame in between disarms."""
    rng = random.Random(8)
    base = _rand_unit(rng)
    mgr, slot, store = _manager(tmp_path)
    _enroll(mgr, slot, base)
    enrolled = store.get("alan")["vector"]
    m1 = mgr.match(_at_cosine(0.85, enrolled, rng), energy=0.05, key="t1")
    assert not mgr.is_match(m1), "single strong frame latched an identity"
    mgr.match(_at_cosine(0.20, enrolled, rng), energy=0.05, key="t1")      # disarms
    m2 = mgr.match(_at_cosine(0.85, enrolled, rng), energy=0.05, key="t1")
    assert not mgr.is_match(m2), "non-consecutive strong frames latched"
    m3 = mgr.match(_at_cosine(0.85, enrolled, rng), energy=0.05, key="t1")
    assert mgr.is_match(m3), "two consecutive strong frames failed to latch"
    m4 = mgr.match(_at_cosine(0.30, enrolled, rng), energy=0.05, key="t1")
    assert mgr.is_match(m4) and m4["name"] == "alan", "held identity flickered off"


def test_held_identity_demotes_on_sustained_contrary_evidence(tmp_path):
    """Revocable hold (gap B2): a decided track taken over by a DIFFERENT voice
    must demote within a bounded number of contrary frames — never held forever.
    Live incident: raw=-0.150 frames held as 'Alan' for ~8 minutes."""
    rng = random.Random(9)
    base = _rand_unit(rng)
    mgr, slot, store = _manager(tmp_path)
    _enroll(mgr, slot, base)
    enrolled = store.get("alan")["vector"]
    for _ in range(2):
        mgr.match(_at_cosine(0.85, enrolled, rng), energy=0.05, key="t1")
    assert mgr.is_match(mgr.match(_at_cosine(0.60, enrolled, rng), energy=0.05, key="t1"))
    demoted_at = None
    for f in range(1, 11):
        m = mgr.match(_at_cosine(0.05, enrolled, rng), energy=0.05, key="t1")
        if not mgr.is_match(m):
            demoted_at = f
            break
    assert demoted_at is not None, "takeover voice never demoted the held identity"


def test_apply_model_hot_swaps_bounds_and_resets(tmp_path):
    """Gap M1: apply_model must swap the running Gaussians, recompute the Wald
    bounds, reseed the CFAR background, and invalidate accumulated decisions."""
    from daredevil.config import IdentityModel
    rng = random.Random(10)
    base = _rand_unit(rng)
    mgr, slot, store = _manager(tmp_path)
    _enroll(mgr, slot, base)
    enrolled = store.get("alan")["vector"]
    for _ in range(2):
        mgr.match(_at_cosine(0.85, enrolled, rng), energy=0.05, key="t1")
    assert mgr._llr, "expected accumulated evidence before apply"
    old_A = mgr._A
    new = IdentityModel(target_mean=0.5, target_std=0.1,
                        impostor_mean=0.2, impostor_std=0.1,
                        alpha=0.001, beta=0.05)
    mgr.apply_model(new)
    assert mgr.idm is new and mgr.config.identity is new
    assert mgr._A != old_A and abs(mgr._A - 6.856) < 0.01   # ln(0.95/0.001)
    assert mgr._bg_mean == 0.2 and not mgr._llr, "accumulators survived hot-apply"
