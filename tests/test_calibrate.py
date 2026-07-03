"""Calibration session + real-time auto-calibration (CFAR) tests — pure stdlib.

Covers the two layers:
  * the human onboarding session writes a sane IdentityModel that a Pipeline reloads;
  * the matcher's background model adapts online (CFAR) without masking the target.
"""
import math
import random

from daredevil.config import Config, IdentityModel, load_calibration
from daredevil.enrollment.manager import EnrollmentManager
from daredevil.fleet.store import LocalStore

DIM = 128


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _rand_unit(rng):
    return _unit([rng.gauss(0, 1) for _ in range(DIM)])


def _at_cosine(target, base, rng):
    g = _rand_unit(rng)
    proj = sum(g[i] * base[i] for i in range(DIM))
    g = _unit([g[i] - proj * base[i] for i in range(DIM)])
    s = math.sqrt(max(0.0, 1.0 - target * target))
    return [target * base[i] + s * g[i] for i in range(DIM)]


# --- Layer 2: the onboarding session ----------------------------------------

def test_calibrate_session_writes_loadable_model(tmp_path):
    from daredevil.calibrate import session
    out = session(name="alan", live=False, seconds=4.0, config=Config(data_dir=str(tmp_path)))
    # A model file was written and round-trips into an IdentityModel.
    cal = load_calibration(tmp_path)
    assert isinstance(cal, IdentityModel)
    assert out["stats"]["synthetic"] is True
    # Voice should sit above the background (the whole point of separating them).
    assert cal.target_mean > cal.impostor_mean
    assert out["dprime"] >= 0.0


def test_pipeline_loads_calibration(tmp_path):
    from daredevil.calibrate import session
    from daredevil.pipeline import Pipeline
    session(name="alan", live=False, seconds=4.0, config=Config(data_dir=str(tmp_path)))
    pipe = Pipeline(config=Config(data_dir=str(tmp_path)))
    saved = load_calibration(tmp_path)
    # The pipeline adopts the human-seeded model rather than the cold defaults.
    assert pipe.config.identity.target_mean == saved.target_mean
    assert pipe.config.identity.impostor_mean == saved.impostor_mean


# --- Layer 1: CFAR background adaptation -------------------------------------

class _VectorSlot:
    backend = "reference"

    def __init__(self):
        self.q = []

    def warmup(self):
        pass

    def run(self, audio, sr, ctx=None):
        return {"vector": self.q.pop(0) if self.q else [0.0] * DIM}


def _manager(tmp, **idm):
    cfg = Config(data_dir=str(tmp))
    if idm:
        cfg.identity = IdentityModel(**idm)
    slot = _VectorSlot()
    store = LocalStore(data_dir=str(tmp))
    mgr = EnrollmentManager(cfg, slot, store)
    slot.q = [list(_seed_base())]
    mgr.enroll(audio=[0.1] * 16000, sr=16000, name="alan", seconds=10)
    return mgr, store


_BASE = None


def _seed_base():
    global _BASE
    if _BASE is None:
        _BASE = _rand_unit(random.Random(99))
    return _BASE


def test_background_adapts_toward_observed_ambient(tmp_path):
    rng = random.Random(11)
    mgr, store = _manager(tmp_path)
    enrolled = store.get("alan")["vector"]
    start = mgr._bg_mean
    # Feed many low-cosine ambient frames; the noise floor should track them down.
    for _ in range(50):
        mgr.match(_at_cosine(0.05, enrolled, rng), energy=0.05, key="amb")
    assert mgr._bg_mean < start          # CFAR pulled the background toward the room
    assert mgr._bg_mean < 0.18


def test_cfar_guard_excludes_the_target(tmp_path):
    rng = random.Random(12)
    mgr, store = _manager(tmp_path)
    enrolled = store.get("alan")["vector"]
    start_mean = mgr._bg_mean
    # Strong same-speaker frames must NOT be absorbed into the noise estimate
    # (guard cells). Otherwise the target would mask itself.
    for _ in range(20):
        mgr.match(_at_cosine(0.75, enrolled, rng), energy=0.05, key="me")
    assert abs(mgr._bg_mean - start_mean) < 0.05   # background essentially unmoved
