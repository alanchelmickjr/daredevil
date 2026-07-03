"""Store hygiene (gaps M4 + M11): the reference voiceprint store must be
impossible to pollute by accident — no synthetic demo writes, no silence
prints, no case-variant duplicate speakers."""
import os

import pytest

from daredevil.config import Config
from daredevil.enrollment.manager import EnrollmentManager
from daredevil.fleet.store import LocalStore


class _VecSlot:
    backend = "reference"
    def warmup(self): pass
    def run(self, audio, sr, ctx=None): return {"vector": [0.5] * 8}


def test_silence_is_unenrollable(tmp_path):
    mgr = EnrollmentManager(Config(data_dir=str(tmp_path)), _VecSlot(),
                            LocalStore(data_dir=str(tmp_path)))
    with pytest.raises(ValueError):
        mgr.enroll(audio=[0.0] * 16000, sr=16000, name="ghost", seconds=10)
    assert LocalStore(data_dir=str(tmp_path)).get("ghost") is None


def test_store_names_are_casefolded(tmp_path):
    st = LocalStore(data_dir=str(tmp_path))
    st.put("ALAN", {"name": "ALAN", "vector": [1.0]})
    assert st.get("alan") is not None, "case variant missed the record"
    st.put("Alan", {"name": "Alan", "vector": [2.0]})
    assert len(st.all()) == 1, "case variants created duplicate speakers"
    st.delete("aLaN")
    assert st.get("Alan") is None


def test_bare_serve_demo_store_is_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("DAREDEVIL_HOME", str(tmp_path))
    from daredevil.viz.server import _State
    _State(live=False)   # auto-enrolls the synthetic demo speaker
    real = tmp_path / "voiceprints"
    demo = tmp_path / "demo" / "voiceprints"
    assert not list(real.glob("*.json")), "synthetic demo wrote into the REAL store"
    assert list(demo.glob("*.json")), "demo speaker missing from the demo store"
