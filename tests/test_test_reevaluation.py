"""The reader-facing test re-evaluation path of scripts/evaluate.py, on SYNTHETIC manifests only (a temporary folder; no real test
manifest, image, feature or score is opened): a test manifest needs --allow-test, a warning is printed, and the output can only go to
results/generated/test_reevaluation/."""
import importlib.util
import sys

import numpy as np
import pytest

from conftest import ROOT, make_episodes

from ccp.episodes import SplitAccessError, write_manifest


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("evaluate_mod", ROOT / "scripts" / "evaluate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture()
def synthetic_root(tmp_path):
    lab = np.repeat(np.arange(12), 30)
    lines = []
    for split, name in (("test", "test_fsor_5way_1shot"), ("val", "valA_fsor_5way_1shot")):
        arrs = make_episodes(lab, 1, 3, seed=1)
        h = write_manifest(tmp_path / "synthetic" / f"{name}.json", arrs,
                           {"dataset": "synthetic", "name": name, "n_episodes": 3, "shot": 1, "seed": 1, "split": split, "set": "A"})
        lines.append(f"{h}  synthetic/{name}.json")
    (tmp_path / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def test_test_manifest_needs_the_flag(ev, synthetic_root, capsys):
    with pytest.raises(SplitAccessError):
        ev.open_manifests("synthetic", ["test_fsor_5way_1shot"], allow_test=False, root=synthetic_root)
    assert ev.open_manifests("synthetic", ["valA_fsor_5way_1shot"], allow_test=False, root=synthetic_root)[0].split == "val"
    assert "WARNING" not in capsys.readouterr().err
    eps = ev.open_manifests("synthetic", ["test_fsor_5way_1shot"], allow_test=True, root=synthetic_root)
    assert eps[0].split == "test"
    err = capsys.readouterr().err
    assert "new evaluation" in err and "recorded single passes" in err


def test_output_only_under_generated(ev, tmp_path):
    gen = tmp_path / "results" / "generated"
    assert ev.resolve_out_dir(None, "h", has_test=True, generated=gen) == (gen / "test_reevaluation" / "h").resolve()
    assert ev.resolve_out_dir(None, "h", has_test=False, generated=gen) == (gen / "eval" / "h").resolve()
    for bad in (tmp_path / "results" / "final_round", tmp_path / "results" / "recorded" / "test", ROOT / "results" / "foundation_round"):
        with pytest.raises(SystemExit, match="outside"):
            ev.resolve_out_dir(bad, "h", has_test=True, generated=gen)
    with pytest.raises(SystemExit, match="test_reevaluation"):
        ev.resolve_out_dir(gen / "eval" / "h", "h", has_test=True, generated=gen)


def test_cli_documents_the_warning():
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--help"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "NEW evaluation" in r.stdout and "test_reevaluation" in r.stdout
