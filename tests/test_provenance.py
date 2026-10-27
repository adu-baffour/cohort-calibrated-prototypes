"""Every verbatim file (recorded results, registration documents, bundles, markers, manifests, Figure 1) has the sha256 listed in
results/PROVENANCE.md, and, when a checkout of the earlier release repository is available, equals the file committed there."""
import hashlib
import subprocess
import sys

import pytest

from conftest import OLD_AVAILABLE, OLD_REASON, OLD_REPO, ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import provenance as P  # noqa: E402

ROWS = P.listed()


def test_provenance_lists_every_verbatim_set():
    dests = [d for d, _, _ in ROWS]
    assert len(dests) == len(set(dests)) and len(dests) >= 179
    for _, d_pref in P.VERBATIM:
        assert any(d == d_pref or d.startswith(d_pref) for d in dests), d_pref
    assert P.FIG1_DEST in dests and P.DRAWIO_DEST in dests


@pytest.mark.parametrize("dest,digest,source", ROWS, ids=[r[0] for r in ROWS])
def test_verbatim_file_sha256(dest, digest, source):
    if dest in P.LOCAL_ONLY and not (ROOT / dest).exists():
        pytest.skip("Figure 1 source: contains dataset images, kept locally only (checked when present)")
    assert hashlib.sha256((ROOT / dest).read_bytes()).hexdigest() == digest


def test_no_unlisted_file_in_verbatim_folders():
    listed = {d for d, _, _ in ROWS}
    for _, d_pref in P.VERBATIM:
        base = ROOT / d_pref
        if base.is_dir():
            for f in base.rglob("*"):
                if f.is_file():
                    assert f.relative_to(ROOT).as_posix() in listed, f


@pytest.mark.skipif(not OLD_AVAILABLE, reason=OLD_REASON)
def test_verbatim_files_equal_the_release_repository():
    for dest, digest, source in ROWS:
        if source.startswith(f"{P.SOURCE_TAG}@"):
            commit, path = source[len(P.SOURCE_TAG) + 1:].split(":", 1)
            data = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=OLD_REPO, capture_output=True, check=True).stdout
            assert hashlib.sha256(data).hexdigest() == digest, dest


def test_figure1_is_the_paper_figure():
    import zipfile
    z, zsha = P.paper_zip()
    if not z.exists():
        pytest.skip("paper zip not available")
    assert hashlib.sha256(z.read_bytes()).hexdigest() == zsha
    with zipfile.ZipFile(z) as zf:
        assert zf.read(P.PAPER_FIG1) == (ROOT / P.FIG1_DEST).read_bytes()
