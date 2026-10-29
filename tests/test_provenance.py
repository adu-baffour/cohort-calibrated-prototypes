"""Every verbatim file (recorded results, registration documents, bundles, markers, manifests) has the sha256 listed in
results/PROVENANCE.md."""
import hashlib
import sys

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import provenance as P  # noqa: E402

ROWS = P.listed()


def test_provenance_lists_every_verbatim_set():
    dests = [d for d, _, _ in ROWS]
    assert len(dests) == len(set(dests)) and len(dests) >= 178
    for _, d_pref in P.VERBATIM:
        assert any(d == d_pref or d.startswith(d_pref) for d in dests), d_pref


@pytest.mark.parametrize("dest,digest,source", ROWS, ids=[r[0] for r in ROWS])
def test_verbatim_file_sha256(dest, digest, source):
    assert hashlib.sha256((ROOT / dest).read_bytes()).hexdigest() == digest


def test_no_unlisted_file_in_verbatim_folders():
    listed = {d for d, _, _ in ROWS}
    for _, d_pref in P.VERBATIM:
        base = ROOT / d_pref
        if base.is_dir():
            for f in base.rglob("*"):
                if f.is_file():
                    assert f.relative_to(ROOT).as_posix() in listed, f
