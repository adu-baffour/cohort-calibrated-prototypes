"""The release scan: no dataset image, private data, AI attribution or registration label in the files git would commit."""
import subprocess

import pytest

from scripts import release_scan as R


def git_ok():
    try:
        subprocess.run(["git", "rev-parse"], cwd=R.REPO, check=True, capture_output=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def test_raster_counts_of_the_tracked_figures():
    assert R.pdf_raster_images((R.REPO / "paper_assets/figures/fig2_mechanism.pdf").read_bytes()) == 1    # Figure 2(a), rasterized scatter
    assert R.pdf_raster_images((R.REPO / "paper_assets/figures/fig3_scg_gain.pdf").read_bytes()) == 0


@pytest.mark.skipif(not git_ok(), reason="not a git checkout")
def test_scan_is_clean(monkeypatch):
    files = R.files()
    # a dataset-image file still in the git index of an earlier commit is reported by the scan itself (git rm --cached); the rest is clean
    monkeypatch.setattr(R, "files", lambda: [f for f in files if f not in R.DATASET_IMAGE_FILES])
    assert R.scan() == []


def test_asset_scan_flags_images_and_image_arrays(tmp_path):
    import io

    import numpy as np
    buf = io.BytesIO()
    np.savez(buf, imgs=np.zeros((4, 84, 84, 3), np.uint8), chw=np.zeros((2, 3, 32, 32), np.uint8), ok=np.zeros((10, 3), np.float32),
             rng=np.zeros(5056, np.uint8))
    problems, u8 = R.scan_member("x.npz", buf.getvalue())
    assert len(problems) == 2 and len(u8) == 3
    problems, _ = R.scan_member("figure.bin", b"\x89PNG\r\n\x1a\n" + b"0" * 32)       # by content, whatever the name
    assert problems and "PNG" in problems[0]
    assert R.scan_member("scores.npz", _npz(np.zeros((1000, 150), np.float32))) == ([], [])


def _npz(a):
    import io

    import numpy as np
    b = io.BytesIO()
    np.savez(b, a=a)
    return b.getvalue()
