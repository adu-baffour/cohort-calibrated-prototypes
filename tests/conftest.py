"""Shared test setup: the repository root is importable first (ahead of any installed copy of an older `ccp`), and the markers
`gpu` / `data` are skipped unless CCP_TEST_GPU=1 / CCP_TEST_DATA=1."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(config, items):
    skips = {"gpu": (os.environ.get("CCP_TEST_GPU") != "1", "set CCP_TEST_GPU=1"),
             "data": (os.environ.get("CCP_TEST_DATA") != "1", "set CCP_TEST_DATA=1")}
    for item in items:
        for mark, (skip, why) in skips.items():
            if skip and mark in item.keywords:
                item.add_marker(pytest.mark.skip(reason=why))


# ---------------------------------------------------------------------------------------------------------------- synthetic data
@pytest.fixture(scope="session")
def synth():
    """Base split: 12 classes x 100 images (>= the registered K = 1000); target split: 10 classes x 30 images; d = 16."""
    import numpy as np
    import torch
    g = torch.Generator().manual_seed(0)
    d = 16
    centers = torch.randn(22, d, generator=g) * 2.0

    def draw(cls, n):
        x = centers[cls][None] + torch.randn(n, d, generator=g)
        return x.abs() + 0.05  # GAP of ReLU-like maps is non-negative
    base = torch.cat([draw(c, 100) for c in range(12)])
    base_f = base + 0.1 * torch.randn(base.shape, generator=g)
    base_lab = np.repeat(np.arange(12), 100)
    tgt = torch.cat([draw(12 + c, 30) for c in range(10)])
    tgt_f = tgt + 0.1 * torch.randn(tgt.shape, generator=g)
    tgt_lab = np.repeat(np.arange(10), 30)
    return {"base": base, "base_f": base_f, "base_lab": base_lab, "tgt": tgt, "tgt_f": tgt_f, "tgt_lab": tgt_lab, "d": d}


def make_episodes(tgt_lab, shot: int, n: int, seed: int = 0):
    """Synthetic FSOR episodes on target rows (5 known classes: shot support + 15 queries; 5 unknown classes: 15 queries)."""
    import numpy as np

    from ccp.episodes import make_fsor_episodes
    c2i = {int(c): np.where(tgt_lab == c)[0] for c in np.unique(tgt_lab)}
    arrs, _ = make_fsor_episodes(c2i, shot, n, seed)
    return arrs


@pytest.fixture(scope="session")
def bank(synth):
    import numpy as np

    from ccp.cohort import build_bank
    return build_bank(synth["base"], synth["base_f"], synth["base_lab"], np.arange(len(synth["base_lab"])), {"host": "synthetic"}, "fp32",
                      with_noflip_stats=True)
