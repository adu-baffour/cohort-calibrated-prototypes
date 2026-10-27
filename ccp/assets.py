"""Recorded per-episode and validation-score files distributed as release assets (too large for the repository), sha256-pinned.

`path(key)` returns the file under $CCP_ASSETS after checking its sha256; `load(key)` opens it with numpy. These files were written by the
recorded runs and are read only to re-render and check recorded summaries (tables, figures, reports); no score is recomputed from images.
The test files hold per-episode metrics of the single recorded test passes (no images, features or labels).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ccp.utils.paths import assets_root, check_sha256

FILES = {   # key -> (path under $CCP_ASSETS, sha256)
    # per-episode metrics of the recorded test passes
    "test/baselines": ("recorded/test/baselines_test_episodes.npz", "bfe12a90bcc1ef0bca5084d75364b1e197a734a94a7f6a5bbf7544232be16a4a"),
    "test/variants": ("recorded/test/variants_test_episodes.npz", "86f9a2150a290dbfa27832434d955a61d5b2c0210a220549a557a75202bd5fcd"),
    "test/final_round": ("recorded/test/final_round_test_episodes.npz", "c4017d760ea5db8460241950aff3f4f9004d98057cc24f62e03316ff4cf0daf9"),
    "test/foundation_round": ("recorded/test/foundation_round_test_episodes.npz",
                              "06a04e25e024705c20ba8b9e1d247d327366aa0640b2a3bd7a018ba91425fc49"),
    "test/dnpg_mini": ("recorded/test/final_mini_dnpg_episodes.npz", "c42ea5f8a947c01eba06b6d1ec6fb7dda2af336f353876150fa89019aa6ccb0e"),
    "test/snatcher_mini": ("recorded/test/final_mini_snatcher_episodes.npz", "a5323f41e155a7d08e658a8733879a1be65859fad567b9e6ded859d7bba49514"),
    # validation scores
    "validation/figure_cache": ("recorded/validation/fig_cache_frozen_feat_mini_valB.npz",
                                "5a6a342e9ee21491168a6bbea274e41be89d59b744b45fc0b4caa5448ac828aa"),
    "validation/dnpg_mini/seed0_1shot": ("recorded/validation/dnpg_raw/seed0_1shot.npz", "f70cbcf6113e2968aec92781cde9aac030deab916e02e857125b6cd58d608d4e"),
    "validation/dnpg_mini/seed0_5shot": ("recorded/validation/dnpg_raw/seed0_5shot.npz", "b05f43d62e8c3133a7743827360ae2025513f5f80a24c3807bbced47ffd8f747"),
    "validation/dnpg_mini/seed1_1shot": ("recorded/validation/dnpg_raw/seed1_1shot.npz", "198e0df4297f06d79a6614fcc43e3d6be3d3fd2e1347d82277289d52ccfe9f91"),
    "validation/dnpg_mini/seed1_5shot": ("recorded/validation/dnpg_raw/seed1_5shot.npz", "650440662c2d0a9e91d1044b64e13df13ee3f405dfe1b2b5b3a756aa6adfa656"),
    "validation/dnpg_mini/seed2_1shot": ("recorded/validation/dnpg_raw/seed2_1shot.npz", "20acbb007c90a605be70de95a4a789ba7cfffa7a79a80854a68ddb521666e724"),
    "validation/dnpg_mini/seed2_5shot": ("recorded/validation/dnpg_raw/seed2_5shot.npz", "fbe8c27e7560b6f369fcb8d3145afcd4f6c314a2b7ebd13cb299c14357db5830"),
}


def path(key: str) -> Path:
    rel, sha = FILES[key]
    p = assets_root() / rel
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: download the release assets (docs/checkpoints.md)")
    check_sha256(p, sha, key)
    return p


def load(key: str):
    return np.load(path(key))
