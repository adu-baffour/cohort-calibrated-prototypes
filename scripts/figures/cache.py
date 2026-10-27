"""Validation score cache for Figures 2 and 4 (frozen FEAT, mini-ImageNet valB; release asset, key validation/figure_cache in ccp/assets.py).

Contents, per shot S in {1, 5} (1,000 valB episodes x 150 queries; queries 0-74 known, 75-149 unknown):
  x_{S}shot       raw max cosine to the mirror-view prototypes (COSF)
  m_{S}shot       the query's cohort statistic m_K(q), K = 1000 (Eq. 3)
  cosf_{S}shot    COSF unknown score (-max cosine)
  ccpa_{S}shot    CCP-A unknown score (Eqs. 4-5)
  nearest_{S}shot argmax_k cosine (class index 0-4 in the episode)
and for every valB image (rows 0..9,599 = global 38,400..47,999): nn_idx (5 nearest bank images, global train indices) and nn_cos.

Built from the fp16 feature cache the paper figures used; scripts/figures/build_cache.py rebuilds it with the release pipeline (GPU,
validation only).
"""
from __future__ import annotations

from ccp import assets

KEY = "validation/figure_cache"


def load():
    return assets.load(KEY)
