"""Fit the fusion weights w and w2 (Eq. 7) from a host's VALIDATION-A per-query scores, and nothing else.

    python scripts/fit_fusion_weight.py results/generated/eval/dnpg_mini_seed0/valA_fsor_5way_1shot.npz [--own D]

Input: an .npz written by `scripts/evaluate.py --save-scores` on a validation-A manifest (score_<row> arrays, n_episodes x 150). The
manifest named in the sibling .json must be a validation-A manifest (split val, set A / tvA); anything else, and any path that mentions a
test manifest, is refused. Output: w = pooled std(own) / pooled std(CCP-A) and w2 = pooled std(own) / pooled std(COSF), ddof 0 over all
query scores. The paper's weights are the FROZEN values in configs/hosts/*.yaml (computed once, before any validation-B / test scoring).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccp.episodes import Episodes  # noqa: E402
from ccp.fusion import fit_weight  # noqa: E402


def fit_from_file(path: Path, own: str | None = None) -> dict:
    path = Path(path)
    if "test" in path.name.lower():
        raise ValueError(f"{path}: refusing a file named after a test manifest")
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    host = meta["host_info"]
    eps = Episodes(host["dataset"], meta["manifest"])  # raises on a test manifest (allow_test is never set here)
    if eps.split != "val" or eps.meta.get("set") not in ("A", "tvA"):
        raise ValueError(f"{meta['manifest']}: weights may only be fitted on a validation-A manifest")
    z = np.load(path)
    own = own or ("S" if "score_S" in z.files else "D")
    return {"manifest": meta["manifest"], "manifest_sha256": eps.sha256, "own": own,
            "w": fit_weight(z[f"score_{own}"], z["score_CCP-A"])["w"], "w2": fit_weight(z[f"score_{own}"], z["score_COSF"])["w"],
            "n_scores": int(z[f"score_{own}"].size)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scores", type=Path)
    ap.add_argument("--own", default=None, help="own-score row name (default S for SnaTCHer-F, D for DNPG)")
    a = ap.parse_args()
    print(json.dumps(fit_from_file(a.scores, a.own), indent=1))


if __name__ == "__main__":
    main()
