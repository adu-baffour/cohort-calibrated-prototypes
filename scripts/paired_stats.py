"""Paired statistics between two rows scored on the same episodes: mean difference, 95% CI over episodes, two-sided Wilcoxon signed-rank p
(zero_method="wilcox") and the fraction of episodes with a gain.

    python scripts/paired_stats.py recorded-test            # the paper's paired test rows from the recorded per-episode files
    python scripts/paired_stats.py file <npz> <rowA> <rowB>   # any evaluate.py --save-scores output (auroc_<row> arrays)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ccp import metrics  # noqa: E402
from ccp.utils.paths import assets_root  # noqa: E402


def recorded_test() -> dict:
    zd = np.load(assets_root() / "recorded" / "test" / "final_mini_dnpg_episodes.npz")
    zs = np.load(assets_root() / "recorded" / "test" / "final_mini_snatcher_episodes.npz")
    out = {}
    for sh in (1, 5):
        k = f"{sh}shot"
        for b in ("S", "K_fix"):
            out[f"SnaTCHer-F {k} H_fix - {b}"] = metrics.paired(zs[f"{k}_T_H_fix_auroc"], zs[f"{k}_T_{b}_auroc"])
        for b in ("D", "K_fix"):
            d = np.mean([zd[f"seed{i}_{k}_T_H_fix_auroc"] - zd[f"seed{i}_{k}_T_{b}_auroc"] for i in range(3)], axis=0)  # seed-averaged per episode
            out[f"DNPG {k} H_fix - {b} (seed-averaged)"] = metrics.paired(d, np.zeros_like(d))
            for i in range(3):
                out[f"DNPG seed {i} {k} H_fix - {b}"] = metrics.paired(zd[f"seed{i}_{k}_T_H_fix_auroc"], zd[f"seed{i}_{k}_T_{b}_auroc"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("recorded-test", "file"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.mode == "recorded-test":
        res = recorded_test()
    else:
        z = np.load(a.args[0])
        res = {f"{a.args[1]} - {a.args[2]}": metrics.paired(z[f"auroc_{a.args[1]}"], z[f"auroc_{a.args[2]}"])}
    for k, v in res.items():
        print(f"{k}: {v['mean']:+.3f} ± {v['ci95']:.3f}, Wilcoxon p {v['wilcoxon_p']:.2e}, gain in {v['frac_gain'] * 100:.1f}% of episodes")
    out = REPO / "results" / "generated" / "tables" / "paired_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
