"""Paired differences that the recorded round files do not store, from the sha256-pinned recorded per-episode test metrics only.

    python scripts/derive_paired.py            # -> results/generated/derived/paired_derived.json
    python scripts/derive_paired.py --check    # exit 1 unless a fresh computation equals results/derived/paired_derived.json byte for byte

results/derived/paired_derived.json is the copy that Tables 4-6 and docs/paper_numbers.md read (verbatim from release 1.5.0, listed in
results/PROVENANCE.md); this script shows how it was computed and checks it. No test manifest, image, feature or score is read: the inputs
are the per-episode metric files written by the single test passes of rounds (vii) and (viii) (ccp/assets.py). No new statistical test:
each entry is the mean and 95% CI (1.96 std / sqrt(n), ddof 0, ccp.metrics.ci95) of a per-episode difference and the fraction of episodes
with a positive difference, the same arithmetic as the recorded `paired` entries of results/final_round/test.json.

  cub             round (vii), mini-ImageNet hosts on CUB with the mini-ImageNet bank: per host and shot, REF - X for every other row X,
                  where REF is CCP-A for frozen FEAT and H_fix for SnaTCHer-F and DNPG (DNPG: per-episode mean over seeds 0-2). AUROC, accuracy.
  foundation      round (viii), per configuration (dataset|host|shot): X - COSF and CCP-A - X for every recorded row X (AUROC).
  resnet_vs_cosf  round (vii) re-evaluation of the round (i)-(ii) frozen hosts: CCP-A - COSF per host and shot with ccp.metrics.paired
                  (mean, 95% CI, two-sided Wilcoxon signed-rank p, fraction improved / tied); FRN: per-episode difference averaged over
                  seeds 0-2. AUROC, accuracy, OSCR, FPR95.
Row keys inside the file are the recorded keys (for example H[B4-CL2N]); ccp.recorded.load translates them to the paper's names.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ccp import assets, metrics, recorded  # noqa: E402

RECORDED = "results/derived/paired_derived.json"
OUT = ROOT / "results" / "generated" / "derived" / "paired_derived.json"
# the source fields as recorded (release-asset names at the time the file was written)
SOURCES = {"final_round": "release asset recorded/test/final_round_test_episodes.npz sha256 "
                          "c4017d760ea5db8460241950aff3f4f9004d98057cc24f62e03316ff4cf0daf9",
           "foundation": "release asset recorded/test/foundation_round_test_episodes.npz sha256 "
                         "06a04e25e024705c20ba8b9e1d247d327366aa0640b2a3bd7a018ba91425fc49"}
CUB_HOSTS = {"frozen FEAT mini": (["frozen_feat_mini"], "CCP-A"), "SnaTCHer-F mini": (["snatcher_f_mini"], "H_fix"),
             "DNPG mini": ([f"dnpg_mini_seed{s}" for s in range(3)], "H_fix")}
FM_HOSTS = ("clip_b16", "dinov2_s14", "dinov2_b14", "dinov2_l14")
FM_DATASETS = ("mini", "tiered", "cifarfs", "cub")
C_HOSTS = {"frozen FEAT mini": ["frozen_feat_mini"], "TANE mini": ["tane_pre_mini"], "FRN mini": [f"frn_mini_seed{s}" for s in range(3)],
           "frozen FEAT tiered": ["frozen_feat_tiered"], "TANE tiered": ["tane_pre_tiered"]}


def stat(d) -> dict:
    d = np.asarray(d, dtype=np.float64)
    return {"mean": float(d.mean()), "ci95": float(metrics.ci95(d)), "frac_gain": float((d > 0).mean()), "n_episodes": int(d.size)}


def rows_of(z, prefix) -> list[str]:
    return sorted({k[len(prefix):].rsplit("__", 1)[0] for k in z.files if k.startswith(prefix)})


def compute() -> dict:
    zf = assets.load("test/final_round")
    cub = {}
    for name, (hosts, ref) in CUB_HOSTS.items():
        for sh in (1, 5):
            pre = [f"B__{h}__{sh}shot__" for h in hosts]
            rows = rows_of(zf, pre[0])
            for p in pre[1:]:
                assert rows_of(zf, p) == rows
            cell = {}
            for m in ("auroc", "acc"):
                get = lambda r: np.mean([zf[f"{p}{r}__{m}"].astype(np.float64) for p in pre], axis=0)  # noqa: E731
                cell[m] = {x: stat(get(ref) - get(x)) for x in rows if x != ref}
            cub[f"{name}|{sh}"] = {"ref": ref, "seeds": len(hosts), **cell}
    zm = assets.load("test/foundation_round")
    fm = {}
    for b in FM_HOSTS:
        for ds in FM_DATASETS:
            for sh in (1, 5):
                pre = f"{ds}_{b}__T__{sh}shot__"
                rows = rows_of(zm, pre)
                g = lambda r: zm[f"{pre}{r}__auroc"].astype(np.float64)  # noqa: E731
                fm[f"{ds}|{b}|{sh}"] = {"minus_COSF": {x: stat(g(x) - g("COSF")) for x in rows if x != "COSF"},
                                        "CCPA_minus": {x: stat(g("CCP-A") - g(x)) for x in rows if x != "CCP-A"}}
    rc = {}
    for name, hosts in C_HOSTS.items():
        for sh in (1, 5):
            cell = {}
            for m in ("auroc", "acc", "oscr", "fpr95"):
                d = np.mean([zf[f"C__{h}__{sh}shot__CCP-A__{m}"].astype(np.float64) - zf[f"C__{h}__{sh}shot__COSF__{m}"].astype(np.float64)
                             for h in hosts], axis=0)
                cell[m] = metrics.paired(d, np.zeros_like(d))
            rc[f"{name}|{sh}"] = {"row": "CCP-A", "ref": "COSF", "seeds": len(hosts), **cell}
    return {"label": "derived from the hash-pinned recorded per-episode files; no evaluation, no new test", "sources": SOURCES,
            "cub": cub, "foundation": fm, "resnet_vs_cosf": rc}


def text() -> str:
    return json.dumps(compute(), indent=1) + "\n"


def check() -> bool:
    return text().encode("utf-8") == recorded.read_bytes(RECORDED)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        same = check()
        print(f"{RECORDED}: {'identical to' if same else 'DIFFERS from'} a fresh computation")
        sys.exit(0 if same else 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
