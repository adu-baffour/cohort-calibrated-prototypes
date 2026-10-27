"""Figure 3: SCG (stratified cohort gap, Eq. 8) against the paired test AUROC gain of CCP-A over COSF, from recorded outputs only (the
per-episode test metrics of the recorded passes are read to recompute the plotted gains and their CIs, which must equal the recorded
gains; no test manifest, image, feature or score is loaded, and no statistical test is computed).

Points (one per frozen-host configuration with a validation SCG, per shot):
  round (vii): frozen FEAT mini, frozen FEAT tiered, frozen FEAT on CUB with the mini-ImageNet bank, RFS CIFAR-FS;
    validation SCG = results/final_round/d1_val.json, test SCG = results/final_round/test.json; gains from the per-episode file of the
    round (ccp/assets.py: test/final_round; parts C, B, A).
  round (viii): the 32 foundation-model configurations; validation SCG = the valA SCG frozen in the round's bundle
    (results/foundation_round/selection_valA.json), test SCG = results/foundation_round/test.json; gains from test/foundation_round.
(a) x = validation SCG, (b) x = test SCG; y = mean per-episode AUROC(CCP-A) - AUROC(COSF) with its 95% CI (1.96 std / sqrt(600)).
Marker = host, color = dataset, open = 1-shot, filled = 5-shot, dashed zero line; gray rings: validation SCG computed after the
configuration's test pass (asserted from the recorded 'when' fields). Each panel prints the registered Spearman correlation over the 32
foundation-model configurations: (a) H2 (validation SCG), (b) H3 (test SCG), from results/foundation_round/test_hypotheses.json.
The plotted data file must equal the data the paper's figure was drawn from (DATA_SHA256).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ccp import assets, metrics, recorded  # noqa: E402
from scripts.figures import style as S  # noqa: E402

# sha256 of the plotted data of the paper's Figure 3 (release 1.5.0, paper_assets/figures/fig3_scg_gain_data.json)
DATA_SHA256 = "9d304150ff19fddaf7fcd52b3e98d49e7a4778ee328340d376ef07ea5afd3f98"
DATASET_COLOR = {"mini": "#0072B2", "tiered": "#D55E00", "CIFAR-FS": "#009E73", "CUB": "#CC79A7"}   # Okabe-Ito
ALPHA = 0.85
LATE_EDGE = "0.45"
HOST_MARKER = {"frozen FEAT (ResNet-12)": "o", "RFS (ResNet-12)": "s", "CLIP ViT-B/16": "D", "DINOv2 ViT-S/14": "^",
               "DINOv2 ViT-B/14": "v", "DINOv2 ViT-L/14": "P"}
FM_HOST = {"clip_b16": "CLIP ViT-B/16", "dinov2_s14": "DINOv2 ViT-S/14", "dinov2_b14": "DINOv2 ViT-B/14", "dinov2_l14": "DINOv2 ViT-L/14"}
FM_DATASET = {"mini": "mini", "tiered": "tiered", "cifarfs": "CIFAR-FS", "cub": "CUB"}


def gain(z, prefix):
    d = z[prefix + "CCP-A__auroc"].astype(np.float64) - z[prefix + "COSF__auroc"].astype(np.float64)
    return float(d.mean()), metrics.ci95(d), int(d.size)


def points() -> list[dict]:
    pts = []
    fv = recorded.load("results/final_round/d1_val.json")["scg"]
    ft = recorded.load("results/final_round/test.json")["scg"]
    zf = assets.load("test/final_round")
    for key, part, host, ds, bb in (("mini", "C", "frozen_feat_mini", "mini", "frozen FEAT (ResNet-12)"),
                                     ("tiered", "C", "frozen_feat_tiered", "tiered", "frozen FEAT (ResNet-12)"),
                                     ("CUB", "B", "frozen_feat_mini", "CUB", "frozen FEAT (ResNet-12)"),
                                     ("CIFAR-FS", "A", "cifarfs_rfs_seed0", "CIFAR-FS", "RFS (ResNet-12)")):
        for sh in (1, 5):
            m, ci, n = gain(zf, f"{part}__{host}__{sh}shot__")
            pts.append({"round": "final", "dataset": ds, "backbone": bb, "shot": sh, "scg_val": fv[f"{key}|{sh}"]["std_gap"],
                        "scg_test": ft[f"{key}|{sh}"]["std_gap"], "gain": m, "ci95": ci, "n": n,
                        "recorded_gain": ft[f"{key}|{sh}"]["ccpa_minus_cosf_auroc"]})
    sel = recorded.load("results/foundation_round/selection_valA.json")["scg_valA"]
    tt = recorded.load("results/foundation_round/test.json")["scg_test"]
    zm = assets.load("test/foundation_round")
    for b, bb in FM_HOST.items():
        for ds, dname in FM_DATASET.items():
            for sh in (1, 5):
                k = f"{ds}|{b}|{sh}"
                m, ci, n = gain(zm, f"{ds}_{b}__T__{sh}shot__")
                pts.append({"round": "foundation", "dataset": dname, "backbone": bb, "shot": sh, "scg_val": sel[k]["std_gap"],
                            "scg_test": tt[k]["std_gap"], "gain": m, "ci95": ci, "n": n, "recorded_gain": tt[k]["gain_ccpa_minus_cosf"]})
    for p in pts:                                     # the gains recomputed from the per-episode files equal the recorded ones
        assert abs(p["gain"] - p["recorded_gain"]) < 1e-9, p
    return pts


def late_scg(p) -> bool:
    """Frozen FEAT mini and tiered: validation SCG first computed on 2 Oct (variants round), after their test passes (25 and 27 Sep)."""
    return p["round"] == "final" and p["backbone"].startswith("frozen FEAT") and p["dataset"] in ("mini", "tiered")


def check_timing():
    """When each validation SCG was computed relative to the configuration's test pass, from the recorded 'when' fields."""
    var = recorded.load("results/variants/d1/d1.json")
    fv = recorded.load("results/final_round/d1_val.json")
    done = {p: recorded.load(f"results/recorded/test/FINAL_TEST_FSOR_{p}_DONE.json")["when"] for p in ("MINI", "TIERED")}
    for ds in ("mini", "tiered"):                      # the plotted values are the 2 Oct values, reproduced exactly on 5 Oct
        for sh in (1, 5):
            assert var["per_dataset"][ds][str(sh)]["std_gap"] == fv["scg"][f"{ds}|{sh}"]["std_gap"], (ds, sh)
    assert var["when"] > max(done.values())
    final_test = recorded.load("results/final_round/test.json")["when"]
    fm_sel = recorded.load("results/foundation_round/selection_valA.json")["when"]
    fm_test = recorded.load("results/foundation_round/test.json")["when"]
    assert fv["when"] < final_test and fm_sel < fm_test


def plot(pts, h, out: Path):
    from matplotlib.lines import Line2D
    S.setup()
    S.plt.rcParams["errorbar.capsize"] = 1.2
    fig, axes = S.plt.subplots(1, 2, figsize=(S.W2, 82 * S.MM), sharey=True)
    for ax, xkey, xlabel, tag, key in ((axes[0], "scg_val", "SCG on validation", "(a)", "H2"), (axes[1], "scg_test", "SCG on test", "(b)", "H3")):
        ax.axhline(0, color="0.35", lw=0.7, ls="--", zorder=1)
        for p in pts:
            c, mk = DATASET_COLOR[p["dataset"]], HOST_MARKER[p["backbone"]]
            ax.errorbar(p[xkey], p["gain"], yerr=p["ci95"], fmt="none", ecolor=c, elinewidth=0.6, alpha=ALPHA, zorder=2)
            ax.scatter(p[xkey], p["gain"], marker=mk, s=18, linewidths=0.8, edgecolors=c, facecolors=c if p["shot"] == 5 else "white",
                       alpha=ALPHA, zorder=3)
            if late_scg(p):
                ax.scatter(p[xkey], p["gain"], marker=mk, s=70, linewidths=0.9, edgecolors=LATE_EDGE, facecolors="none", zorder=4)
        ax.set_xlabel(xlabel)
        ax.text(-0.02, 1.02, tag, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")
        ax.text(0.98, 0.03, f"{key}: Spearman $\\rho$ = {h[key]['rho']:.3f}\n(32 foundation cells)", transform=ax.transAxes, va="bottom", ha="right")
    axes[0].set_ylabel("Test AUROC gain of CCP-A over COSF (points)")
    handles = [Line2D([], [], marker=m, ls="none", markerfacecolor="0.45", markeredgecolor="0.45", markersize=4.5, label=b)
               for b, m in HOST_MARKER.items()]
    handles += [Line2D([], [], marker="s", ls="none", markerfacecolor=c, markeredgecolor=c, markersize=4.5, alpha=ALPHA, label=d)
                for d, c in DATASET_COLOR.items()]
    handles += [Line2D([], [], marker="o", ls="none", markerfacecolor="white", markeredgecolor="0.3", markersize=4.5, label="1-shot"),
                Line2D([], [], marker="o", ls="none", markerfacecolor="0.3", markeredgecolor="0.3", markersize=4.5, label="5-shot"),
                Line2D([], [], marker="o", ls="none", markerfacecolor="none", markeredgecolor=LATE_EDGE, markersize=8, markeredgewidth=0.9,
                       label="SCG (val.) computed after the test pass")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    S.save(fig, "fig3_scg_gain", out)


def data_json(pts) -> str:
    return json.dumps(pts, indent=1)


def main(out: Path):
    pts = points()
    h = recorded.load("results/foundation_round/test_hypotheses.json")
    check_timing()
    plot(pts, h, out)
    S.GEN.mkdir(parents=True, exist_ok=True)
    (S.GEN / "fig3_scg_gain_data.json").write_text(data_json(pts), encoding="utf-8", newline="\n")
    print(f"Figure 3: {len(pts)} points; H2 rho = {h['H2']['rho']:.3f}, H3 rho = {h['H3']['rho']:.3f}")
    return pts


if __name__ == "__main__":
    main(S.GEN)
