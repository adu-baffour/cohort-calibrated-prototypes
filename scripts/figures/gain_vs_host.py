"""Repository figure (former Supplementary Figure S3): test AUROC gain of CCP against the host's own test AUROC. Recorded numbers only:
the round (i)-(iv) records (results/recorded/test/FINAL_TEST_FSOR_*_DONE.json) and, for DNPG's paired CI, the recorded per-episode test
metrics (ccp/assets.py: test/dnpg_mini). The data file is compared with results/recorded/figures_v0/fig4_data.csv.

x = host's own AUROC on test; y = gain on test:
  frozen hosts (circles): frozen FEAT, FRN (mean of seeds 0-2), TANE pre-trained (mini-ImageNet); frozen FEAT and TANE pre-trained
  (tieredImageNet): CCP-A - COS; trained hosts (squares): SnaTCHer-F H_fix - host score (paired 95% CI over 600 episodes); DNPG (mean of
  seeds 0-2): H_fix - host score (95% CI over 600 episodes of the seed-averaged per-episode difference).
Filled = mini-ImageNet, open = tieredImageNet. TANE (dagger): released checkpoints whose stored epochs are consistent with selection on the
test split.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ccp import assets, recorded  # noqa: E402
from scripts.figures import style as S  # noqa: E402

T = "†"
# label positions in data coordinates (x, y, ha, leader line), set so that no label overlaps a marker, an error bar or another label
POS = {(1, "Frozen FEAT", "mini"): (68.55, 2.3, "left", False), (1, f"TANE pre-trained{T}", "mini"): (70.5, 2.9, "left", False),
       (1, "SnaTCHer-F", "mini"): (70.25, 1.67, "left", False), (1, "FRN (3 seeds)", "mini"): (69.85, 0.95, "left", False),
       (1, "DNPG (3 seeds)", "mini"): (73.2, 0.12, "right", True), (1, "Frozen FEAT", "tiered"): (74.35, 1.45, "left", True),
       (1, f"TANE pre-trained{T}", "tiered"): (74.45, 0.2, "left", False),
       (5, "Frozen FEAT", "mini"): (74.95, 4.1, "left", False), (5, f"TANE pre-trained{T}", "mini"): (77.1, 4.61, "left", False),
       (5, "FRN (3 seeds)", "mini"): (74.1, 1.7, "left", False), (5, "SnaTCHer-F", "mini"): (77.65, 2.1, "left", False),
       (5, "DNPG (3 seeds)", "mini"): (83.5, 1.45, "left", True), (5, "Frozen FEAT", "tiered"): (78.6, 1.25, "left", True),
       (5, f"TANE pre-trained{T}", "tiered"): (81.2, 0.18, "right", True)}
XLIM = {1: (66.8, 77.6), 5: (73.2, 86.8)}


def points():
    J = lambda p: recorded.load(f"results/recorded/test/FINAL_TEST_FSOR_{p}_DONE.json", paper_names=False)  # noqa: E731
    m, t, s, d = J("MINI"), J("TIERED"), J("MINI-SNATCHER"), J("MINI-DNPG")
    z = assets.load("test/dnpg_mini")
    P = []
    for sh in (1, 5):
        k = f"{sh}shot"
        for name, rows, ds in (("Frozen FEAT", [m["rows"]["frozen_FEAT"]], "mini"),
                               ("FRN (3 seeds)", [m["rows"][f"FRN_seed{i}"] for i in range(3)], "mini"),
                               (f"TANE pre-trained{T}", [m["rows"]["TANE_DNPG_pretrain"]], "mini"),
                               ("Frozen FEAT", [t["rows"]["frozen_FEAT_tiered"]], "tiered"),
                               (f"TANE pre-trained{T}", [t["rows"]["TANE_tiered_pretrain"]], "tiered")):
            x = float(np.mean([r[k]["COS"]["auroc"] for r in rows]))
            y = float(np.mean([r[k]["CCP-A"]["auroc"] - r[k]["COS"]["auroc"] for r in rows]))
            P.append({"shot": sh, "host": name, "dataset": ds, "kind": "backbone + CCP-A", "x": x, "y": y, "ci": None})
        P.append({"shot": sh, "host": "SnaTCHer-F", "dataset": "mini", "kind": "trained host + H_fix", "x": s["rows"][f"{k}_T"]["S"]["auroc"],
                  "y": s["paired"][f"{k}_T"]["Hfix_minus_S"]["mean"], "ci": s["paired"][f"{k}_T"]["Hfix_minus_S"]["ci95"]})
        dd = np.mean([z[f"seed{i}_{k}_T_H_fix_auroc"] - z[f"seed{i}_{k}_T_D_auroc"] for i in range(3)], axis=0)
        P.append({"shot": sh, "host": "DNPG (3 seeds)", "dataset": "mini", "kind": "trained host + H_fix",
                  "x": d["seed_summary"][f"{k}_T"]["D"]["auroc"]["mean"], "y": d["seed_summary"][f"{k}_T"]["Hfix_minus_D"]["mean"],
                  "ci": float(1.96 * dd.std() / np.sqrt(len(dd)))})
        assert abs(dd.mean() - d["seed_summary"][f"{k}_T"]["Hfix_minus_D"]["mean"]) < 1e-9
    return P


def main(out: Path) -> str:
    from matplotlib.lines import Line2D
    from scipy.stats import spearmanr
    S.setup()
    P = points()
    mini_pts = [p for p in P if p["dataset"] == "mini"]
    rho, pv = spearmanr([p["x"] for p in mini_pts], [p["y"] for p in mini_pts])
    per_shot = {sh: spearmanr([p["x"] for p in mini_pts if p["shot"] == sh], [p["y"] for p in mini_pts if p["shot"] == sh]) for sh in (1, 5)}
    fig, axs = S.plt.subplots(1, 2, figsize=(S.W2, 62 * S.MM), sharey=False)
    for ax, sh in zip(axs, (1, 5)):
        for p in [q for q in P if q["shot"] == sh]:
            mk = "o" if p["kind"].startswith("backbone") else "s"
            filled = p["dataset"] == "mini"
            if p["ci"] is not None:
                ax.errorbar(p["x"], p["y"], yerr=p["ci"], fmt="none", ecolor="0.3", elinewidth=0.8, capsize=1.5, zorder=2)
            ax.plot(p["x"], p["y"], marker=mk, ms=4.5, ls="none", color=S.KNOWN, mfc=S.KNOWN if filled else "white", mew=1.0, zorder=3)
            tx, ty, ha, arrow = POS[(sh, p["host"], p["dataset"])]
            ax.annotate(p["host"] + ("" if filled else "\n(tiered)"), (p["x"], p["y"]), xytext=(tx, ty), textcoords="data", fontsize=7, ha=ha,
                        va="center", linespacing=1.0,
                        arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 0.5, "shrinkA": 1, "shrinkB": 3} if arrow else None)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlabel("Host's own AUROC on test")
        ax.text(0.02, 0.98, f"{sh}-shot", transform=ax.transAxes, va="top", ha="left", fontsize=7)
        ax.set_xlim(*XLIM[sh])
    axs[0].set_ylabel("AUROC gain on test")
    ymax = max(p["y"] + (p["ci"] or 0) for p in P) + 0.3
    for ax in axs:
        ax.set_ylim(-0.2, ymax)
    h = [Line2D([], [], marker="o", color=S.KNOWN, ls="none", ms=4, label="Frozen host + CCP-A (CCP-A − COS)"),
         Line2D([], [], marker="s", color=S.KNOWN, ls="none", ms=4, label=r"Trained host + CCP ($H_{\mathrm{fix}}$ − host score)"),
         Line2D([], [], marker="o", color=S.KNOWN, mfc="white", ls="none", ms=4, label="Open marker: tieredImageNet")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7, columnspacing=1.2, handletextpad=0.3)
    fig.tight_layout(w_pad=1.5)
    S.save(fig, "gain_vs_host", out, png_dpi=200)
    S.write_csv("gain_vs_host", ["shot", "host", "dataset", "kind", "host_test_auroc", "test_gain", "gain_ci95"],
                [[p["shot"], p["host"], p["dataset"], p["kind"], round(p["x"], 4), round(p["y"], 4), "" if p["ci"] is None else round(p["ci"], 4)]
                 for p in P] + [["all", "spearman_mini_10_points", "", "", "", round(float(rho), 4), f"p={pv:.4g}"]])
    assert min(p["y"] - (p["ci"] or 0) for p in P) > 0
    return ("**Test gain against the host's own AUROC (recorded test numbers; no re-evaluation).** Each point is one host on the 600 fixed "
            "test episodes per shot. x: the host's own AUROC on test (COS for frozen hosts, the host score for SnaTCHer-F and DNPG). y: the "
            "AUROC gain, CCP-A − COS for frozen hosts (circles) and H_fix − host score for trained hosts (squares). Filled: mini-ImageNet; "
            "open: tieredImageNet (BGR input). FRN and DNPG are means over seeds 0-2. Error bars (trained hosts only): paired 95% CIs over "
            f"the 600 episodes (DNPG: of the seed-averaged per-episode difference). Every gain is above zero (smallest {min(p['y'] for p in P):+.2f}). "
            f"No monotone relation: Spearman ρ = {rho:.2f} (p = {pv:.2f}) over the ten mini-ImageNet points; within a shot ρ = "
            f"{per_shot[1].statistic:.2f} (1 shot, p = {per_shot[1].pvalue:.2f}) and {per_shot[5].statistic:.2f} (5 shots, p = "
            f"{per_shot[5].pvalue:.2f}). † Released TANE checkpoints whose stored epochs are consistent with selection on the test split.")


if __name__ == "__main__":
    print(main(S.GEN))
