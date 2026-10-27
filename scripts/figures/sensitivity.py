"""Repository figure (former Supplementary Figure S2): sensitivity to the cohort size K and the fusion-weight factor (validation only; not
used for any selection). From results/recorded/validation/ccp_ablations.json and, for the paired differences of (b), the recorded DNPG
validation scores (ccp/assets.py: validation/dnpg_mini/*). The data file is compared with results/recorded/figures_v0/fig6_data.csv.

(a) valB AUROC against K in {100, 300, 1000, 3000} (log axis) plus "all" (the whole bank) after an axis break: frozen FEAT mini (CCP-A),
    frozen FEAT tiered (CCP-A, BGR input), DNPG + CCP (H_fix with K, w refitted on valA, mean of seeds 0-2); 1-shot solid, 5-shot dashed;
    registered K = 1000 dotted.
(b) DNPG H_fix valB AUROC against the fusion-weight factor (H_fix = D + factor * w * CCP-A, w from valA), mean of seeds 0-2; references:
    the host score D alone and CCP-A alone; registered factor 1 dotted.
Error bars: 95% CI over the 1,000 episodes (DNPG: mean over seeds of the per-seed CIs).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ccp import assets, recorded  # noqa: E402
from scripts.figures import style as S  # noqa: E402

KS = [100, 300, 1000, 3000]
KEY = {100: "K=100", 300: "K=300", 1000: "CCP-A (registered)", 3000: "K=3000", "all": "K=all"}
FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]
HOSTS = [("Frozen FEAT (mini)", "FEAT mini"), ("Frozen FEAT (tiered)", "FEAT tiered"), ("DNPG + CCP (H_fix)", "DNPG + CCP")]


def data():
    H = recorded.load("results/recorded/validation/ccp_ablations.json", paper_names=False)["hosts"]
    fm, ft = H["frozen FEAT (mini)"]["rows"], H["frozen FEAT tiered (BGR)"]["rows"]
    dn = H["DNPG seeds 0-2 (mini)"]["per_seed"]
    dm = lambda key, sh, f="auroc": float(np.mean([dn[f"seed{s}_{sh}shot"][key][f] for s in (0, 1, 2)]))  # noqa: E731
    A = {}
    for sh in (1, 5):
        for k in KS + ["all"]:
            A[("Frozen FEAT (mini)", sh, k)] = (fm[f"{KEY[k]}|B{sh}"]["auroc"], fm[f"{KEY[k]}|B{sh}"]["auroc_ci95"])
            A[("Frozen FEAT (tiered)", sh, k)] = (ft[f"{KEY[k]}|B{sh}"]["auroc"], ft[f"{KEY[k]}|B{sh}"]["auroc_ci95"])
            A[("DNPG + CCP (H_fix)", sh, k)] = (dm(f"H_fix({KEY[k]})|B", sh), dm(f"H_fix({KEY[k]})|B", sh, "auroc_ci95"))
    B = {}
    for sh in (1, 5):
        for f in FACTORS:
            B[(sh, f)] = (dm(f"w x {f}|B", sh), dm(f"w x {f}|B", sh, "auroc_ci95"))
        B[(sh, "D")] = (dm("D|B", sh), dm("D|B", sh, "auroc_ci95"))
        B[(sh, "CCP-A")] = (dm("CCP-A (registered)|B", sh), dm("CCP-A (registered)|B", sh, "auroc_ci95"))
    return A, B


def paired():
    """(b) paired differences from the registered factor 1, from the recorded DNPG validation scores: per seed, w = pooled std(D_A) /
    pooled std(CCPA_A), per-episode valB AUROC of D_B + factor w CCPA_B minus that of factor 1, averaged over seeds 0-2 per episode;
    mean and 95% CI (1.96 std / sqrt(1000)) over the 1,000 valB episodes."""
    from ccp.metrics import episode_auroc
    P, level = {}, {}
    for sh in (1, 5):
        per = {f: [] for f in FACTORS}
        for sd in (0, 1, 2):
            z = assets.load(f"validation/dnpg_mini/seed{sd}_{sh}shot")
            w = float(z["D_A"].std() / z["CCPA_A"].std())
            for f in FACTORS:
                per[f].append(episode_auroc(z["D_B"] + f * w * z["CCPA_B"]))
        for f in FACTORS:
            level[(sh, f)] = float(np.mean([v.mean() for v in per[f]]))
            d = np.mean([a - b for a, b in zip(per[f], per[1.0])], axis=0)
            P[(sh, f)] = (float(d.mean()), float(1.96 * d.std() / np.sqrt(len(d))))
    return P, level


def main(out: Path) -> str:
    from matplotlib.lines import Line2D
    S.setup()
    A, B = data()
    P, level = paired()
    for sh in (1, 5):
        for f in FACTORS:
            assert abs(level[(sh, f)] - B[(sh, f)][0]) < 0.01, (sh, f, level[(sh, f)], B[(sh, f)][0])
    style = {"Frozen FEAT (mini)": (S.KNOWN, "o"), "Frozen FEAT (tiered)": (S.UNKNOWN, "D"), "DNPG + CCP (H_fix)": (S.THIRD, "s")}
    fig = S.plt.figure(figsize=(S.W2, 68 * S.MM))
    gs = fig.add_gridspec(1, 4, width_ratios=[3.3, 0.42, 1.9, 1.9], wspace=0.35)
    ax, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    for name, short in HOSTS:
        col, mk = style[name]
        for sh, ls in ((1, "-"), (5, "--")):
            ys = [A[(name, sh, k)][0] for k in KS]
            es = [A[(name, sh, k)][1] for k in KS]
            ax.errorbar(KS, ys, yerr=es, color=col, ls=ls, marker=mk, ms=3.2, lw=1.1, elinewidth=0.6, capsize=1.2, mfc=col if sh == 1 else "white")
            ya, ea = A[(name, sh, "all")]
            axb.errorbar([0], [ya], yerr=[ea], color=col, marker=mk, ms=3.2, ls="none", elinewidth=0.6, capsize=1.2, mfc=col if sh == 1 else "white")
            nudge = {("Frozen FEAT (tiered)", 5): +0.75, ("DNPG + CCP (H_fix)", 1): -0.55}.get((name, sh), 0.0)
            ax.text(82, ys[0] + nudge, f"{short}, {sh}-shot", color=col if col != S.THIRD else "#0f7a55", fontsize=7, ha="right", va="center")
    ax.set_xscale("log")
    ax.set_xlim(1.0, 4200)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.minorticks_off()
    ax.axvline(1000, color="0.3", lw=0.8, ls=":")
    ax.text(1000, 1.01, "registered $K$ = 1000", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color="0.3")
    ax.set_xlabel("Cohort size $K$ (top-$K$ base images, log scale)")
    ax.set_ylabel("valB AUROC")
    axb.set_xlim(-1, 1)
    axb.set_xticks([0])
    axb.set_xticklabels(["all"])
    lo = min(v[0] - v[1] for v in A.values()) - 0.8
    hi = max(v[0] + v[1] for v in A.values()) + 0.8
    ax.set_ylim(lo, hi)
    axb.set_ylim(lo, hi)
    axb.tick_params(axis="y", left=False, labelleft=False)
    axb.spines["left"].set_visible(False)
    for a_, x0 in ((ax, 1.0), (axb, 0.0)):   # axis-break marks
        d = 0.015
        w = d * (3.3 / 0.42) if a_ is axb else d
        a_.plot((x0 - w, x0 + w), (-d * 1.6, d * 1.6), transform=a_.transAxes, color="0.2", clip_on=False, lw=0.7)
    ax.spines["right"].set_visible(False)
    S.panel_label(ax, -0.13, 1.06, "(a)")
    ax.legend(handles=[Line2D([], [], color="0.3", ls="-", marker="o", ms=3, label="1-shot (filled)"),
                       Line2D([], [], color="0.3", ls="--", marker="o", mfc="white", ms=3, label="5-shot (open)")],
              loc="lower right", bbox_to_anchor=(1.0, 0.13), fontsize=7, frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
    for j, sh in enumerate((1, 5)):
        bx = fig.add_subplot(gs[0, 2 + j])
        ys = [B[(sh, f)][0] for f in FACTORS]
        es = [B[(sh, f)][1] for f in FACTORS]
        bx.errorbar(FACTORS, ys, yerr=es, color=S.KNOWN, marker="o", ms=3.2, lw=1.1, elinewidth=0.6, capsize=1.2,
                    label="$H_{\\mathrm{fix}}$\n(DNPG + CCP)")
        bx.axhline(B[(sh, "D")][0], color=S.UNKNOWN, ls="--", lw=1.0)
        bx.axhline(B[(sh, "CCP-A")][0], color="0.35", ls="-.", lw=1.0)
        bx.axvline(1.0, color="0.3", lw=0.8, ls=":")
        bx.set_xscale("log")
        bx.set_xticks(FACTORS)
        bx.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
        bx.minorticks_off()
        bx.set_xlabel("Fusion-weight factor")
        bx.set_ylim(min(min(ys) - max(es), B[(sh, "D")][0]) - 0.35, max(max(ys) + max(es), B[(sh, "CCP-A")][0]) + 0.6)
        bx.text(1.1, B[(sh, "D")][0] - 0.03, "host score alone", color=S.UNKNOWN, fontsize=7, ha="left", va="top")
        bx.text(1.1, B[(sh, "CCP-A")][0] - 0.03, "CCP-A alone", color="0.3", fontsize=7, ha="left", va="top")
        bx.text(0.98, 0.02, f"{sh}-shot", transform=bx.transAxes, ha="right", va="bottom", fontsize=7)
        bx.set_xlim(0.2, 5)
        if j == 0:
            bx.set_ylabel("DNPG valB AUROC")
            S.panel_label(bx, -0.3, 1.06, "(b)")
            bx.text(1.0, 1.01, "registered factor 1", transform=bx.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color="0.3")
            bx.legend(loc="upper left", fontsize=7, handlelength=1.6, frameon=True, facecolor="white", edgecolor="none", framealpha=1.0).set_zorder(5)
    S.save(fig, "sensitivity", out, png_dpi=200)
    rows = [["a", n, sh, k, round(A[(n, sh, k)][0], 4), round(A[(n, sh, k)][1], 4)] for n, _ in HOSTS for sh in (1, 5) for k in KS + ["all"]]
    rows += [["b", "DNPG H_fix", sh, f, round(B[(sh, f)][0], 4), round(B[(sh, f)][1], 4)] for sh in (1, 5) for f in FACTORS]
    rows += [["b", ref, sh, "", round(B[(sh, ref)][0], 4), round(B[(sh, ref)][1], 4)] for sh in (1, 5) for ref in ("D", "CCP-A")]
    rows += [["b", "DNPG H_fix paired diff vs m=1 (mean of seeds per episode)", sh, f, round(P[(sh, f)][0], 4), round(P[(sh, f)][1], 4)]
             for sh in (1, 5) for f in FACTORS if f != 1.0]
    S.write_csv("sensitivity", ["panel", "series", "shot", "K_or_m", "valB_auroc", "ci95"], rows)
    a = lambda n, sh, k: A[(n, sh, k)][0]  # noqa: E731
    pd = lambda sh: "; ".join(f"factor {f:g}: {P[(sh, f)][0]:+.2f} ± {P[(sh, f)][1]:.2f}" for f in FACTORS if f != 1.0)  # noqa: E731
    fm = "Frozen FEAT (mini)"
    return ("**Sensitivity to the cohort size and the fusion weight (validation only; not used for any selection).** (a) valB AUROC against "
            "the cohort size K (log scale; \"all\" = mean over the whole bank, after the axis break) for frozen FEAT on mini-ImageNet (CCP-A), "
            "frozen FEAT on tieredImageNet (CCP-A, BGR input) and DNPG + CCP (H_fix with its weight refitted on valA for each K, mean of seeds "
            "0-2); 1-shot solid and filled, 5-shot dashed and open; registered K = 1000 dotted. Frozen FEAT mini-ImageNet, 1 / 5 shots: "
            f"{a(fm, 1, 100):.2f} / {a(fm, 5, 100):.2f} (K = 100), {a(fm, 1, 1000):.2f} / {a(fm, 5, 1000):.2f} (K = 1000), "
            f"{a(fm, 1, 3000):.2f} / {a(fm, 5, 3000):.2f} (K = 3000), {a(fm, 1, 'all'):.2f} / {a(fm, 5, 'all'):.2f} (all). "
            "(b) DNPG valB AUROC of H_fix = host score + factor · w · CCP-A against the fusion-weight factor (w fitted on valA; mean of seeds "
            "0-2), with the host score alone (dashed) and CCP-A alone (dash-dot); registered factor 1 dotted. Paired AUROC difference from "
            f"factor 1 (mean and 95% CI over the 1,000 valB episodes): 1 shot {pd(1)}; 5 shots {pd(5)}. Error bars are per-row 95% CIs over "
            "episodes (DNPG: mean of the per-seed CIs); all rows use the same episodes, so differences between rows are paired.")


if __name__ == "__main__":
    print(main(S.GEN))
