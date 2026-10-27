"""Repository figure (former Supplementary Figure S1): per-episode test AUROC gains of fixed-weight fusion over the trained host's score.

Recorded per-episode test metrics only (ccp/assets.py: test/dnpg_mini, DNPG seeds 0-2, H_fix - D averaged over the seeds per episode;
test/snatcher_mini, SnaTCHer-F, H_fix - S); no evaluation. The data file is compared with results/recorded/figures_v0/fig5_data.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ccp import assets  # noqa: E402
from scripts.figures import style as S  # noqa: E402


def diffs():
    zd, zs = assets.load("test/dnpg_mini"), assets.load("test/snatcher_mini")
    out = {}
    for sh in (1, 5):
        k = f"{sh}shot"
        out[("DNPG", sh)] = np.mean([zd[f"seed{i}_{k}_T_H_fix_auroc"] - zd[f"seed{i}_{k}_T_D_auroc"] for i in range(3)], axis=0)
        out[("SnaTCHer-F", sh)] = zs[f"{k}_T_H_fix_auroc"] - zs[f"{k}_T_S_auroc"]
    return out


def main(out: Path) -> str:
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from scipy.stats import wilcoxon
    S.setup()
    D = diffs()
    fig, axs = S.plt.subplots(1, 2, figsize=(S.W2, 55 * S.MM))
    csv_rows, stats = [], {}
    labels = [r"DNPG: $H_{\mathrm{fix}}$ − host score (mean of 3 seeds)", r"SnaTCHer-F: $H_{\mathrm{fix}}$ − host score"]
    for ax, sh in zip(axs, (1, 5)):
        series = (("DNPG", S.KNOWN, "-"), ("SnaTCHer-F", S.UNKNOWN, "--"))
        lo = np.floor(min(D[(h, sh)].min() for h, *_ in series))
        hi = np.ceil(max(D[(h, sh)].max() for h, *_ in series))
        bins = np.arange(lo, hi + 1, 1.0)
        ymax = 0
        for (h, col, ls), lab in zip(series, labels):
            cnt, edges, _ = ax.hist(D[(h, sh)], bins=bins, histtype="step", color=col, lw=1.2, ls=ls, label=lab)
            ymax = max(ymax, cnt.max())
            csv_rows += [[sh, h, a, b, int(c)] for a, b, c in zip(edges[:-1], edges[1:], cnt)]
        ax.axvline(0, color="0.35", lw=0.8)
        for n, (h, col, _) in enumerate(series):
            d = D[(h, sh)]
            stats[(h, sh)] = (d.mean(), 1.96 * d.std() / np.sqrt(len(d)), wilcoxon(d).pvalue, (d > 0).mean())
            ax.axvline(d.mean(), color=col, lw=1.0, ls=":")
            ax.text(0.98, 0.97 - 0.16 * n, f"{h}: mean {d.mean():+.2f}\n{(d > 0).mean() * 100:.0f}% of episodes > 0", color=col, fontsize=7,
                    va="top", ha="right", transform=ax.transAxes, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.5},
                    zorder=5)
        ax.set_xlabel("Per-episode AUROC difference (points)")
        ax.text(0.02, 0.98, f"{sh}-shot", transform=ax.transAxes, va="top", ha="left", fontsize=7)
        ax.set_ylim(0, ymax * 1.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_ylabel("Episodes (of 600)")
    hnd = [Line2D([], [], color=S.KNOWN, lw=1.2, ls="-"), Line2D([], [], color=S.UNKNOWN, lw=1.2, ls="--")]
    fig.legend(hnd, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    fig.tight_layout(w_pad=1.5)
    S.save(fig, "episode_gains", out, png_dpi=200)
    S.write_csv("episode_gains", ["shot", "host", "bin_lo", "bin_hi", "episodes"], csv_rows)
    f = lambda h, sh: (f"{stats[(h, sh)][0]:+.2f} ± {stats[(h, sh)][1]:.2f} (Wilcoxon p = {stats[(h, sh)][2]:.1e}; "  # noqa: E731
                       f"{stats[(h, sh)][3] * 100:.0f}% of episodes above 0)")
    return ("**Per-episode test gains of fixed-weight fusion (recorded per-episode values; no re-evaluation).** Step histograms (1-point bins) "
            "of the per-episode AUROC difference between fixed-weight fusion H_fix and the trained host's own score over the 600 "
            "mini-ImageNet test episodes per shot: DNPG (averaged over seeds 0-2 per episode; solid, blue) and SnaTCHer-F (one released model "
            "per shot; dashed, orange). Solid gray line: 0; dotted lines: means. Mean and paired 95% CI over episodes: DNPG "
            f"{f('DNPG', 1)} at 1 shot and {f('DNPG', 5)} at 5 shots; SnaTCHer-F {f('SnaTCHer-F', 1)} and {f('SnaTCHer-F', 5)}.")


if __name__ == "__main__":
    print(main(S.GEN))
