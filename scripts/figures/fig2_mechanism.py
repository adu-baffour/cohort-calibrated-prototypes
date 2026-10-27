"""Figure 2: cohort affinity of known and unknown queries (validation only: frozen FEAT, mini-ImageNet valB; registered CCP-A). Full width.

(a) Per-query scatter, first 40 valB 1-shot episodes: x = raw max cosine to the prototypes (COSF), y = the query's cohort statistic m_K(q)
    (Eq. 3: mean of the top-1000 cosines to the 38,400 base images, mirror views), with 2-D density contours per group. Lines: the pooled
    equal-error-rate (EER) thresholds of cosine alone (vertical) and of the query-only score max_k c_k - m_K(q).
(b) Mean m_K(q) with its 95% CI over the queries in each of 10 decile bins of the raw max cosine (known and unknown pooled for the binning;
    all 1,000 valB episodes), known (blue circles, solid) and unknown (orange triangles, dashed); 1 shot top, 5 shots bottom.
The decomposition into query-only and prototype-only correction (panel (c) of the earlier version) is Table 7(a); its rows are still
written to the data file, which is compared with the recorded results/recorded/figures_v0/fig2_data.csv.
Inputs: the validation score cache (scripts/figures/cache.py) and the recorded ablation JSON. CPU only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ccp import recorded  # noqa: E402
from scripts.figures import cache, style as S  # noqa: E402

N_PLOT = 40
N_BINS = 10
ABLATIONS = "results/recorded/validation/ccp_ablations.json"
VARIANTS = [("COSF", "COSF"), ("query-side only", "Query-side"), ("prototype-side only", "Proto-side"), ("CCP-A (registered)", "CCP-A")]


def query_stats(C, shot):
    """X (raw max cosine), M (m_K(q)) and Y (1 = unknown) for all 1,000 valB episodes, from the score cache."""
    X, M = C[f"x_{shot}shot"], C[f"m_{shot}shot"]
    Y = np.tile(np.r_[np.zeros(75), np.ones(75)], (X.shape[0], 1))
    return X, M, Y


def eer_threshold(score, y):
    best, bt = 9, None
    for t in np.quantile(score, np.linspace(0.001, 0.999, 4000)):
        fpr = (score[y == 0] >= t).mean()
        fnr = (score[y == 1] < t).mean()
        if abs(fpr - fnr) < best:
            best, bt = abs(fpr - fnr), t
    return bt


def pooled_auroc(score, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, score) * 100


def cond_means(x, m, y):
    edges = np.quantile(x, np.linspace(0, 1, N_BINS + 1))
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, N_BINS - 1)
    out = []
    for i in range(N_BINS):
        for k in (0, 1):
            sel = (b == i) & (y == k)
            v = m[sel]
            out.append({"bin": i + 1, "x_lo": float(edges[i]), "x_hi": float(edges[i + 1]), "x_mean": float(x[sel].mean()),
                        "group": "known" if k == 0 else "unknown", "n": int(sel.sum()), "m_mean": float(v.mean()),
                        "m_ci95": float(1.96 * v.std() / np.sqrt(len(v)))})
    return out


def main(out: Path):
    from matplotlib.lines import Line2D
    from scipy.stats import gaussian_kde
    S.setup()
    C = cache.load()
    st = {sh: query_stats(C, sh) for sh in (1, 5)}
    X, M, Y = st[1]
    corr = {g: float(np.corrcoef(X.ravel()[Y.ravel() == k], M.ravel()[Y.ravel() == k])[0, 1]) for g, k in (("known", 0), ("unknown", 1))}
    x, m, y = X[:N_PLOT].ravel(), M[:N_PLOT].ravel(), Y[:N_PLOT].ravel()
    s_cos, s_q = -x, -(x - m)                     # query-only score (coefficient 1)
    t_cos, t_q = eer_threshold(s_cos, y), eer_threshold(s_q, y)
    au_cos, au_q = pooled_auroc(s_cos, y), pooled_auroc(s_q, y)
    cm = {sh: cond_means(st[sh][0].ravel(), st[sh][1].ravel(), st[sh][2].ravel()) for sh in (1, 5)}
    abl = recorded.load(ABLATIONS, paper_names=False)["hosts"]["frozen FEAT (mini)"]["rows"]

    fig = S.plt.figure(figsize=(S.W2, 66 * S.MM))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.30)
    # ---- (a)
    ax = fig.add_subplot(outer[0, 0])
    for k, col, mk in ((0, S.KNOWN, "o"), (1, S.UNKNOWN, "^")):
        ax.scatter(x[y == k], m[y == k], s=1.0, marker=mk, color=col, alpha=0.25, linewidths=0, rasterized=True)
    gx = np.linspace(np.quantile(x, 0.002), np.quantile(x, 0.998), 150)
    gy = np.linspace(np.quantile(m, 0.002), np.quantile(m, 0.998), 150)
    GX, GY = np.meshgrid(gx, gy)
    for k, col, ls in ((0, S.KNOWN, "-"), (1, S.UNKNOWN, "--")):
        kde = gaussian_kde(np.vstack([x[y == k], m[y == k]]))
        Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
        dens = kde(np.vstack([x[y == k], m[y == k]]))
        ax.contour(GX, GY, Z, levels=np.sort(np.quantile(dens, [0.75, 0.40, 0.10])), colors=col, linewidths=0.8, linestyles=ls)
    ax.axvline(-t_cos, color="0.25", lw=1.0, ls=":")
    xs = np.array([gx[0], gx[-1]])
    ax.plot(xs, xs + t_q, color="0.25", lw=1.0, ls="-.")        # x - m = -t_q  ->  m = x + t_q
    ax.set_xlim(gx[0], gx[-1])
    ax.set_ylim(gy[0] - 0.42 * (gy[-1] - gy[0]), gy[-1])        # empty band at the bottom holds the legend
    ax.set_xlabel("Raw max cosine to the prototypes")
    ax.set_ylabel(r"Cohort statistic $m_K(q)$ ($K$ = 1000)")
    ax.legend(handles=[Line2D([], [], marker="o", color=S.KNOWN, ls="-", ms=3, lw=0.8, label="Known"),
                       Line2D([], [], marker="^", color=S.UNKNOWN, ls="--", ms=3, lw=0.8, label="Unknown"),
                       Line2D([], [], color="0.25", ls=":", lw=1.0, label="cosine, EER"),
                       Line2D([], [], color="0.25", ls="-.", lw=1.0, label=r"$\mathrm{max}_k\,c_k - m_K(q)$, EER")],
              loc="lower right", fontsize=7, handlelength=2.0, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    S.panel_label(ax, -0.14, 1.02, "(a)")
    # ---- (b)
    inner_b = outer[0, 1].subgridspec(2, 1, hspace=0.18)
    for j, sh in enumerate((1, 5)):
        bx = fig.add_subplot(inner_b[j, 0])
        for g, col, mk, ls in (("known", S.KNOWN, "o", "-"), ("unknown", S.UNKNOWN, "^", "--")):
            pts = [r for r in cm[sh] if r["group"] == g]
            bx.errorbar([r["x_mean"] for r in pts], [r["m_mean"] for r in pts], yerr=[r["m_ci95"] for r in pts], color=col, marker=mk, ms=2.8,
                        ls=ls, lw=1.0, elinewidth=0.6, capsize=1.0, label="Known" if g == "known" else "Unknown")
        bx.text(0.03, 0.95, f"{sh}-shot", transform=bx.transAxes, va="top", ha="left", fontsize=7)
        bx.set_ylabel(r"Mean $m_K(q)$", fontsize=7)
        if j == 0:
            S.panel_label(bx, -0.2, 1.04, "(b)")
            bx.tick_params(labelbottom=False)
            bx.legend(loc="lower right", fontsize=7, handlelength=2.0)
        else:
            bx.set_xlabel("Raw max cosine (decile-bin mean)")
    S.save(fig, "fig2_mechanism", out)

    rows = [["a", "known" if yy == 0 else "unknown", round(float(xx), 5), round(float(mm), 5), "", ""] for xx, mm, yy in zip(x, m, y)]
    rows += [["a-meta", "eer_threshold_cosine_x", round(-t_cos, 5), "", "", ""],
             ["a-meta", "eer_threshold_onesided_x_minus_m", round(-t_q, 5), "", "", ""],
             ["a-meta", "pooled_auroc_cosine", round(au_cos, 3), "", "", ""], ["a-meta", "pooled_auroc_onesided_query", round(au_q, 3), "", "", ""]]
    rows += [["b", f"{sh}shot {r['group']} bin{r['bin']} x[{r['x_lo']:.4f},{r['x_hi']:.4f}] n={r['n']}", round(r["x_mean"], 5), "",
              round(r["m_mean"], 5), round(r["m_ci95"], 5)] for sh in (1, 5) for r in cm[sh]]
    rows += [["c", f"{k} {mt}", sh, "", round(abl[f"{k}|B{sh}"][mt], 4), round(abl[f"{k}|B{sh}"][f"{mt}_ci95"], 4)]
             for k, _ in VARIANTS for sh in (1, 5) for mt in ("auroc", "acc")]
    rows += [["corr", g, "all 1,000 valB 1-shot episodes", "", round(v, 4), ""] for g, v in corr.items()]
    S.write_csv("fig2", ["panel", "series", "x", "y", "value", "ci95"], rows)
    for sh in (1, 5):
        higher = sum(1 for i in range(N_BINS)
                     if cm[sh][2 * i + 1]["m_mean"] - cm[sh][2 * i + 1]["m_ci95"] > cm[sh][2 * i]["m_mean"] + cm[sh][2 * i]["m_ci95"])
        print(f"Figure 2(b) {sh}-shot: unknown above known with non-overlapping CIs in {higher} of {N_BINS} bins; lowest bin known "
              f"{cm[sh][0]['m_mean']:.3f} / unknown {cm[sh][1]['m_mean']:.3f}, highest bin {cm[sh][-2]['m_mean']:.3f} / {cm[sh][-1]['m_mean']:.3f}")


if __name__ == "__main__":
    main(S.GEN)
