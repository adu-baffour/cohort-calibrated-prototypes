"""Figure 4: queries with the largest unknown-score rank changes under CCP-A (validation only: frozen FEAT, mini-ImageNet valB, 1 shot).

SELECTION RULE (fixed before any image was looked at):
  For every valB 1-shot query, its openness percentile within its episode = 100 * (average rank of its score among the episode's 150 queries,
  0-based) / 149, higher = more unknown, under COSF (score = -max cosine) and under CCP-A (Eqs. 4-5). delta = pct_CCP-A - pct_COSF.
  Rows 1-3: the three UNKNOWN queries with the largest delta (CCP-A pushed them toward unknown).
  Row 4: the KNOWN query with the most negative delta (CCP-A moved it toward known).
  Row 5: the KNOWN query with the largest delta (failure case: CCP-A pushed it toward unknown).
  Ties broken by episode index, then query index; at most one query per episode over the whole figure (rows chosen in the order 1-5).
  The selection is (episode, query) = (375, 99), (156, 103), (385, 79), (258, 9), (459, 61); it is recomputed and must equal SELECTION.
Each row: the query (thin border: blue = known, orange = unknown), the support image of its nearest class by raw cosine, its scores (raw max
cosine, m_K(q), both percentiles), and its 5 nearest base images in the cohort bank with their cosines; ImageNet class names (WordNet,
first lemma; configs/datasets/mini_class_names.tsv) in small gray text. Inputs: the validation score cache (scores, nearest neighbors), the valB
manifest and, for the image panels only, the prepared mini-ImageNet eval84 arrays ($CCP_DATA; images are not redistributed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ccp.episodes import Episodes  # noqa: E402
from scripts.figures import cache, style as S  # noqa: E402

SELECTION = [(375, 99), (156, 103), (385, 79), (258, 9), (459, 61)]  # as in results/recorded/figures_v0/fig3_data.csv
N_TRAIN = 38400                                                       # mini-ImageNet base images; valB images follow in the global index


def pct(s):
    from scipy.stats import rankdata
    return 100.0 * (rankdata(s) - 1) / (len(s) - 1)


def select(C):
    cf, ca = C["cosf_1shot"], C["ccpa_1shot"]
    recs = []   # (episode, query, is_unknown, delta, pct_cos, pct_ccp)
    for e in range(cf.shape[0]):
        pc, pa = pct(cf[e]), pct(ca[e])
        for q in range(150):
            recs.append((e, q, int(q >= 75), pa[q] - pc[q], pc[q], pa[q]))
    recs = np.array(recs, dtype=float)
    used, chosen = set(), []

    def pick(mask, sign, n, tag):
        idx = np.where(mask)[0]
        order = sorted(idx, key=lambda i: (-sign * recs[i, 3], recs[i, 0], recs[i, 1]))
        k = 0
        for i in order:
            if int(recs[i, 0]) in used:
                continue
            used.add(int(recs[i, 0]))
            chosen.append((tag, i))
            k += 1
            if k == n:
                break
    pick(recs[:, 2] == 1, +1, 3, "unknown, largest increase")
    pick(recs[:, 2] == 0, -1, 1, "known, largest decrease")
    pick(recs[:, 2] == 0, +1, 1, "known, largest increase (failure)")
    return recs, chosen


def class_names():
    rows = [line.split("\t") for line in (ROOT / "configs" / "datasets" / "mini_class_names.tsv").read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")][1:]
    return {int(r[0]): r[2].split(",")[0].strip() for r in rows}


def main(out: Path, draw_images: bool = True):
    from ccp.data.images import labels as split_labels, open_split
    S.setup()
    C = cache.load()
    recs, chosen = select(C)
    got = [(int(recs[i, 0]), int(recs[i, 1])) for _, i in chosen]
    if got != SELECTION:
        raise SystemExit(f"selection {got} differs from the recorded selection {SELECTION}")
    print("Figure 4 selection (episode, query):", got)
    eps = Episodes("mini", "valB_fsor_5way_1shot")
    data = []
    for tag, i in chosen:
        e, q = int(recs[i, 0]), int(recs[i, 1])
        ids = eps.indices(e)
        gq = int(ids[5 + q])
        k = int(C["nearest_1shot"][e, q])
        data.append({"tag": tag, "episode": e, "query": q, "unknown": q >= 75, "gq": gq, "gs": int(ids[k]), "raw": float(C["x_1shot"][e, q]),
                     "m": float(C["m_1shot"][e, q]), "pc": recs[i, 4], "pa": recs[i, 5],
                     "nn": [int(x) for x in C["nn_idx"][gq - N_TRAIN]], "nncos": [float(x) for x in C["nn_cos"][gq - N_TRAIN]]})
    write_data(data, names=None)
    if not draw_images:
        print("Figure 4 image panels not drawn (--no-images); selection and plotted data written")
        return data
    tr, va = open_split("mini", "train", "eval84"), open_split("mini", "val", "eval84")
    lab = np.concatenate([split_labels("mini", "train"), split_labels("mini", "val")])
    img = lambda g: tr.arr[g] if g < N_TRAIN else va.arr[g - N_TRAIN]  # noqa: E731
    names = class_names()
    cname = lambda g: names[int(lab[g])]  # noqa: E731
    row_text = {"unknown, largest increase": "Unknown,\npushed toward unknown", "known, largest decrease": "Known,\npushed toward known",
                "known, largest increase (failure)": "Known, pushed toward\nunknown (failure)"}
    fig = S.plt.figure(figsize=(S.W2, 5 * 28.5 * S.MM))
    gsp = fig.add_gridspec(5, 8, width_ratios=[1, 1, 2.45, 1, 1, 1, 1, 1], wspace=0.08, hspace=0.55)
    heads = ["Query", "Nearest-class\nsupport", "", "Nearest base images in the cohort bank (cosine)", "", "", "", ""]
    for rix, dct in enumerate(data):
        imgs = [dct["gq"], dct["gs"]] + dct["nn"]
        dct["names"] = [cname(g) for g in imgs]
        for c, g, nm in zip([0, 1, 3, 4, 5, 6, 7], imgs, dct["names"]):
            ax = fig.add_subplot(gsp[rix, c])
            ax.imshow(img(g), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(c == 0)
                if c == 0:
                    sp.set_color(S.UNKNOWN if dct["unknown"] else S.KNOWN)
                    sp.set_linewidth(1.5)
            ax.set_xlabel(nm if c < 3 else f"{nm}\n{dct['nncos'][c - 3]:.3f}", fontsize=7, labelpad=1, color="0.4", linespacing=1.0)
            if rix == 0 and c in (0, 1, 3):
                ax.set_title(heads[c], fontsize=7, pad=3, loc="left" if c == 3 else "center")
        tx = fig.add_subplot(gsp[rix, 2])
        tx.axis("off")
        tx.text(0.04, 0.5, f"{row_text[dct['tag']]}\nraw max cos {dct['raw']:.3f}\n$m_K(q)$ {dct['m']:.3f}\n"
                           f"percentile: COSF {dct['pc']:.0f}\n→ CCP-A {dct['pa']:.0f}",
                transform=tx.transAxes, va="center", ha="left", fontsize=7, linespacing=1.15)
    S.save(fig, "fig4_examples", out)
    write_data(data, names)
    return data


def write_data(data, names):
    """The plotted data (results/recorded/figures_v0/fig3_data.csv). Class names come from configs/datasets/mini_class_names.tsv and the
    recorded labels of the split (labels only, no images); without the prepared labels they are taken from the recorded data file."""
    if names is None:
        try:
            from ccp.data.images import labels as split_labels
            lab = np.concatenate([split_labels("mini", "train"), split_labels("mini", "val")])
            cn = class_names()
            for d in data:
                d["names"] = [cn[int(lab[g])] for g in [d["gq"], d["gs"]] + d["nn"]]
        except FileNotFoundError:
            from ccp import recorded
            rec = recorded.load("results/recorded/figures_v0/fig3_data.csv")
            for d, r in zip(data, rec):
                d["names"] = r["class_names_query_support_nn1-5"].split(" | ")
    S.write_csv("fig4", ["row", "rule", "valB_episode", "query_in_episode", "unknown", "query_global_index", "support_global_index", "raw_max_cos",
                         "m_K", "pct_COSF", "pct_CCPA", "nn_base_global_indices", "nn_cosines", "class_names_query_support_nn1-5"],
                [[n + 1, d["tag"], d["episode"], d["query"], int(d["unknown"]), d["gq"], d["gs"], round(d["raw"], 4), round(d["m"], 4),
                  round(d["pc"], 1), round(d["pa"], 1), " ".join(map(str, d["nn"])), " ".join(f"{x:.4f}" for x in d["nncos"]),
                  " | ".join(d["names"])] for n, d in enumerate(data)])


if __name__ == "__main__":
    main(S.GEN)
