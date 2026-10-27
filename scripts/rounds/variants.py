"""Round (vi): the four registered method variants (paper Section 5.6, Table 7(b); docs/prereg_variants.md). GPU, validation only.

    python scripts/rounds/variants.py select     # valA: every grid point; choose and freeze   -> results/generated/rounds/variants/selection_valA.json
    python scripts/rounds/variants.py report     # valB with the frozen choices, criteria (i)-(iii) -> .../validation_valB.{json,md}
    python scripts/rounds/variants.py valc       # valC with the frozen choices, criterion (iv), selection -> .../validation_valC.{json,md}
    python scripts/rounds/variants.py scg        # SCG of frozen FEAT on valA (mini-ImageNet, tieredImageNet) -> .../scg/{scg.json, scg.csv, scg_panel.png}
    python scripts/rounds/variants.py dry-run    # the registered bundle code path on valA / valB / valC, compared with the recorded records

Recorded originals: results/variants/* (test.json: the single pass of 2026-10-02). The valC manifests were generated once by the registered
generator (manifests/{mini,tiered}/valC_fsor_5way_{1,5}shot.json; tests/test_manifests.py rebuilds them from their recorded seeds).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rounds.common import (bundle_block, ccp_weight, cfg, generated_or_recorded, git_head, now, out_dir,  # noqa: E402
                                   require_clean_code, trained, write)

from ccp import metrics, scg as SCG, variants as V  # noqa: E402
from ccp.utils.logging import get_logger  # noqa: E402

log = get_logger("variants")
ROUND = "variants"


def key(cell):
    return f"{cell[0]}|{cell[1]}"


def summary(pe):
    return metrics.summarise(pe[0], pe[1])


def selection():
    return generated_or_recorded(ROUND, "selection_valA.json", "results/variants/selection_valA.json")


# ------------------------------------------------------------------------------------------------------------------ select (valA)
def select():
    require_clean_code()
    t0 = time.time()
    per, weights = {}, {}
    names = V.SCORED + V.FUSION_WEIGHT_FACTOR
    for g, hosts in V.HOST_GROUPS.items():
        for h in hosts:
            per[h], weights[h] = {}, {}
            for sh in V.SHOTS:
                m = V.val_sets(cfg(h)["dataset"], sh)["A"]
                raw = V.raw_scores(h, sh, [m], log=log.info)[m]
                w = V.fit_weights(raw, V.SCORED) if trained(h) else {}
                pe = V.per_episode(V.rows(raw, trained(h), names, w, ccp_weight(h, sh)))
                per[h][sh] = {r: summary(x) for r, x in pe.items()}
                weights[h][sh] = w
    cell = {}
    for v in ["reference"] + names:
        cell[v] = {}
        for g, hosts in V.HOST_GROUPS.items():
            for sh in V.SHOTS:
                if v in per[hosts[0]][sh]:
                    cell[v][(g, sh)] = float(np.mean([per[h][sh][v]["auroc"] for h in hosts]))
    qw, qw_means = V.select_global(cell, V.CCP_QW)
    sc, sc_means = V.select_global(cell, V.SHARED_COHORT)
    factor = V.select_factor(cell)
    chosen = {"CCP-QW": qw, "full AS-norm": "full AS-norm", "shared-cohort correction": sc, "fusion-weight factor": factor}
    sens = {f: {key(c): V.choose({v: cell[v][c] for v in vs if c in cell[v]}, vs) for c in V.CELLS if any(c in cell[v] for v in vs)}
            for f, vs in V.FAMILIES.items() if f != "full AS-norm"}
    out = {"rule": "CCP-QW / shared-cohort correction: one global setting by the mean valA AUROC over the 8 configurations (DNPG = mean over "
                   "seeds 0-2); fusion-weight factor per trained host by the mean over its two shots; ties to the earlier entry of the registered order",
           "chosen": chosen, "qw_is_ccp_a": qw == V.CCP_QW_NULL,
           "qw_status": "not selected (the selected setting is CCP-A itself)" if qw == V.CCP_QW_NULL else "selected",
           "valA_mean_auroc_over_cells": {"CCP-QW": qw_means, "shared-cohort correction": sc_means,
                                          "fusion-weight factor": {g: {v: float(np.mean([cell[v][(g, sh)] for sh in V.SHOTS])) for v in V.FUSION_WEIGHT_FACTOR}
                                                                   for g in V.TRAINED_GROUPS}},
           "valA_cell_auroc": {v: {key(c): a for c, a in cv.items()} for v, cv in cell.items()},
           "sensitivity_per_cell_optimum (reported only, never used)": sens,
           "weights": {h: {str(sh): {v: weights[h][sh][v] for v in (qw, "full AS-norm", sc)} for sh in V.SHOTS} for h in per if trained(h)},
           "valA_per_host": {h: {str(sh): per[h][sh] for sh in V.SHOTS} for h in per},
           "gpu_minutes": (time.time() - t0) / 60, "when": now(), "git_head": git_head()}
    write(out_dir(ROUND) / "selection_valA.json", out)
    log.info(f"selection: {chosen}; CCP-QW {out['qw_status']}")


# ------------------------------------------------------------------------------------------------------------------ evaluation with frozen choices
def group_variants(chosen, g):
    vs = [chosen["CCP-QW"], chosen["full AS-norm"], chosen["shared-cohort correction"]]
    return vs + ([chosen["fusion-weight factor"][g]] if g in V.TRAINED_GROUPS else [])


def evaluate(sel, s):
    """s = 'A', 'B' or 'C' (validation). Per configuration and family: summaries, paired AUROC change vs the reference, accuracy change."""
    chosen = sel["chosen"]
    pe = {}
    for g, hosts in V.HOST_GROUPS.items():
        vs = group_variants(chosen, g)
        for h in hosts:
            for sh in V.SHOTS:
                m = V.val_sets(cfg(h)["dataset"], sh)[s]
                raw = V.raw_scores(h, sh, [m], log=log.info)[m]
                w = sel["weights"].get(h, {}).get(str(sh), {})
                pe[(h, sh)] = V.per_episode(V.rows(raw, trained(h), vs, w, ccp_weight(h, sh)))
    cells = {}
    for g, hosts in V.HOST_GROUPS.items():
        vs = group_variants(chosen, g)
        for sh in V.SHOTS:
            c = {"reference": {"per_host": {h: summary(pe[(h, sh)]["reference"]) for h in hosts}}}
            for v in vs:
                p = V.cell_paired([pe[(h, sh)][v][0] for h in hosts], [pe[(h, sh)]["reference"][0] for h in hosts])
                acc = float(np.mean([pe[(h, sh)][v][1].mean() - pe[(h, sh)]["reference"][1].mean() for h in hosts]))
                c[V.family_of(v)] = {"variant": v, "paired_auroc_vs_reference": p, "acc_change": acc,
                                     "per_host": {h: summary(pe[(h, sh)][v]) for h in hosts}}
            cells[key((g, sh))] = c
    return cells, pe


def gains(cells, fam, cell_list):
    return ({c: cells[key(c)][fam]["paired_auroc_vs_reference"]["mean"] for c in cell_list},
            {c: cells[key(c)][fam]["acc_change"] for c in cell_list})


def cells_of(f):
    return V.FACTOR_CELLS if f == "fusion-weight factor" else V.CELLS


def report():
    require_clean_code()
    sel = selection()
    t0 = time.time()
    cells, _ = evaluate(sel, "B")
    crit = {f: V.criteria_valB(*gains(cells, f, cells_of(f)), cells_of(f)) for f in V.FAMILIES}
    cand = V.candidate(crit, {"CCP-QW": not sel["qw_is_ccp_a"]})
    out = {"chosen": sel["chosen"], "qw_status": sel["qw_status"], "cells_valB": cells, "criteria_valB": crit, "candidate": cand,
           "gpu_minutes": (time.time() - t0) / 60, "when": now(), "git_head": git_head()}
    write(out_dir(ROUND) / "validation_valB.json", out)
    write_md(out, "B")


def valc():
    require_clean_code()
    sel = selection()
    vb = generated_or_recorded(ROUND, "validation_valB.json", "results/variants/validation_valB.json")
    t0 = time.time()
    cells, _ = evaluate(sel, "C")
    iv = {f: V.criterion_valC(gains(cells, f, cells_of(f))[0], cells_of(f)) for f in V.FAMILIES}
    selected = V.promotion(vb["candidate"], iv)
    out = {"chosen": sel["chosen"], "candidate_from_valB": vb["candidate"], "cells_valC": cells, "criterion_iv_valC": iv, "selected": selected,
           "decision": (f"{selected} selected (registered extension; a test bundle is drafted for approval)" if selected else
                        "no variant selected: no test pass for this round; reported as a validation result"),
           "gpu_minutes": (time.time() - t0) / 60, "when": now(), "git_head": git_head()}
    write(out_dir(ROUND) / "validation_valC.json", out)
    write_md(out, "C")


def write_md(out, s):
    cells = out[f"cells_val{s}"]
    title = {"B": "valB: criteria (i)-(iii)", "C": "valC (new episodes): criterion (iv) and selection"}[s]
    L = [f"# Method variants on {title}", "",
         "Settings chosen on valA and frozen. Reference = CCP-A (frozen FEAT) or H_fix = D + w CCP-A (trained hosts). Cells: paired per-episode "
         "AUROC change vs the reference, mean ± 95% CI over 1,000 episodes (DNPG: averaged over seeds 0-2 per episode); accuracy change in "
         "points (frozen FEAT; trained hosts keep their classifier).", "", f"Chosen on valA: {json.dumps(out['chosen'])}.", ""]
    if s == "B":
        L += [f"CCP-QW status: {out['qw_status']}.", ""]
    L += ["| Configuration | " + " | ".join(V.FAMILIES) + " |", "|---|" + "---|" * len(V.FAMILIES)]
    for c in V.CELLS:
        row = []
        for f in V.FAMILIES:
            x = cells[key(c)].get(f)
            if x is None:
                row.append("n/a")
            else:
                p = x["paired_auroc_vs_reference"]
                acc = f", acc {x['acc_change']:+.2f}" if c[0] in V.FROZEN_GROUPS else ""
                row.append(f"{p['mean']:+.2f} ± {p['ci95']:.2f}{acc}")
        L.append(f"| {c[0]} {c[1]}-shot | " + " | ".join(row) + " |")
    L.append("")
    if s == "B":
        L += ["| Family | mean gain, (i) >= +0.30 | min configuration, (ii) >= -0.20 | min frozen acc change, (iii) >= -0.20 | (i)-(iii) |",
              "|---|---|---|---|---|"]
        for f, g in out["criteria_valB"].items():
            a = "-" if g["min_frozen_acc_change"] is None else f"{g['min_frozen_acc_change']:+.2f}"
            yn = lambda b: "met" if b else "not met"  # noqa: E731
            L.append(f"| {f} | {g['mean_gain']:+.3f} {yn(g['(i)'])} | {g['min_cell_gain']:+.3f} {yn(g['(ii)'])} | {a} {yn(g['(iii)'])} | "
                     f"{yn(g['pass_i_iii'])} |")
        L += ["", f"Candidate (highest valB mean gain among the families meeting (i)-(iii)): {out['candidate']}."]
    else:
        L += ["| Family | mean gain (>= +0.15) | min configuration (>= -0.20) | (iv) |", "|---|---|---|---|"]
        for f, g in out["criterion_iv_valC"].items():
            L.append(f"| {f} | {g['mean_gain']:+.3f} | {g['min_cell_gain']:+.3f} | {'met' if g['(iv)'] else 'not met'} |")
        L += ["", f"Candidate from valB: {out['candidate_from_valB']}. Decision: {out['decision']}."]
    (out_dir(ROUND) / f"validation_val{s}.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


# ------------------------------------------------------------------------------------------------------------------ SCG (developed in this round)
def scg():
    require_clean_code()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = time.time()
    d = out_dir(ROUND) / "scg"
    d.mkdir(parents=True, exist_ok=True)
    res, rows = {}, []
    for ds, h in (("mini", "frozen_feat_mini"), ("tiered", "frozen_feat_tiered")):
        res[ds] = {}
        for sh in V.SHOTS:
            m = V.val_sets(ds, sh)["A"]
            r = V.scg_raw(h, m, log=log.info)
            bins = SCG.scg_bins(r["x"], r["m"], r["y"])
            res[ds][str(sh)] = {"manifest": m, "bins": bins, "std_gap": SCG.scg_gap(bins), "raw_gap": SCG.scg_gap(bins, False)}
            rows += [[ds, sh, b["bin"], b["x_lo"], b["x_hi"], b["x_mean"], b["n_known"], b["n_unknown"], b["m_known"], b["m_unknown"], b["gap"],
                      b["sd_pooled"], b["std_gap"]] for b in bins]
        for g in ("std_gap", "raw_gap"):
            res[ds][f"{g}_mean_over_shots"] = float(np.mean([res[ds][str(sh)][g] for sh in V.SHOTS]))
    gm, gt = res["mini"]["std_gap_mean_over_shots"], res["tiered"]["std_gap_mean_over_shots"]
    out = {"registered_reading": "standardized SCG (mean over the 10 bins, then over the two shots): tieredImageNet <= 1/3 of mini-ImageNet -> "
                                 "'the base cohort carries little signal on tiered'; otherwise 'the signal exists; the weighting or fusion is the "
                                 "limitation'. Raw gaps are reported, not read.",
           "std_gap_mini": gm, "std_gap_tiered": gt, "ratio_tiered_over_mini": gt / gm if gm else None, "reading": SCG.scg_reading(gm, gt),
           "raw_gap_mini": res["mini"]["raw_gap_mean_over_shots"], "raw_gap_tiered": res["tiered"]["raw_gap_mean_over_shots"],
           "per_dataset": res, "used_for_decisions": False, "gpu_minutes": (time.time() - t0) / 60, "when": now(), "git_head": git_head()}
    write(d / "scg.json", out)
    hdr = "dataset,shot,bin,x_lo,x_hi,x_mean,n_known,n_unknown,m_known,m_unknown,gap,sd_pooled,std_gap"
    (d / "scg.csv").write_text("\n".join([hdr] + [",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) for r in rows]) + "\n",
                               encoding="utf-8", newline="\n")
    fig, ax = plt.subplots(2, 2, figsize=(7, 5), sharey="row")
    for i, ds in enumerate(("mini", "tiered")):
        for j, sh in enumerate(V.SHOTS):
            b = res[ds][str(sh)]["bins"]
            x = [r["x_mean"] for r in b]
            ax[i, j].plot(x, [r["m_known"] for r in b], "o-", ms=3, lw=1, label="known")
            ax[i, j].plot(x, [r["m_unknown"] for r in b], "^--", ms=3, lw=1, label="unknown")
            ax[i, j].set_title(f"{ds} valA, {sh}-shot: SCG {res[ds][str(sh)]['std_gap']:+.3f} (raw {res[ds][str(sh)]['raw_gap']:+.4f})", fontsize=8)
            ax[i, j].set_xlabel("raw max cosine (bin mean)", fontsize=7)
            ax[i, j].set_ylabel("mean m_K(q)", fontsize=7)
            ax[i, j].tick_params(labelsize=6)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"SCG (diagnostic): mean m_K(q) per bin of the raw max cosine. Reading: {out['reading']}", fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "scg_panel.png", dpi=200)
    log.info(f"SCG mini {gm:+.3f}, tiered {gt:+.3f} -> {out['reading']}")


# ------------------------------------------------------------------------------------------------------------------ dry run of the registered bundle
def flat(cells):
    """{(configuration, family or reference, host): summary} of a cells dict."""
    return {(c, f, h): sm for c, x in cells.items() for f, y in x.items() for h, sm in y["per_host"].items()}


def dry_run():
    """The registered bundle code path on valA, valB and valC; every summary must equal the recorded records exactly (tolerance 0)."""
    require_clean_code()
    b, token = bundle_block(ROUND)
    sel = {"chosen": b["chosen"], "weights": {h: {sh[0]: w for sh, w in x["variant_weights"].items()} for h, x in b["hosts"].items() if x["variant_weights"]}}
    rec_sel = selection()
    vb = generated_or_recorded(ROUND, "validation_valB.json", "results/variants/validation_valB.json")
    vc = generated_or_recorded(ROUND, "validation_valC.json", "results/variants/validation_valC.json")
    t0 = time.time()
    worst, n = {"auroc": 0.0, "acc": 0.0}, 0
    for s in ("A", "B", "C"):
        cells, pe = evaluate(sel, s)
        if s == "A":
            got = {(h, sh, r): summary(x) for (h, sh), rows_ in pe.items() for r, x in rows_.items()}
            ref = {k: rec_sel["valA_per_host"][k[0]][str(k[1])][k[2]] for k in got}
        else:
            got, ref = flat(cells), flat((vb if s == "B" else vc)[f"cells_val{s}"])
        for k, x in ref.items():
            n += 1
            worst["auroc"] = max(worst["auroc"], abs(got[k]["auroc"] - x["auroc"]))
            worst["acc"] = max(worst["acc"], abs(got[k]["acc"] - x["acc"]))
    ok = worst["auroc"] == 0.0 and worst["acc"] == 0.0
    out = {"bundle_token": token, "compared_summaries": n, "max_abs_diff_vs_validation": worst, "tolerance": 0.0, "pass": ok,
           "gpu_minutes": (time.time() - t0) / 60, "when": now(), "git_head": git_head()}
    write(out_dir(ROUND) / "dryrun_val.json", out)
    log.info(f"dry run: {n} summaries, max |dAUROC| {worst['auroc']:.2e}, max |dacc| {worst['acc']:.2e} -> {'PASS' if ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("select", "report", "valc", "scg", "dry-run"))
    a = ap.parse_args()
    {"select": select, "report": report, "valc": valc, "scg": scg, "dry-run": dry_run}[a.cmd]()


if __name__ == "__main__":
    main()
