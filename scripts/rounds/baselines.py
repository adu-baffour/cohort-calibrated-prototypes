"""Round (v): the nine training-free baselines on frozen FEAT (mini-ImageNet, tieredImageNet), SnaTCHer-F and DNPG (mini-ImageNet)
(paper Section 4.4, Table 5; docs/preregistration/baselines.md). GPU, validation only, one job at a time.

    python scripts/rounds/baselines.py select     # valA: every grid point; choose and freeze  -> results/generated/rounds/baselines/selection_valA.json
    python scripts/rounds/baselines.py report     # valB with the frozen choices                -> .../validation.{json,md}
    python scripts/rounds/baselines.py dry-run    # the registered bundle code path on valA + valB, compared with the recorded records

Recorded originals: results/baselines/{selection_valA,validation,dryrun_val,test}.json (test: the single pass of 2026-10-01).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rounds.common import (bundle_block, ccp_weight, generated_or_recorded, git_head, kind, now, out_dir,  # noqa: E402
                                   require_clean_code, write)

from ccp import baselines as BL, metrics  # noqa: E402
from ccp.utils.logging import get_logger  # noqa: E402

log = get_logger("baselines")
ROUND = "baselines"
TOL = 1e-4


def dataset(h):
    from scripts.rounds.common import cfg
    return cfg(h)["dataset"]


def group_of(h):
    return next(g for g, hs in BL.HOST_GROUPS.items() if h in hs)


def references(h):
    return ["host", "CCP"] + (["COS"] if kind(h) == "feat_frozen" else [])


# ------------------------------------------------------------------------------------------------------------------ select (valA)
def select():
    require_clean_code()
    t0 = time.time()
    per = {}
    for g, hosts in BL.HOST_GROUPS.items():
        for h in hosts:
            per[h] = {}
            for sh in (1, 5):
                m = BL.val_sets(dataset(h), sh)["A"]
                raw = BL.raw_scores(h, sh, [m], BL.ALL_SETTINGS, log=log.info)[m]
                w = BL.fit_weights(kind(h), raw, BL.ALL_SETTINGS)
                summ, _ = BL.summarise(BL.rows(kind(h), raw, BL.ALL_SETTINGS, w, ccp_weight(h, sh)))
                per[h][str(sh)] = {"manifest": m, "weights": w, "summary": summ}
    chosen, valA_mean = {}, {}
    for g, hosts in BL.HOST_GROUPS.items():
        means = {v: float(np.mean([per[h][str(sh)]["summary"][v]["auroc"] for h in hosts for sh in (1, 5)])) for v in BL.ALL_SETTINGS}
        valA_mean[g] = means
        chosen[g] = BL.choose(means)
    frozen = {h: {str(sh): {"weights": {v: per[h][str(sh)]["weights"][v] for v in chosen[group_of(h)].values() if v in per[h][str(sh)]["weights"]},
                            "valA": {r: per[h][str(sh)]["summary"][r] for r in references(h) + list(chosen[group_of(h)].values())}}
                  for sh in (1, 5)} for h in per}
    out = {"rule": "per host group, mean valA AUROC over both shots (DNPG: and seeds 0-2); ties to the earlier setting",
           "chosen": chosen, "valA_mean_auroc_all_settings": valA_mean, "frozen": frozen, "gpu_minutes": (time.time() - t0) / 60, "when": now(),
           "git_head": git_head()}
    write(out_dir(ROUND) / "selection_valA.json", out)
    for g, c in chosen.items():
        log.info(f"{g}: " + ", ".join(c.values()))


# ------------------------------------------------------------------------------------------------------------------ evaluation with frozen choices
def evaluate_frozen(chosen: dict, frozen: dict, sets: list[str]):
    """sets: 'A' / 'B' (validation). Returns {host: {shot: {set: {'summary', 'pe'}}}}."""
    res = {}
    for g, hosts in BL.HOST_GROUPS.items():
        settings = list(chosen[g].values())
        for h in hosts:
            res[h] = {}
            for sh in (1, 5):
                names = {s: BL.val_sets(dataset(h), sh)[s] for s in sets}
                raw = BL.raw_scores(h, sh, list(names.values()), settings, log=log.info)
                res[h][str(sh)] = {}
                for s, m in names.items():
                    summ, pe = BL.summarise(BL.rows(kind(h), raw[m], settings, frozen[h][str(sh)]["weights"], ccp_weight(h, sh)))
                    res[h][str(sh)][s] = {"manifest": m, "summary": summ, "pe": pe}
    return res


def paired_tables(chosen, res, s):
    """Paired per-episode AUROC differences vs the host alone and vs the host with CCP (DNPG: seed-averaged per episode)."""
    out = {}
    for g, hosts in BL.HOST_GROUPS.items():
        out[g] = {}
        for sh in (1, 5):
            out[g][str(sh)] = {}
            for v in chosen[g].values():
                d = {}
                for ref in ("host", "CCP"):
                    diff = np.mean([res[h][str(sh)][s]["pe"][v] - res[h][str(sh)][s]["pe"][ref] for h in hosts], axis=0)
                    d[f"minus_{ref}"] = metrics.paired(diff, np.zeros_like(diff))
                out[g][str(sh)][v] = d
    return out


def group_summary(chosen, res, s):
    """AUROC / accuracy per group, shot and row; DNPG: mean +- std over seeds (and the per-seed values)."""
    out = {}
    for g, hosts in BL.HOST_GROUPS.items():
        out[g] = {}
        for sh in (1, 5):
            out[g][str(sh)] = {}
            for r in references(hosts[0]) + list(chosen[g].values()):
                xs = [res[h][str(sh)][s]["summary"][r] for h in hosts]
                out[g][str(sh)][r] = xs[0] if len(xs) == 1 else {"auroc": metrics.mean_std([x["auroc"] for x in xs]),
                                                                  "acc": metrics.mean_std([x["acc"] for x in xs]), "per_seed": xs}
    return out


def selection():
    return generated_or_recorded(ROUND, "selection_valA.json", "results/baselines/selection_valA.json")


def report():
    require_clean_code()
    sel = selection()
    t0 = time.time()
    res = evaluate_frozen(sel["chosen"], sel["frozen"], ["B"])
    out = {"chosen": sel["chosen"], "valB": group_summary(sel["chosen"], res, "B"), "paired_valB": paired_tables(sel["chosen"], res, "B"),
           "valA_chosen": {h: {sh: sel["frozen"][h][sh]["valA"] for sh in ("1", "5")} for h in sel["frozen"]},
           "per_host_valB": {h: {sh: res[h][sh]["B"]["summary"] for sh in ("1", "5")} for h in res},
           "gpu_minutes": (time.time() - t0) / 60, "when": now(), "git_head": git_head()}
    write(out_dir(ROUND) / "validation.json", out)
    write_md(out, sel)


def fmt(x):
    if "auroc" in x and isinstance(x["auroc"], dict):
        return f"{x['auroc']['mean']:.2f} ± {x['auroc']['std']:.2f} / {x['acc']['mean']:.2f}"
    return f"{x['auroc']:.2f} ± {x['auroc_ci95']:.2f} / {x['acc']:.2f}"


def write_md(out, sel):
    L = ["# Training-free baselines on valB", "",
         "Settings chosen on valA and frozen (selection_valA.json). Cells: AUROC ± 95% CI / accuracy (DNPG: AUROC mean ± std over seeds 0-2 / "
         "mean accuracy). Paired: per-episode AUROC difference, mean ± 95% CI over 1,000 episodes (DNPG: averaged over seeds per episode). "
         "'host' = COSF (frozen FEAT), S (SnaTCHer-F) or D (DNPG); 'CCP' = CCP-A (frozen FEAT) or H_fix (trained hosts).", ""]
    for g in BL.HOST_GROUPS:
        L += [f"## {g}", "", "| Row | chosen on valA | 1-shot AUROC / acc | 5-shot AUROC / acc | − host (1 / 5) | − CCP (1 / 5) |", "|---|---|---|---|---|---|"]
        for r in ["host", "CCP"] + (["COS"] if "frozen" in g else []):
            L.append(f"| {r} | | {fmt(out['valB'][g]['1'][r])} | {fmt(out['valB'][g]['5'][r])} | | |")
        for fam, v in sel["chosen"][g].items():
            p1, p5 = out["paired_valB"][g]["1"][v], out["paired_valB"][g]["5"][v]
            pr = lambda a, b: f"{a['mean']:+.2f} ± {a['ci95']:.2f} / {b['mean']:+.2f} ± {b['ci95']:.2f}"  # noqa: E731
            L.append(f"| {fam} | `{v}` | {fmt(out['valB'][g]['1'][v])} | {fmt(out['valB'][g]['5'][v])} | {pr(p1['minus_host'], p5['minus_host'])} | "
                     f"{pr(p1['minus_CCP'], p5['minus_CCP'])} |")
        L.append("")
    (out_dir(ROUND) / "validation.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


# ------------------------------------------------------------------------------------------------------------------ dry run of the registered bundle
def dry_run():
    """The registered bundle code path (its frozen choices and weights) on valA and valB, compared with the recorded selection and validation
    records (tolerance 1e-4 on AUROC and accuracy, as registered)."""
    require_clean_code()
    b, token = bundle_block(ROUND)
    t0 = time.time()
    frozen = {h: {str(sh): {"weights": b["hosts"][h]["weights"][f"{sh}shot"]} for sh in (1, 5)} for h in b["hosts"]}
    sel, val = selection(), generated_or_recorded(ROUND, "validation.json", "results/baselines/validation.json")
    res = evaluate_frozen(b["chosen"], frozen, ["A", "B"])
    worst = {"auroc": 0.0, "acc": 0.0}
    for h in res:
        for sh in ("1", "5"):
            for s, ref in (("A", sel["frozen"][h][sh]["valA"]), ("B", val["per_host_valB"][h][sh])):
                for r, x in ref.items():
                    y = res[h][sh][s]["summary"][r]
                    worst["auroc"] = max(worst["auroc"], abs(y["auroc"] - x["auroc"]))
                    worst["acc"] = max(worst["acc"], abs(y["acc"] - x["acc"]))
    ok = worst["auroc"] <= TOL and worst["acc"] <= TOL
    out = {"bundle_token": token, "max_abs_diff_vs_validation": worst, "tolerance": TOL, "pass": ok, "gpu_minutes": (time.time() - t0) / 60,
           "when": now(), "git_head": git_head()}
    write(out_dir(ROUND) / "dryrun_val.json", out)
    log.info(f"dry run: max |dAUROC| {worst['auroc']:.2e}, max |dacc| {worst['acc']:.2e} -> {'PASS' if ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("select", "report", "dry-run"))
    a = ap.parse_args()
    {"select": select, "report": report, "dry-run": dry_run}[a.cmd]()


if __name__ == "__main__":
    main()
