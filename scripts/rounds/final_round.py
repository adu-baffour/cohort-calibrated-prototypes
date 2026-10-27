"""Round (vii), the final round (paper Tables 3-5; docs/preregistration/final_round.md): (A) CIFAR-FS, (B) mini-ImageNet -> CUB transfer,
(C) OSCR / FPR95 for the configurations of rounds (i)-(iv), (D) SCG on validation. GPU, validation only, one job at a time.

    python scripts/rounds/final_round.py build-banks      # CIFAR-FS base bank and the CUB target banks; prints the sha256 values
    python scripts/rounds/final_round.py select-cifarfs   # (A) valA: every baseline setting; choose and freeze -> results/generated/rounds/final_round/
    python scripts/rounds/final_round.py report-cifarfs   # (A) valB with the frozen choices
    python scripts/rounds/final_round.py cub-preview      # (B) CUB validation preview
    python scripts/rounds/final_round.py recorded-val     # (C) the configurations of rounds (i)-(iv) on valA + reproduction of their recorded dry runs
    python scripts/rounds/final_round.py scg-val          # (D) SCG on validation
    python scripts/rounds/final_round.py dry-run          # the registered bundle code path on validation, compared with the recorded records

Recorded originals: results/final_round/* (test.json: the single pass of 2026-10-05).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rounds.common import (REPO, bundle_block, cfg, generated_or_recorded, git_head, kind, now, out_dir, record,  # noqa: E402
                                   require_clean_code, trained, write)

from ccp import baselines as BL, metrics, scg as SCG  # noqa: E402
from ccp.rounds import final_round as FR  # noqa: E402
from ccp.utils.logging import get_logger  # noqa: E402
from ccp.utils.paths import load_yaml, sha256_file  # noqa: E402

log = get_logger("final_round")
ROUND = "final_round"
TARGET_BANKS = REPO / "configs" / "final_round_target_banks.yaml"
CIFAR_HOST = "cifarfs_rfs_seed0"
SHOTS = (1, 5)
QW_NAME = "CCP-QW/lambda=1.0/beta=0.5"
RFS_PUBLISHED = {"1": "71.5 +- 0.8", "5": "86.0 +- 0.5", "source": "Tian et al., ECCV 2020, Table 2, 'Ours-simple' (CIFAR-FS test, logistic regression)"}
CUB_GROUPS = {"frozen FEAT mini": ["frozen_feat_mini"], "SnaTCHer-F mini": ["snatcher_f_mini"],
              "DNPG mini": ["dnpg_mini_seed0", "dnpg_mini_seed1", "dnpg_mini_seed2"]}
C_GROUPS = {"frozen FEAT mini": ["frozen_feat_mini"], "FRN mini": ["frn_mini_seed0", "frn_mini_seed1", "frn_mini_seed2"],
            "TANE mini": ["tane_pre_mini"], "SnaTCHer-F mini": ["snatcher_f_mini"],
            "DNPG mini": ["dnpg_mini_seed0", "dnpg_mini_seed1", "dnpg_mini_seed2"],
            "frozen FEAT tiered": ["frozen_feat_tiered"], "TANE tiered": ["tane_pre_tiered"]}
REC_KEY = {"frozen_feat_mini": "frozen_FEAT", "frn_mini_seed0": "FRN_seed0", "frn_mini_seed1": "FRN_seed1", "frn_mini_seed2": "FRN_seed2",
           "tane_pre_mini": "TANE_DNPG_pretrain", "frozen_feat_tiered": "frozen_FEAT_tiered", "tane_pre_tiered": "TANE_tiered_pretrain"}
# SCG: name -> (part, host, m array of the configuration's SCG block, row whose AUROC gain over COSF is reported)
SCG_CELLS = {"mini": ("C", "frozen_feat_mini", "m", "CCP-A"), "tiered": ("C", "frozen_feat_tiered", "m", "CCP-A"),
             "CUB": ("B", "frozen_feat_mini", "m", "CCP-A"), "CIFAR-FS": ("A", CIFAR_HOST, "m", "CCP-A"),
             # registered amendment 2 (descriptive): CUB with the target bank at K = 1000 and at the matched K = 154
             "CUB@target": ("B", "frozen_feat_mini", "m_t1000", FR.TARGET_ROWS[0]),
             "CUB@target-K154": ("B", "frozen_feat_mini", "m_tK154", FR.TARGET_ROWS[1])}


def dataset_of(h):
    return cfg(h)["dataset"]


def manifest(part: str, h: str, sh: int, s: str) -> tuple[str, str]:
    """(dataset, manifest name) of a configuration. s: 'val' (the part's validation set), 'valA' / 'valB' (part A)."""
    if part == "A":
        return "cifarfs", f"{s}_fsor_5way_{sh}shot"
    if part == "B":
        return "cub", f"val_fsor_5way_{sh}shot"
    ds = dataset_of(h)
    return ds, f"{'valA' if ds == 'mini' else 'tvA'}_fsor_5way_{sh}shot"


# ------------------------------------------------------------------------------------------------------------------ frozen constants
def qw_weight(h, sh):
    return record("results/variants/selection_valA.json")["weights"][h][str(sh)][QW_NAME]


def ccp_weight(h, sh):
    return float(cfg(h)["fusion"]["frozen_weights"][f"{sh}shot"]["w"])


def mini_baselines(group: str, h: str, sh: int) -> dict:
    """The round-(v) choice of the host group and its frozen valA weights (mini-ImageNet; used unchanged on CUB)."""
    sel = record("results/baselines/selection_valA.json")
    w = sel["frozen"][h][str(sh)]["weights"]
    return {v: w.get(v) for v in sel["chosen"][group].values()}


def target_bank(h, sh):
    from ccp.cohort import Bank
    from ccp.utils.paths import assets_root
    e = load_yaml(TARGET_BANKS)["banks"][f"{h}|{sh}" if trained(h) else h]
    return Bank.load(assets_root() / e["file"], e["sha256"])


def cifarfs_selection():
    return generated_or_recorded(ROUND, "cifarfs_selection_valA.json", "results/final_round/cifarfs_selection_valA.json")


# ------------------------------------------------------------------------------------------------------------------ one configuration
def run_cell(part, h, sh, s, baselines: dict, with_target=False, need_scg=False, fit_evidence=False):
    """Per-episode metrics {row: {auroc, acc, oscr, fpr95}} of one (part, host, shot, set); baselines {setting: weight | None}.
    fit_evidence: fit the evidence weights on this set (CIFAR-FS valA selection only) and return them."""
    ds, m = manifest(part, h, sh, s)
    rows = ["COSF", "CCP-A", "CCP-QW"] + (["COS"] if part == "C" and not trained(h) else []) + list(baselines)
    ev = set(baselines) & BL.EVIDENCE
    raw = FR.score_host(h, sh, m, dataset=ds, rows=rows + (list(FR.TARGET_ROWS) if with_target else []), evidence=ev,
                        target_bank=target_bank(h, sh) if with_target else None, need_scg=need_scg, log=log.info)
    if trained(h):
        w = {"H_fix": ccp_weight(h, sh), "CCP-QW fused": qw_weight(h, sh), FR.TARGET_WEIGHT: ccp_weight(h, sh), **baselines}
    elif fit_evidence:
        from ccp import fusion
        w = {b: fusion.fit_weight(raw["rows"]["COSF"]["u"], raw["rows"][b]["u"])["w"] for b in ev}
    else:
        w = {b: baselines[b] for b in ev}
    pe = FR.per_episode(FR.assemble(raw, trained(h), part, w, ev, kind(h)))
    return {"pe": pe, "scg": raw["scg"], "weights": w if fit_evidence else None, "manifest": f"{ds}/{m}"}


def flat_summaries(res: dict) -> dict:
    """{'part|host|shot|set|row': summary} over {part: {set: {host: {shot: configuration}}}}."""
    return {f"{part}|{h}|{sh}|{s}|{r}": FR.summary(pe) for part, a in res.items() for s, b in a.items() for h, c in b.items()
            for sh, cell in c.items() for r, pe in cell["pe"].items()}


def run_part(part, s, chosen_a=None):
    """All configurations of a part on set s ('val'; part A: 'valA' or 'valB')."""
    out = {}
    if part == "A":
        sel = chosen_a or cifarfs_selection()
        out[CIFAR_HOST] = {sh: run_cell("A", CIFAR_HOST, sh, s, {v: sel["weights"][str(sh)].get(v) for v in sel["chosen"].values()},
                                        need_scg=s == "valA") for sh in SHOTS}
    elif part == "B":
        for g, hs in CUB_GROUPS.items():
            for h in hs:
                out[h] = {sh: run_cell("B", h, sh, s, mini_baselines(g, h, sh), with_target=True, need_scg=h == "frozen_feat_mini") for sh in SHOTS}
    else:
        for g, hs in C_GROUPS.items():
            for h in hs:
                out[h] = {sh: run_cell("C", h, sh, s, {}, need_scg=h in ("frozen_feat_mini", "frozen_feat_tiered")) for sh in SHOTS}
    return out


# ------------------------------------------------------------------------------------------------------------------ reporting helpers
def group_table(groups: dict, cells: dict, ref_rows: tuple = ("host",)) -> dict:
    """Per group and shot: summaries per row (multi-host groups: mean / std over hosts and per host) and paired differences vs each ref."""
    out = {}
    for g, hs in groups.items():
        hs = [h for h in hs if h in cells]
        if not hs:
            continue
        out[g] = {}
        for sh in SHOTS:
            rows = list(cells[hs[0]][sh]["pe"])
            t = {"rows": {}, "paired": {}}
            for r in rows:
                xs = [FR.summary(cells[h][sh]["pe"][r]) for h in hs]
                t["rows"][r] = xs[0] if len(xs) == 1 else {k: metrics.mean_std([x[k] for x in xs]) for k in ("auroc", "acc", "oscr", "fpr95")} | {"per_host": xs}
                for ref in ref_rows:
                    if ref in cells[hs[0]][sh]["pe"] and r != ref:
                        t["paired"].setdefault(ref, {})[r] = FR.paired_vs([cells[h][sh]["pe"][r] for h in hs], [cells[h][sh]["pe"][ref] for h in hs])
            out[g][str(sh)] = t
    return out


def fmt(x, k):
    v = x[k]
    if isinstance(v, dict):
        return f"{v['mean']:.2f} ± {v['std']:.2f}"
    return f"{v:.2f} ± {x[k + '_ci95']:.2f}"


def md_table(title: str, table: dict, ref: str, note: str) -> list[str]:
    L = [f"## {title}", "", note, ""]
    for g, t in table.items():
        for sh in ("1", "5"):
            L += [f"### {g}, {sh}-shot", "", "| Row | AUROC | acc | OSCR | FPR95 | AUROC − ref | OSCR − ref | FPR95 − ref |", "|---|---|---|---|---|---|---|---|"]
            for r, x in t[sh]["rows"].items():
                p = t[sh]["paired"].get(ref, {}).get(r)
                pr = (lambda k: f"{p[k]['mean']:+.2f} ± {p[k]['ci95']:.2f} ({p[k]['frac_gain'] * 100:.0f}%)") if p else (lambda k: "")
                L.append(f"| {r} | {fmt(x, 'auroc')} | {fmt(x, 'acc')} | {fmt(x, 'oscr')} | {fmt(x, 'fpr95')} | {pr('auroc')} | {pr('oscr')} | {pr('fpr95')} |")
            L.append("")
    return L


# ------------------------------------------------------------------------------------------------------------------ banks
def build_banks():
    """CIFAR-FS base bank (needs the pinned checkpoint) and the CUB target banks; writes their sha256 (compare with the pinned values)."""
    require_clean_code()
    from ccp.hosts import build_host
    from ccp.pipeline import bank_path, load_bank, make_target_bank
    from ccp.utils.paths import assets_root
    host = build_host(cfg(CIFAR_HOST))
    load_bank(host, allow_build=True, log=log.info)
    out = {"cifarfs_base": {"file": cfg(CIFAR_HOST)["bank"]["file"], "sha256": sha256_file(bank_path(host))}}
    banks = {}
    for g, hs in CUB_GROUPS.items():
        for h in hs:
            for sh in (SHOTS if trained(h) else (None,)):
                hh = build_host(cfg(h), shot=sh)
                hh.shot = sh
                rel = f"banks/cub_target/{h}{f'_{sh}shot' if sh else ''}.npz"
                p = assets_root() / rel
                if not p.exists():
                    make_target_bank(hh, "cub", log=log.info).save(p)
                banks[f"{h}|{sh}" if sh else h] = {"file": rel, "sha256": sha256_file(p)}
    out["cub_target"] = banks
    write(out_dir(ROUND) / "banks_built.json", {**out, "when": now(), "git_head": git_head()})


# ------------------------------------------------------------------------------------------------------------------ validation steps
def select_cifarfs():
    require_clean_code()
    t0 = time.time()
    per, weights = {}, {}
    for sh in SHOTS:
        c = run_cell("A", CIFAR_HOST, sh, "valA", {v: None for v in BL.ALL_SETTINGS}, need_scg=True, fit_evidence=True)
        per[sh] = {r: FR.summary(pe) for r, pe in c["pe"].items()}
        weights[str(sh)] = c["weights"]
    means = {v: float(np.mean([per[sh][v]["auroc"] for sh in SHOTS])) for v in BL.ALL_SETTINGS}
    chosen = BL.choose(means)
    sanity = {str(sh): {"COSF_closed_set_acc_valA": per[sh]["COSF"]["acc"], "CCP-A_acc_valA": per[sh]["CCP-A"]["acc"]} for sh in SHOTS}
    out = {"rule": "the round-(v) selection rule applied to CIFAR-FS valA: mean AUROC over both shots, ties to the earlier setting",
           "chosen": chosen, "valA_mean_auroc_all_settings": means, "weights": weights, "valA": {str(sh): per[sh] for sh in SHOTS},
           "sanity_backbone": {**sanity, "rfs_published_test_5way": RFS_PUBLISHED,
                               "note": "ours: nearest-prototype cosine (COSF) on the 75 known queries of the valA episodes; RFS: logistic "
                                       "regression on the test split. Sanity check only."},
           "gpu_minutes": (time.time() - t0) / 60, "when": now(), "git_head": git_head()}
    write(out_dir(ROUND) / "cifarfs_selection_valA.json", out)
    log.info(f"CIFAR-FS selection: {chosen}")


def report_cifarfs():
    require_clean_code()
    t0 = time.time()
    cells = run_part("A", "valB")
    tab = group_table({"CIFAR-FS": [CIFAR_HOST]}, cells, ("host", "CCP-A"))
    out = {"chosen": cifarfs_selection()["chosen"], "table": tab, "summaries": flat_summaries({"A": {"valB": cells}}),
           "gpu_minutes": (time.time() - t0) / 60, "when": now(), "git_head": git_head()}
    write(out_dir(ROUND) / "cifarfs_valB.json", out)
    L = ["# CIFAR-FS valB (final round, part A)", "", "Baseline settings chosen on valA. Preview only; every registered row went to test.", ""]
    L += md_table("CIFAR-FS valB", tab, "host", "host = COSF. Paired: per-episode difference vs host, mean ± 95% CI (% of episodes improved; "
                  "FPR95: % with a lower value).")
    (out_dir(ROUND) / "cifarfs_valB.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def cub_preview():
    require_clean_code()
    t0 = time.time()
    cells = run_part("B", "val")
    tab = group_table(CUB_GROUPS, cells, ("host",))
    out = {"table": tab, "summaries": flat_summaries({"B": {"val": cells}}), "gpu_minutes": (time.time() - t0) / 60, "when": now(), "git_head": git_head()}
    write(out_dir(ROUND) / "cub_val_preview.json", out)
    L = ["# mini-ImageNet -> CUB, validation preview (final round, part B)", "", "Every constant fixed on mini-ImageNet. The target-bank rows use "
         "a cohort bank of CUB base images: a separate analysis (unlabeled target-domain images), not the registered transfer result.", ""]
    L += md_table("CUB val", tab, "host", "host = COSF (frozen FEAT), S (SnaTCHer-F), D (DNPG; mean ± std over seeds).")
    (out_dir(ROUND) / "cub_val_preview.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def recorded_reference(h, sh) -> dict:
    """{row: {auroc, acc}} of the recorded validation dry run (valA) of a configuration of rounds (i)-(iv)."""
    k = kind(h)
    if k in ("snatcher_f", "dnpg"):
        f = "fsor_final_dryrun_val_mini-snatcher.json" if k == "snatcher_f" else "fsor_final_dryrun_val_mini-dnpg.json"
        key = (f"{sh}shot_" if k == "snatcher_f" else f"seed{h[-1]}_{sh}shot_") + "A"
        rows = record(f"results/recorded/validation/{f}")["rows"][key]
        return {"host": rows["S" if k == "snatcher_f" else "D"], "H_fix": rows["H_fix"], "CCP-A": rows["CCP-A"], "COSF": rows["COSF"]}
    tiered = dataset_of(h) == "tiered"
    rows = record(f"results/recorded/validation/fsor_final_dryrun_val_{'tiered' if tiered else 'mini'}.json")["rows"][REC_KEY[h]][f"{sh}shot"]
    return {"host": rows["COS"], "COSF": rows["COSF"], "CCP-A": rows["CCP-A"]}


def reproduction(cells: dict) -> dict:
    worst, n = 0.0, 0
    for h, c in cells.items():
        for sh, cell in c.items():
            for r, x in recorded_reference(h, sh).items():
                s = FR.summary(cell["pe"][r])
                worst = max(worst, abs(s["auroc"] - x["auroc"]), abs(s["acc"] - x["acc"]))
                n += 1
    return {"compared_rows": n, "max_abs_diff": worst, "reproduced": worst == 0.0}


def recorded_val():
    require_clean_code()
    t0 = time.time()
    cells = run_part("C", "val")
    rep = reproduction(cells)
    tab = group_table(C_GROUPS, cells, ("host",))
    out = {"reproduction_of_recorded_dry_runs": rep, "table": tab, "summaries": flat_summaries({"C": {"val": cells}}),
           "gpu_minutes": (time.time() - t0) / 60, "when": now(), "git_head": git_head()}
    write(out_dir(ROUND) / "recorded_valA.json", out)
    L = ["# The configurations of rounds (i)-(iv) with OSCR / FPR95 on valA (final round, part C)", "",
         f"Reproduction of the recorded validation dry runs (AUROC and accuracy of {rep['compared_rows']} rows): "
         f"{'exact' if rep['reproduced'] else 'FAILED'} (max |difference| {rep['max_abs_diff']:.1e}).", ""]
    L += md_table("valA", tab, "host", "host = COS (frozen hosts) or S / D (trained hosts).")
    (out_dir(ROUND) / "recorded_valA.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def scg_block(parts: dict) -> dict:
    """SCG per dataset and shot from the configurations that carry SCG arrays, with the CCP-A - COSF AUROC gain on the same episodes."""
    out = {}
    for name, (part, h, mkey, row) in SCG_CELLS.items():
        for sh in SHOTS:
            cell = parts[part][h][sh]
            d = SCG.scg_per_shot(cell["scg"]["x"], cell["scg"][mkey])
            gain = float(np.mean(cell["pe"][row]["auroc"] - cell["pe"]["COSF"]["auroc"]))
            out[f"{name}|{sh}"] = {"std_gap": d["std_gap"], "raw_gap": d["raw_gap"], "ccpa_minus_cosf_auroc": gain, "gain_row": row,
                                   "bank": "target" if mkey != "m" else "base", "manifest": cell["manifest"], "bins": d["bins"]}
        out[f"{name}|mean_over_shots"] = {"std_gap": float(np.mean([out[f"{name}|{sh}"]["std_gap"] for sh in SHOTS]))}
    return out


def scg_val():
    require_clean_code()
    t0 = time.time()
    parts = {"A": {CIFAR_HOST: {sh: run_cell("A", CIFAR_HOST, sh, "valA", {}, need_scg=True) for sh in SHOTS}},
             "B": {"frozen_feat_mini": {sh: run_cell("B", "frozen_feat_mini", sh, "val", {}, with_target=True, need_scg=True) for sh in SHOTS}},
             "C": {h: {sh: run_cell("C", h, sh, "val", {}, need_scg=True) for sh in SHOTS} for h in ("frozen_feat_mini", "frozen_feat_tiered")}}
    scg = scg_block(parts)
    write(out_dir(ROUND) / "scg_val.json", {"scg": scg, "used_for_decisions": False, "gpu_minutes": (time.time() - t0) / 60, "when": now(),
                                           "git_head": git_head()})
    log.info("SCG validation: " + ", ".join(f"{k} {v['std_gap']:+.3f}" for k, v in scg.items()))


# ------------------------------------------------------------------------------------------------------------------ dry run of the registered bundle
def dry_run():
    """The registered bundle code path on the validation sets; every summary and SCG value must equal the recorded records (tolerance 0)."""
    require_clean_code()
    b, token = bundle_block(ROUND)
    rec = {"A": record("results/final_round/cifarfs_valB.json"), "B": record("results/final_round/cub_val_preview.json"),
           "C": record("results/final_round/recorded_valA.json"), "D": record("results/final_round/d1_val.json")}
    selA = record("results/final_round/cifarfs_selection_valA.json")
    if b["cifarfs_chosen"] != selA["chosen"] or b["cifarfs_weights"] != selA["weights"]:
        raise SystemExit("the bundle's frozen CIFAR-FS choices differ from the recorded selection")
    t0 = time.time()
    parts = {"A": run_part("A", "valB", selA), "B": run_part("B", "val"), "C": run_part("C", "val")}
    partA_valA = {CIFAR_HOST: {sh: run_cell("A", CIFAR_HOST, sh, "valA", {v: selA["weights"][str(sh)].get(v) for v in selA["chosen"].values()},
                                            need_scg=True) for sh in SHOTS}}
    got = {**flat_summaries({"A": {"valB": parts["A"]}}), **flat_summaries({"B": {"val": parts["B"]}}), **flat_summaries({"C": {"val": parts["C"]}})}
    ref = {**rec["A"]["summaries"], **rec["B"]["summaries"], **rec["C"]["summaries"]}
    worst, n = 0.0, 0
    for k, x in ref.items():
        for m in ("auroc", "acc", "oscr", "fpr95"):
            worst = max(worst, abs(got[k][m] - x[m]))
            n += 1
    for sh in SHOTS:
        for r, pe in partA_valA[CIFAR_HOST][sh]["pe"].items():
            s, x = FR.summary(pe), selA["valA"][str(sh)][r]
            worst = max(worst, *(abs(s[m] - x[m]) for m in ("auroc", "acc", "oscr", "fpr95")))
            n += 4
    scg = scg_block({"A": partA_valA, "B": parts["B"], "C": parts["C"]})
    for k, x in rec["D"]["scg"].items():
        for m in ("std_gap", "raw_gap", "ccpa_minus_cosf_auroc"):
            if m in x:
                worst = max(worst, abs(scg[k][m] - x[m]))
                n += 1
    ok = worst == 0.0
    out = {"bundle_token": token, "compared_values": n, "max_abs_diff_vs_validation": worst, "tolerance": 0.0, "pass": ok,
           "gpu_minutes": (time.time() - t0) / 60, "when": now(), "git_head": git_head()}
    write(out_dir(ROUND) / "dryrun_val.json", out)
    log.info(f"dry run: {n} values, max |diff| {worst:.2e} -> {'PASS' if ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cmds = {"build-banks": build_banks, "select-cifarfs": select_cifarfs, "report-cifarfs": report_cifarfs, "cub-preview": cub_preview,
            "recorded-val": recorded_val, "scg-val": scg_val, "dry-run": dry_run}
    ap.add_argument("cmd", choices=list(cmds))
    cmds[ap.parse_args().cmd]()


if __name__ == "__main__":
    main()
