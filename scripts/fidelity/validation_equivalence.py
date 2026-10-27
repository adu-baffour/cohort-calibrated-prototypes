"""Code-path equivalence on VALIDATION ONLY: scripts/evaluate.py against the recorded validation outputs (tier 2).

    python scripts/fidelity/validation_equivalence.py run        # GPU, sequential: every host on valA / valB (mini) and tvA / tvB (tiered)
    python scripts/fidelity/validation_equivalence.py compare    # CPU: tables in results/generated/validation_equivalence.{md,json}

Criteria: AUROC max |difference| <= 1e-4 and identical accuracy, per row, host, shot and split. References (results/recorded/validation/
and release_assets/recorded/validation/):
  exact test-pass code path: the bundle dry runs (fsor_final_dryrun_val_*.json: mini valA, tiered tvA, mini-DNPG and mini-SnaTCHer valA and
  valB), tiered_gap ccp_BGR / second_TANE (tvA / tvB, same path), snatcher.json + snatcher_raw (SnaTCHer-F tiered), dnpg_tiered_eval.json
  + dnpg_tiered_raw (DNPG tiered);
  other recorded paths: mini valB for frozen FEAT / FRN (fp16 feature caches: ccp_ablations.json, fsor_ccp.json), TANE mini valB (joint
  train+val embedding, fsor_ccp_plugin.json). These are also reproduced with --feature-dtype fp16 where that was the only difference.
No test manifest is loaded by this script.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ccp import recorded  # noqa: E402
from ccp.utils.paths import assets_root  # noqa: E402

REC = REPO / "results" / "recorded" / "validation"
GEN = REPO / "results" / "generated" / "eval"
M = lambda sh: [f"valA_fsor_5way_{sh}shot", f"valB_fsor_5way_{sh}shot"]  # noqa: E731
T = lambda sh: [f"tvA_fsor_5way_{sh}shot", f"tvB_fsor_5way_{sh}shot"]  # noqa: E731
RUNS = (
    [("frozen_feat_mini", M(1) + M(5), [])] + [(f"frn_mini_seed{s}", M(1) + M(5), []) for s in (0, 1, 2)] + [("tane_pre_mini", M(1) + M(5), [])]
    + [("frozen_feat_mini", M(1) + M(5), ["--feature-dtype", "fp16"])] + [(f"frn_mini_seed{s}", M(1) + M(5), ["--feature-dtype", "fp16"]) for s in (0, 1, 2)]
    + [("frozen_feat_tiered", T(1) + T(5), []), ("tane_pre_tiered", T(1) + T(5), [])]
    + [("snatcher_f_mini", M(1) + M(5), ["--save-scores"]), ("snatcher_f_tiered", T(1) + T(5), ["--save-scores"])]
    + [(f"dnpg_mini_seed{s}", M(1) + M(5), ["--save-scores"]) for s in (0, 1, 2)]
    + [("dnpg_tiered_seed0", T(1) + T(5), ["--save-scores"])]
)


def run():
    for host, eps, extra in RUNS:
        cmd = [sys.executable, str(REPO / "scripts" / "evaluate.py"), "--host", host, "--episodes", *eps, "--build-bank", *extra]
        print("+", " ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------------------------------------------- compare
def J(name):
    """A recorded validation record (checked against results/PROVENANCE.md, keys in the paper's names)."""
    return recorded.load(f"results/recorded/validation/{name}")


def gen(host, manifest):
    p = GEN / host / f"{manifest}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


CCP_ROWS = ("COS", "COSF", "CCP-A", "CCP-S")
ABL = {"COS": "COS", "COSF": "COSF", "CCP-A": "CCP-A (registered)", "CCP-S": "CCP-S (mean + std)"}
PREM = {"COS": "cos_noflip", "COSF": "cos_flip", "CCP-A": "Sm_img1000_flip"}


def references():
    """[(host_dir, manifest, rows, ref {row: {auroc, acc}}, reference label, path kind)]"""
    out = []
    dm, dt = J("fsor_final_dryrun_val_mini.json")["rows"], J("fsor_final_dryrun_val_tiered.json")["rows"]
    ab = J("ccp_ablations.json")["hosts"]
    prem, cf = J("fsor_ccp.json")["c2"]["grid"], J("fsor_ccp_confirm.json")["cfcheck"]["results"]
    plug = J("fsor_ccp_plugin.json")["results"]
    gap = J("fsor_tiered_gap.json")
    for sh in (1, 5):
        for host, key in [("frozen_feat_mini", "frozen_FEAT")] + [(f"frn_mini_seed{s}", f"FRN_seed{s}") for s in (0, 1, 2)] + [("tane_pre_mini", "TANE_DNPG_pretrain")]:
            out.append((host, f"valA_fsor_5way_{sh}shot", CCP_ROWS, dm[key][f"{sh}shot"], "bundle dry run (fsor_final_dryrun_val_mini.json)", "exact test-pass path"))
        for t in ("A", "B"):
            ref = {r: ab["frozen FEAT (mini)"]["rows"][f"{ABL[r]}|{t}{sh}"] for r in CCP_ROWS}
            out.append(("frozen_feat_mini_fp16compat", f"val{t}_fsor_5way_{sh}shot", CCP_ROWS, ref, "ccp_ablations.json (fp16 features, batched ablation arithmetic)",
                        "other path: ablation arithmetic (cumsum top-K means, float32 accuracies)"))
            for host, bk in [("frozen_feat_mini", "mini_frozen_FEAT")] + [(f"frn_mini_seed{s}", f"mini_FRN_seed{s}") for s in (0, 1, 2)]:
                g = prem[bk][f"{t}{sh}"]
                ref = {r: g[PREM[r]] for r in ("COS", "COSF", "CCP-A")}
                ref["CCP-S"] = {"auroc": cf[bk][f"{t}{sh}"]["closed_form"], "acc": None}
                out.append((host, f"val{t}_fsor_5way_{sh}shot", CCP_ROWS, ref, "fsor_ccp.json / fsor_ccp_confirm.json (fp16 feature cache)", "other path: fp16 features"))
                out.append((f"{host}_fp16compat", f"val{t}_fsor_5way_{sh}shot", CCP_ROWS, ref, "fsor_ccp.json / fsor_ccp_confirm.json (fp16 feature cache)", "same path (--feature-dtype fp16)"))
            out.append(("tane_pre_mini", f"val{t}_fsor_5way_{sh}shot", CCP_ROWS, plug[f"{t}{sh}"], "fsor_ccp_plugin.json (joint train+val embedding, fp32)", "other path: embedding batches"))
        out.append(("frozen_feat_tiered", f"tvA_fsor_5way_{sh}shot", CCP_ROWS, dt["frozen_FEAT_tiered"][f"{sh}shot"], "bundle dry run (fsor_final_dryrun_val_tiered.json)", "exact test-pass path"))
        out.append(("tane_pre_tiered", f"tvA_fsor_5way_{sh}shot", CCP_ROWS, dt["TANE_tiered_pretrain"][f"{sh}shot"], "bundle dry run (fsor_final_dryrun_val_tiered.json)", "exact test-pass path"))
        for t in ("A", "B"):
            out.append(("frozen_feat_tiered", f"tv{t}_fsor_5way_{sh}shot", CCP_ROWS, gap["ccp_BGR"]["results"][f"tv{t}{sh}"], "fsor_tiered_gap.json ccp_BGR", "same path"))
            out.append(("tane_pre_tiered", f"tv{t}_fsor_5way_{sh}shot", CCP_ROWS, gap["second_TANE"]["BGR"]["fsor"][f"tv{t}{sh}"], "fsor_tiered_gap.json second_TANE BGR", "same path"))
    ds = J("fsor_final_dryrun_val_mini-snatcher.json")["rows"]
    sn = J("snatcher.json")["val"]["results"]
    dd = J("fsor_final_dryrun_val_mini-dnpg.json")["rows"]
    dte = J("dnpg_tiered_eval.json")["summaries"]
    for sh in (1, 5):
        for t in ("A", "B"):
            rows = ("S", "H_fix", "K_fix", "CCP-A", "COSF", "Dist")
            out.append(("snatcher_f_mini", f"val{t}_fsor_5way_{sh}shot", rows, ds[f"{sh}shot_{t}"], "bundle dry run (fsor_final_dryrun_val_mini-snatcher.json)", "exact test-pass path"))
            out.append(("snatcher_f_tiered", f"tv{t}_fsor_5way_{sh}shot", rows, sn[f"tiered_{sh}shot_{t}"]["rows"], "snatcher.json (recorded SnaTCHer-F validation scoring)", "same path"))
            rows = ("D", "H_fix", "K_fix", "episode-wise standardization", "episode-wise standardization, COSF control", "CCP-A", "COSF")
            for s in (0, 1, 2):
                out.append((f"dnpg_mini_seed{s}", f"val{t}_fsor_5way_{sh}shot", rows, dd[f"seed{s}_{sh}shot_{t}"], "bundle dry run (fsor_final_dryrun_val_mini-dnpg.json)", "exact test-pass path"))
            out.append(("dnpg_tiered_seed0", f"tv{t}_fsor_5way_{sh}shot", rows, dte[f"seed0_max_acc_{sh}shot_{t}"]["summary"], "dnpg_tiered_eval.json", "same path"))
    return out


def raw_checks():
    """Per-query raw scores against the recorded validation score caches (release_assets/recorded/validation/)."""
    A = assets_root() / "recorded" / "validation"
    res = []
    for ds_, host, set_ in (("mini", "snatcher_f_mini", "val"), ("tiered", "snatcher_f_tiered", "tv")):
        for sh in (1, 5):
            f = np.load(A / "snatcher_raw" / f"{ds_}_{sh}shot.npz")
            for t in ("A", "B"):
                p = GEN / host / f"{set_}{t}_fsor_5way_{sh}shot.npz"
                if not p.exists():
                    continue
                g = np.load(p)
                for mine, theirs in (("S", "S"), ("CCP-A", "CCPA"), ("COSF", "COSF"), ("Dist", "Dist")):
                    res.append((host, f"{set_}{t} {sh}-shot", mine, float(np.abs(g[f"score_{mine}"] - f[f"{theirs}_{t}"]).max()), f"snatcher_raw/{ds_}_{sh}shot.npz"))
    for s in (0, 1, 2):
        for sh in (1, 5):
            f = np.load(A / "dnpg_raw" / f"seed{s}_{sh}shot.npz")
            for t in ("A", "B"):
                p = GEN / f"dnpg_mini_seed{s}" / f"val{t}_fsor_5way_{sh}shot.npz"
                if p.exists():
                    g = np.load(p)
                    for mine, theirs in (("D", "D"), ("CCP-A", "CCPA"), ("COSF", "COSF")):
                        res.append((f"dnpg_mini_seed{s}", f"val{t} {sh}-shot", mine, float(np.abs(g[f"score_{mine}"] - f[f"{theirs}_{t}"]).max()),
                                    f"dnpg_raw/seed{s}_{sh}shot.npz (CCP-A / COSF from a joint train+val embedding)"))
    for sh in (1, 5):
        f = np.load(A / "dnpg_tiered_raw" / f"seed0_max_acc_{sh}shot.npz")
        for t in ("A", "B"):
            p = GEN / "dnpg_tiered_seed0" / f"tv{t}_fsor_5way_{sh}shot.npz"
            if p.exists():
                g = np.load(p)
                for mine, theirs in (("D", "D"), ("CCP-A", "CCPA"), ("COSF", "COSF")):
                    res.append(("dnpg_tiered_seed0", f"tv{t} {sh}-shot", mine, float(np.abs(g[f"score_{mine}"] - f[f"{theirs}_{t}"]).max()), f"dnpg_tiered_raw/seed0_max_acc_{sh}shot.npz"))
    return res


def compare():
    rows, fails = [], []
    for host, man, rnames, ref, label, kind in references():
        g = gen(host, man)
        if g is None:
            rows.append((host, man, "-", None, None, label, kind, "NOT RUN"))
            continue
        da = max(abs(g["rows"][r]["auroc"] - ref[r]["auroc"]) for r in rnames)
        accs = [abs(g["rows"][r]["acc"] - ref[r]["acc"]) for r in rnames if ref[r].get("acc") is not None]
        dc = max(accs) if accs else None
        ok = da <= 1e-4 and (dc is None or dc <= 1e-9)
        verdict = "PASS" if ok else ("DIFFERENT PATH" if kind.startswith("other") else "FAIL")
        rows.append((host, man, ",".join(rnames), da, dc, label, kind, verdict))
        if verdict == "FAIL":
            fails.append((host, man))
    raw = raw_checks()
    out = REPO / "results" / "generated"
    L = ["# Validation code-path equivalence (scripts/evaluate.py vs recorded validation outputs)", "",
         "Criteria: AUROC max |difference| <= 1e-4 over the listed rows, accuracy identical. 'DIFFERENT PATH' marks references recorded with a "
         "different precision or embedding batching than the registered test-pass path (docs/protocol.md, 'Numerical precision'); they are "
         "reported, not counted as failures, and the fp16 cases are reproduced exactly by the `_fp16compat` runs.", "",
         "| Host | Manifest | Rows | max abs dAUROC | max abs dAcc | Reference | Path | Verdict |", "|---|---|---|---|---|---|---|---|"]
    for h, m, rn, da, dc, lab, kind, v in rows:
        f = lambda x: "-" if x is None else f"{x:.2e}"  # noqa: E731
        L.append(f"| {h} | {m} | {rn} | {f(da)} | {f(dc)} | {lab} | {kind} | {v} |")
    L += ["", "## Per-query raw scores vs recorded validation score caches", "", "| Host | Set | Score | max abs difference | Reference |", "|---|---|---|---|---|"]
    L += [f"| {h} | {s} | {k} | {d:.2e} | {r} |" for h, s, k, d, r in raw]
    (out / "validation_equivalence.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "validation_equivalence.json").write_text(json.dumps({"rows": rows, "raw": raw, "fails": fails}, indent=1), encoding="utf-8")
    print("\n".join(L))
    return fails


if __name__ == "__main__":
    {"run": run, "compare": compare}[sys.argv[1]]()
