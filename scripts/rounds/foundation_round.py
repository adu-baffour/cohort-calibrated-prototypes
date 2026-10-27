"""Round (viii), the foundation-model round (paper Section 5.5, Table 6; docs/preregistration/foundation_round.md): frozen CLIP ViT-B/16 and
DINOv2 ViT-S/14, ViT-B/14, ViT-L/14 on mini-ImageNet, tieredImageNet, CIFAR-FS and CUB. GPU, validation only, one job at a time.

    python scripts/rounds/foundation_round.py download-weights   # the four weight files from their authors (about 2.2 GB; ask first)
    python scripts/rounds/foundation_round.py prepare-cub224     # the CUB eval224 view (input-resolution analysis)
    python scripts/rounds/foundation_round.py build-banks        # the 18 banks; one is rebuilt in memory and compared byte for byte
    python scripts/rounds/foundation_round.py sanity             # bf16 vs fp32 features, weight sha256, NaN in the banks
    python scripts/rounds/foundation_round.py sanity-acc         # COSF closed-set accuracy on valA
    python scripts/rounds/foundation_round.py select-baselines   # valA: baseline settings, CCP rows and SCG of the 32 configurations
    python scripts/rounds/foundation_round.py report-valB        # valB with the frozen choices and the analysis configurations
    python scripts/rounds/foundation_round.py dry-run            # the registered bundle code path on validation: 4,384 values vs the records

Outputs: results/generated/rounds/foundation_round/. Recorded originals: results/foundation_round/* (test.json: the single pass of
2026-10-06; test_hypotheses.json: H1, H2, H3 computed from the recorded files).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rounds.common import (REPO, bundle_block, fmt, generated_or_recorded, git_head, now, out_dir, require_clean_code, write)

from ccp import baselines as BL, scg as SCG, stats  # noqa: E402
from ccp.hosts import load_host_config  # noqa: E402
from ccp.rounds import final_round as FR, foundation as FD  # noqa: E402
from ccp.utils.logging import get_logger  # noqa: E402
from ccp.utils.paths import sha256_file  # noqa: E402

log = get_logger("foundation")
ROUND = "foundation_round"
SHOTS = (1, 5)
BACKBONES = ("clip_b16", "dinov2_s14", "dinov2_b14", "dinov2_l14")
PRIMARY = ("clip_b16", "dinov2_b14")
DATASETS = ("mini", "tiered", "cifarfs", "cub")
SETS = {"mini": {"A": "valA", "B": "valB"}, "tiered": {"A": "tvA", "B": "tvB"}, "cifarfs": {"A": "valA", "B": "valB"}, "cub": {"A": "val", "B": "valB"}}
WEIGHT_FILES = {
    "clip_b16": ("third_party/clip/ViT-B-16.pt",
                 "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
                 "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"),
    "dinov2_s14": ("third_party/dinov2/dinov2_vits14_pretrain.pth", "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth", None),
    "dinov2_b14": ("third_party/dinov2/dinov2_vitb14_pretrain.pth", "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth", None),
    "dinov2_l14": ("third_party/dinov2/dinov2_vitl14_pretrain.pth", "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", None),
}
BASE_ROWS = ["COS", "COSF", "CCP-A", "CCP-QW"]


def host(ds, b):
    return f"{ds}_{b}"


def primary_hosts():
    return [host(ds, b) for b in BACKBONES for ds in DATASETS]


def analysis_cells():
    """(name, host, dataset): CUB with the mini-ImageNet base bank (every backbone) and the CUB eval224 view (CLIP, DINOv2 ViT-B)."""
    out = [(f"cub@minibank_{b}", host("mini", b), "cub") for b in BACKBONES]
    out += [(f"cub_eval224_{b}", f"cub_{b}_eval224", "cub") for b in PRIMARY]
    return out


def mname(ds, s, sh):
    return f"{SETS[ds][s]}_fsor_5way_{sh}shot"


def cell(ds, b, sh):
    return f"{ds}|{b}|{sh}"


# ------------------------------------------------------------------------------------------------------------------ setup
def download_weights():
    """The four weight files to the first CCP_CKPT root (about 2.2 GB); CLIP's sha256 is checked against the one in its URL."""
    require_clean_code()
    import urllib.request

    from ccp.utils.paths import ckpt_roots
    root = ckpt_roots()[0]
    out = {}
    for b, (rel, url, expect) in WEIGHT_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            tmp = p.with_suffix(p.suffix + ".part")
            log.info(f"downloading {url}")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(p)
        h = sha256_file(p)
        if expect and h != expect:
            raise SystemExit(f"{rel}: sha256 {h} != the sha256 in CLIP's URL {expect}")
        out[b] = {"path": rel, "url": url, "bytes": p.stat().st_size, "sha256": h}
    write(out_dir(ROUND) / "weights.json", {"weights": out, "when": now(), "git_head": git_head()})


def prepare_cub224():
    require_clean_code()
    subprocess.run([sys.executable, str(REPO / "scripts" / "prepare_data" / "cub.py"), "--views", "eval224"], check=True)
    from ccp.utils.paths import data_root
    d = data_root() / "cub" / "prepared"
    out = {s: {"file": f"cub/prepared/eval224_{s}.npy", "sha256": sha256_file(d / f"eval224_{s}.npy")} for s in ("train", "val", "test")}
    out["index"] = {"file": "cub/prepared/index.json", "sha256": sha256_file(d / "index.json")}
    write(out_dir(ROUND) / "cub_eval224.json", {"arrays": out, "when": now(), "git_head": git_head()})


def build_banks():
    require_clean_code()
    import torch

    from ccp.hosts import build_host
    from ccp.pipeline import bank_path, load_bank, make_bank
    t0 = time.time()
    out = {}
    for n in primary_hosts() + [f"cub_{b}_eval224" for b in PRIMARY]:
        t = time.time()
        h = build_host(load_host_config(n))
        load_bank(h, allow_build=True, log=log.info)
        out[n] = {"file": load_host_config(n)["bank"]["file"], "sha256": sha256_file(bank_path(h)), "seconds": time.time() - t}
        del h
    h = build_host(load_host_config("cub_dinov2_s14"))          # determinism: rebuild one bank in memory and compare bytes
    tmp = out_dir(ROUND) / "_rebuild_check.npz"
    same = make_bank(h, log=lambda *a: None).save(tmp) == out["cub_dinov2_s14"]["sha256"]
    tmp.unlink()
    torch.cuda.empty_cache()
    write(out_dir(ROUND) / "banks_built.json", {"banks": out, "rebuild_check_cub_dinov2_s14_byte_identical": same,
                                               "gpu_minutes": (time.time() - t0) / 60, "when": now(), "git_head": git_head()})


def sanity():
    """bf16 (registered) vs fp32 features on 500 base images per backbone; weight sha256; NaN in every bank."""
    require_clean_code()
    import torch

    from ccp.cohort import Bank
    from ccp.hosts import build_host
    from ccp.pipeline import host_images
    from ccp.utils.paths import assets_root
    out = {"bf16_vs_fp32": {}, "bank_nan": {}, "weights_sha256_match": {}}
    for b in BACKBONES:
        h = build_host(load_host_config(host("mini", b)))           # loads the weights after their sha256 check
        out["weights_sha256_match"][b] = True
        x = torch.from_numpy(np.array(host_images(h, "train")[:500])).cuda()
        with torch.no_grad():
            a = h.gap_pair(x)[0].float()
            f32 = h.encoder_forward(x).flatten(1).float()
        cos = torch.nn.functional.cosine_similarity(a, f32, dim=1)
        out["bf16_vs_fp32"][b] = {"n_images": 500, "images": "the first 500 mini base images (eval84)", "cosine_mean": float(cos.mean()),
                                  "cosine_min": float(cos.min())}
        del h
        torch.cuda.empty_cache()
    for n in primary_hosts() + [f"cub_{b}_eval224" for b in PRIMARY]:
        e = load_host_config(n)["bank"]
        bank = Bank.load(assets_root() / e["file"], e["sha256"])
        out["bank_nan"][n] = bool(not (np.isfinite(bank.raw_noflip).all() and np.isfinite(bank.raw_flip).all()))
    write(out_dir(ROUND) / "sanity.json", {**out, "when": now(), "git_head": git_head()})


def sanity_acc():
    """5-way closed-set COSF accuracy on valA per host and shot (COSF only)."""
    require_clean_code()
    from ccp import metrics
    t0 = time.time()
    out = {}
    for b in BACKBONES:
        for ds in DATASETS:
            raw = FD.score_frozen(host(ds, b), [(sh, ds, mname(ds, "A", sh)) for sh in SHOTS], ["COSF"], need_scg=False, log=log.info)
            for sh in SHOTS:
                acc = raw[(sh, ds, mname(ds, "A", sh))]["rows"]["COSF"]["acc"]
                out[cell(ds, b, sh)] = {"cosf_acc_valA": float(acc.mean()), "ci95": metrics.ci95(acc), "manifest": mname(ds, "A", sh)}
    write(out_dir(ROUND) / "sanity_acc.json", {"cosf_closed_set_acc_valA": out, "gpu_minutes": (time.time() - t0) / 60, "when": now(),
                                              "git_head": git_head()})


# ------------------------------------------------------------------------------------------------------------------ scoring helpers
def run_host(n, ds, sets: list[str], baselines: dict, fit=False):
    """Per (set, shot): per-episode metrics of the reported rows, SCG, fitted evidence weights (fit=True, valA only)."""
    jobs = [(sh, ds, mname(ds, s, sh)) for s in sets for sh in SHOTS]
    rows = BASE_ROWS + list(baselines)
    ev = set(baselines) & BL.EVIDENCE
    raw = FD.score_frozen(n, jobs, rows, ev, need_scg=True, log=log.info)
    out = {}
    for s in sets:
        for sh in SHOTS:
            r = raw[(sh, ds, mname(ds, s, sh))]
            if fit:
                from ccp import fusion
                w = {v: fusion.fit_weight(r["rows"]["COSF"]["u"], r["rows"][v]["u"])["w"] for v in ev}
            else:
                w = {v: baselines[v][str(sh)] for v in ev}
            pe = FR.per_episode(FR.assemble(r, False, "A", w, ev))
            g = SCG.scg_per_shot(r["scg"]["x"], r["scg"]["m"])
            out[(s, sh)] = {"pe": pe, "weights": w, "scg": {"std_gap": g["std_gap"], "raw_gap": g["raw_gap"]}}
    return out


def summaries(res: dict, prefix: str) -> dict:
    return {f"{prefix}|{s}|{sh}|{r}": FR.summary(pe) for (s, sh), c in res.items() for r, pe in c["pe"].items()}


def chosen_baselines(sel, n):
    """{chosen setting: {shot: frozen evidence weight or None}} of host n."""
    return {v: {sh: sel["weights"][n][sh].get(v) for sh in ("1", "5")} for v in sel["chosen"][n].values()}


def selection():
    return generated_or_recorded(ROUND, "selection_valA.json", "results/foundation_round/selection_valA.json")


# ------------------------------------------------------------------------------------------------------------------ validation steps
def select_baselines():
    require_clean_code()
    t0 = time.time()
    out = {"chosen": {}, "weights": {}, "valA": {}, "valA_mean_auroc": {}, "scg_valA": {}, "cosf_acc_valA": {}}
    for b in BACKBONES:
        for ds in DATASETS:
            n = host(ds, b)
            res = run_host(n, ds, ["A"], {v: None for v in BL.ALL_SETTINGS}, fit=True)
            means = {v: float(np.mean([FR.summary(res[("A", sh)]["pe"][v])["auroc"] for sh in SHOTS])) for v in BL.ALL_SETTINGS}
            out["chosen"][n] = BL.choose(means)
            out["weights"][n] = {str(sh): res[("A", sh)]["weights"] for sh in SHOTS}
            out["valA"][n] = {str(sh): {r: FR.summary(pe) for r, pe in res[("A", sh)]["pe"].items()} for sh in SHOTS}
            out["valA_mean_auroc"][n] = means
            for sh in SHOTS:
                out["scg_valA"][cell(ds, b, sh)] = {**res[("A", sh)]["scg"], "gain_ccpa_minus_cosf": float(np.mean(
                    res[("A", sh)]["pe"]["CCP-A"]["auroc"] - res[("A", sh)]["pe"]["COSF"]["auroc"]))}
                out["cosf_acc_valA"][cell(ds, b, sh)] = out["valA"][n][str(sh)]["COSF"]["acc"]
    out.update({"rule": "the round-(v) selection rule per host on valA (CUB val): mean AUROC over both shots, ties to the earlier setting",
                "gpu_minutes": (time.time() - t0) / 60, "when": now(), "git_head": git_head()})
    write(out_dir(ROUND) / "selection_valA.json", out)


def evaluate_B(sel):
    """valB: the 16 hosts with the chosen rows, and the analysis configurations (CUB with the mini-ImageNet bank; CUB eval224) on valA and valB."""
    res, ana = {}, {}
    for b in BACKBONES:
        for ds in DATASETS:
            n = host(ds, b)
            res[n] = run_host(n, ds, ["B"], chosen_baselines(sel, n))
    for name, n, ds in analysis_cells():
        ana[name] = run_host(n, ds, ["A", "B"], {})
    return res, ana


def paired_cosf(res_cell) -> dict:
    return {r: FR.paired_vs([pe], [res_cell["pe"]["COSF"]]) for r, pe in res_cell["pe"].items() if r != "COSF"}


def headroom_cosf(res_cell) -> dict:
    """The registered descriptive headroom ratio of every row over COSF, per episode."""
    return {r: stats.headroom_paired(pe["auroc"], res_cell["pe"]["COSF"]["auroc"]) for r, pe in res_cell["pe"].items() if r != "COSF"}


def report_valB():
    require_clean_code()
    sel = selection()
    t0 = time.time()
    res, ana = evaluate_B(sel)
    out = {"summaries": {**{k: v for n, r in res.items() for k, v in summaries(r, n).items()},
                         **{k: v for n, r in ana.items() for k, v in summaries(r, n).items()}},
           "paired_vs_cosf": {f"{n}|{sh}": {r: x["auroc"] for r, x in paired_cosf(r_[("B", sh)]).items()} for n, r_ in res.items() for sh in SHOTS},
           "scg_valB": {f"{n}|{sh}": r_[("B", sh)]["scg"] for n, r_ in res.items() for sh in SHOTS},
           "headroom_vs_cosf": {**{f"{n}|B|{sh}": headroom_cosf(r_[("B", sh)]) for n, r_ in res.items() for sh in SHOTS},
                                **{f"{n}|{s}|{sh}": headroom_cosf(c) for n, r_ in ana.items() for (s, sh), c in r_.items()}},
           "gpu_minutes": (time.time() - t0) / 60, "when": now(), "git_head": git_head()}
    write(out_dir(ROUND) / "validation_valB.json", out)
    L = ["# Foundation-model round: valB", "", "Settings chosen on valA. Preview only: every registered configuration went to test. Paired AUROC "
         "difference vs COSF, mean ± 95% CI (% of episodes improved); headroom ratio (descriptive): per-episode (AUROC_row − AUROC_COSF) / "
         "(100 − AUROC_COSF), mean ± 95% CI.", "",
         "| Host | shot | COSF AUROC | CCP-A − COSF | CCP-A headroom ratio | CCP-QW − COSF | best inductive baseline − COSF |", "|---|---|---|---|---|---|---|"]
    for n, r_ in res.items():
        for sh in SHOTS:
            p = paired_cosf(r_[("B", sh)])
            base = [v for v in p if v in BL.ALL_SETTINGS and v not in BL.TRANSDUCTIVE]
            bb = max(base, key=lambda v: p[v]["auroc"]["mean"])
            f = lambda x: f"{x['auroc']['mean']:+.2f} ± {x['auroc']['ci95']:.2f} ({x['auroc']['frac_gain'] * 100:.0f}%)"  # noqa: E731
            hr = headroom_cosf(r_[("B", sh)])["CCP-A"]
            L.append(f"| {n} | {sh} | {fmt(FR.summary(r_[('B', sh)]['pe']['COSF']), 'auroc')} | {f(p['CCP-A'])} | {hr['mean']:+.3f} ± {hr['ci95']:.3f} | "
                     f"{f(p['CCP-QW'])} | {bb}: {f(p[bb])} |")
    (out_dir(ROUND) / "validation_valB.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


# ------------------------------------------------------------------------------------------------------------------ dry run of the registered bundle
def dry_run():
    """The registered bundle code path on validation: the valA rows and SCG of the 32 configurations and every valB summary must equal the
    recorded records exactly (4,384 values, tolerance 0)."""
    require_clean_code()
    b, token = bundle_block(ROUND)
    sel = selection()
    for n, x in b["hosts"].items():
        if "chosen" in x and (x["chosen"] != sel["chosen"][n] or x["evidence_weights"] != sel["weights"][n]):
            raise SystemExit(f"{n}: frozen choices differ from the registered bundle")
    vb = generated_or_recorded(ROUND, "validation_valB.json", "results/foundation_round/validation_valB.json")
    t0 = time.time()
    worst, n_cmp = 0.0, 0
    for bb in BACKBONES:
        for ds in DATASETS:
            n = host(ds, bb)
            ra = run_host(n, ds, ["A"], chosen_baselines(sel, n))
            for (s, sh), c in ra.items():
                for r, pe in c["pe"].items():
                    x, y = FR.summary(pe), sel["valA"][n][str(sh)][r]
                    worst = max(worst, *(abs(x[m] - y[m]) for m in ("auroc", "acc", "oscr", "fpr95")))
                    n_cmp += 4
                d = sel["scg_valA"][cell(ds, bb, sh)]
                worst = max(worst, abs(c["scg"]["std_gap"] - d["std_gap"]), abs(c["scg"]["raw_gap"] - d["raw_gap"]))
                n_cmp += 2
    res, ana = evaluate_B(sel)
    got = {**{k: v for n, r in res.items() for k, v in summaries(r, n).items()}, **{k: v for n, r in ana.items() for k, v in summaries(r, n).items()}}
    for k, x in vb["summaries"].items():
        worst = max(worst, *(abs(got[k][m] - x[m]) for m in ("auroc", "acc", "oscr", "fpr95")))
        n_cmp += 4
    ok = worst == 0.0
    out = {"bundle_token": token, "compared_values": n_cmp, "max_abs_diff_vs_validation": worst, "tolerance": 0.0, "pass": ok,
           "gpu_minutes": (time.time() - t0) / 60, "when": now(), "git_head": git_head()}
    write(out_dir(ROUND) / "dryrun_val.json", out)
    log.info(f"dry run: {n_cmp} values, max |diff| {worst:.2e} -> {'PASS' if ok else 'FAIL'} ({out['gpu_minutes']:.1f} min)")
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cmds = {"download-weights": download_weights, "prepare-cub224": prepare_cub224, "build-banks": build_banks, "sanity": sanity,
            "sanity-acc": sanity_acc, "select-baselines": select_baselines, "report-valB": report_valB, "dry-run": dry_run}
    ap.add_argument("cmd", choices=list(cmds))
    cmds[ap.parse_args().cmd]()


if __name__ == "__main__":
    main()
