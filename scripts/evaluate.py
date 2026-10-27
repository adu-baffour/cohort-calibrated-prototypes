"""Evaluate a host with CCP on fixed episode manifests (the registered test-pass code path).

    python scripts/evaluate.py --host frozen_feat_mini --episodes valA_fsor_5way_1shot valB_fsor_5way_1shot
    python scripts/evaluate.py --host dnpg_mini_seed0 --episodes valA_fsor_5way_1shot valB_fsor_5way_1shot --save-scores
    python scripts/evaluate.py --host snatcher_f_mini --episodes valA_fsor_5way_5shot --weights fit:valA_fsor_5way_5shot

Rows. Encoders (frozen FEAT, FRN, TANE): COS, COSF, CCP-A, CCP-S (and any --scores from the registry, e.g. the ablations).
Trained hosts: the host's own score (SnaTCHer-F: S and Dist; DNPG: D), H_fix = own + w * CCP-A and K_fix = own + w2 * COSF with the
FROZEN weights of configs/hosts/<host>.yaml (default) or weights fitted here on a named validation-A manifest (--weights fit:<name>);
DNPG also reports the episode-wise standardization rows (transductive; Section 3.4).
Accuracy: each encoder score's own argmax; SnaTCHer-F: FEAT's classifier for S / H_fix / K_fix / Dist; DNPG: DNPG's classifier for
every row (as registered).

Outputs: results/generated/eval/<host>[_<checkpoint>]/<manifest>.json (+ .npz with per-query scores and per-episode metrics with
--save-scores). --out must lie under results/generated/; nothing else can be written.

RE-EVALUATION OF TEST EPISODES (tier 2; not run by the authors for this release):

    python scripts/evaluate.py --host frozen_feat_mini --episodes test_fsor_5way_1shot test_fsor_5way_5shot --allow-test

  The paper's test numbers are the recorded single passes of the eight registered rounds (results/recorded/, results/*/test.json). Scoring
  the test manifests again is a NEW evaluation: it is not how the paper's numbers were produced and does not replace them. Test manifests
  refuse to load without --allow-test; with it, a warning is printed, the output goes to results/generated/test_reevaluation/<host>/ (never
  to a recorded path) and every output file is marked "new_evaluation": true.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccp import fusion, metrics  # noqa: E402
from ccp.episodes import Episodes  # noqa: E402
from ccp.hosts import build_host, load_host_config  # noqa: E402
from ccp.pipeline import Target, load_bank, score_ccp, score_host_own  # noqa: E402
from ccp.utils.io import save_npz  # noqa: E402
from ccp.utils.logging import get_logger  # noqa: E402
from ccp.utils.paths import REPO  # noqa: E402

log = get_logger("evaluate")
ENCODER_SCORES = ["COS", "COSF", "CCP-A", "CCP-S"]
GENERATED = REPO / "results" / "generated"
TEST_WARNING = ("WARNING: re-scoring TEST episodes. This is a new evaluation. The paper's test numbers are the recorded single passes of the "
                "eight registered rounds (results/recorded/, results/*/test.json); this run does not reproduce how they were produced and "
                "does not replace them. Output: results/generated/test_reevaluation/.")


def open_manifests(dataset: str, names: list[str], allow_test: bool, root: Path | None = None) -> list[Episodes]:
    """The manifests, sha256-checked. A test manifest raises SplitAccessError unless allow_test; with it, TEST_WARNING is logged."""
    kw = {} if root is None else {"root": Path(root)}
    eps = [Episodes(dataset, m, allow_test=allow_test, **kw) for m in names]
    if any(e.split == "test" for e in eps):
        log.warning(TEST_WARNING)
        print(TEST_WARNING, file=sys.stderr, flush=True)
    return eps


def resolve_out_dir(out: Path | None, default_name: str, has_test: bool, generated: Path = GENERATED) -> Path:
    """The output folder: under results/generated/ only (eval/ or, for test manifests, test_reevaluation/). Any other --out is refused, so
    no recorded file can be written."""
    base = generated / ("test_reevaluation" if has_test else "eval")
    d = Path(out).resolve() if out else (base / default_name).resolve()
    if not d.is_relative_to(generated.resolve()):
        raise SystemExit(f"--out {d} is outside {generated}; this script writes only under results/generated/")
    if has_test and not d.is_relative_to((generated / "test_reevaluation").resolve()):
        raise SystemExit("a run with test manifests writes only under results/generated/test_reevaluation/")
    return d


def split_of(eps: Episodes) -> str:
    return "test" if eps.split == "test" else "val"


def evaluate(host_name: str, manifests: list[str], scores: list[str] | None = None, checkpoint: str = "max_acc", weights: str = "frozen",
             allow_test: bool = False, save_scores: bool = False, out_dir: Path | None = None, build_bank_if_missing: bool = False,
             max_episodes: int = 0, feature_dtype: str | None = None) -> dict:
    cfg = load_host_config(host_name)
    if feature_dtype and feature_dtype != cfg["feature_dtype"]:
        # compatibility mode: reproduce results that were computed from features stored in another precision (docs/protocol.md,
        # "Numerical precision"); the bank goes to a separate file and has no pinned hash
        import copy
        cfg = copy.deepcopy(cfg)
        cfg["feature_dtype"] = feature_dtype
        for e in ([cfg["bank"]] if "file" in cfg["bank"] else list(cfg["bank"].values())):
            e["file"] = e["file"].replace(".npz", f"_{feature_dtype}compat.npz")
            e["sha256"] = None
        host_name_out = f"{host_name}_{feature_dtype}compat"
    else:
        host_name_out = host_name
    eps_all = open_manifests(cfg["dataset"], manifests, allow_test)
    if max_episodes:
        eps_all = [e.subset(max_episodes) for e in eps_all]
    shots = sorted({e.shot for e in eps_all})
    splits = sorted({split_of(e) for e in eps_all})
    trained = cfg["kind"] in ("snatcher_f", "dnpg")
    has_test = "test" in splits
    out_dir = resolve_out_dir(out_dir, host_name_out + (f"_{checkpoint}" if cfg["kind"] == "dnpg" and checkpoint != "max_acc" else ""), has_test)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for shot in shots:
        t0 = time.time()
        kw = {"checkpoint": checkpoint} if cfg["kind"] == "dnpg" else {}
        host = build_host(cfg, shot=shot if trained else None, **kw)
        host.shot = shot
        bank = load_bank(host, allow_build=build_bank_if_missing, log=log.info)
        names = scores or (["CCP-A", "COSF"] if trained else ENCODER_SCORES)
        for split in splits:
            eps_here = [e for e in eps_all if e.shot == shot and split_of(e) == split]
            if not eps_here:
                continue
            target = Target(host, split, bank, need_pooled=cfg["kind"] == "snatcher_f")
            raw = {}
            for eps in eps_here:
                r = score_ccp(target, eps, names)
                own, own_acc = score_host_own(host, target, eps) if trained else ({}, None)
                raw[eps.name] = (r, own, own_acc)
            # weights: frozen from the config, or fitted on a validation-A manifest scored in this call
            w = w2 = None
            if trained:
                if weights == "frozen":
                    table = cfg["fusion"]["frozen_weights" + ("_last" if checkpoint == "last" else "")]
                    w, w2 = float(table[f"{shot}shot"]["w"]), float(table[f"{shot}shot"]["w2"])
                    wsrc = f"frozen ({cfg['fusion']['provenance']})"
                else:
                    ref = weights.split(":", 1)[1]
                    if ref not in raw:
                        raise SystemExit(f"--weights fit:{ref}: that manifest must be evaluated in the same call")
                    ref_eps = next(e for e in eps_here if e.name == ref)
                    if ref_eps.split == "test" or ref_eps.meta.get("set") not in ("A", "tvA"):
                        raise SystemExit("weights may only be fitted on a validation-A manifest")
                    rr, own_r, _ = raw[ref]
                    key = "S" if cfg["kind"] == "snatcher_f" else "D"
                    w = fusion.fit_weight(own_r[key], rr["CCP-A"][0])["w"]
                    w2 = fusion.fit_weight(own_r[key], rr["COSF"][0])["w"]
                    wsrc = f"fitted on {ref}"
            for eps in eps_here:
                r, own, own_acc = raw[eps.name]
                rows, accs = {}, {}
                for n, (S, A) in r.items():
                    rows[n], accs[n] = S, A
                if cfg["kind"] == "snatcher_f":
                    S = own["S"]
                    rows.update({"S": S, "H_fix": fusion.h_fix(S, r["CCP-A"][0], w), "K_fix": fusion.h_fix(S, r["COSF"][0], w2), "Dist": own["Dist"]})
                    accs.update({k: own_acc for k in ("S", "H_fix", "K_fix", "Dist")})
                elif cfg["kind"] == "dnpg":
                    rows = fusion.fused_rows(own["D"], r["CCP-A"][0], r["COSF"][0], w, w2)
                    accs = {k: own_acc for k in rows}
                pe = {k: metrics.episode_auroc(v) for k, v in rows.items()}
                summary = {k: metrics.summarise(pe[k], accs[k]) for k in rows}
                res = {"new_evaluation": eps.split == "test", "host": host_name, "shot": shot, "manifest": eps.name, "manifest_sha256": eps.sha256,
                       "split": eps.split, "n_episodes": eps.n,
                       "rows": summary, "host_info": host.describe(), "bank_sha256": cfg["bank"].get(f"{shot}shot", cfg["bank"]).get("sha256")}
                if trained:
                    own_key = "S" if cfg["kind"] == "snatcher_f" else "D"
                    res.update({"weights": {"w": w, "w2": w2, "source": wsrc}, "checkpoint": checkpoint if cfg["kind"] == "dnpg" else f"{shot}shot",
                                "paired": {f"Hfix_minus_{own_key}": metrics.paired(pe["H_fix"], pe[own_key]),
                                           "Hfix_minus_Kfix": metrics.paired(pe["H_fix"], pe["K_fix"])}})
                results[eps.name] = res
                (out_dir / f"{eps.name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
                if save_scores:
                    arrs = {f"score_{k}": v for k, v in rows.items()}
                    arrs.update({f"auroc_{k}": v for k, v in pe.items()})
                    arrs.update({f"acc_{k}": np.asarray(v) for k, v in accs.items()})
                    save_npz(out_dir / f"{eps.name}.npz", **arrs)
                log.info(f"{host_name} {eps.name}: " + ", ".join(f"{k} {v['auroc']:.2f}/{v['acc']:.2f}" for k, v in summary.items()))
            del target
        log.info(f"{host_name} shot {shot}: {(time.time() - t0) / 60:.1f} min")
        del host
        import torch
        torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True)
    ap.add_argument("--episodes", nargs="+", required=True, help="manifest names under manifests/<dataset>/")
    ap.add_argument("--scores", nargs="*", default=None, help="cohort scores from the registry (default: the registered rows)")
    ap.add_argument("--checkpoint", default="max_acc", choices=("max_acc", "last"), help="DNPG only")
    ap.add_argument("--weights", default="frozen", help="frozen (configs) or fit:<validation-A manifest in this call>")
    ap.add_argument("--allow-test", action="store_true",
                    help="load TEST manifests: a NEW evaluation, not the paper's recorded single passes (output: results/generated/test_reevaluation/)")
    ap.add_argument("--save-scores", action="store_true")
    ap.add_argument("--build-bank", action="store_true", help="build the bank if its file is missing")
    ap.add_argument("--max-episodes", type=int, default=0, help="smoke tests only")
    ap.add_argument("--feature-dtype", choices=("fp16", "fp32"), default=None,
                    help="override the stored feature precision (compatibility with results computed from fp16 feature caches)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    evaluate(a.host, a.episodes, a.scores, a.checkpoint, a.weights, a.allow_test, a.save_scores, a.out, a.build_bank, a.max_episodes, a.feature_dtype)


if __name__ == "__main__":
    main()
