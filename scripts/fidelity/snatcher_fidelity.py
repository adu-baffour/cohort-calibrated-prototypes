"""SnaTCHer-F adapter (ccp/hosts/snatcher_f.py) vs SnaTCHer's own FSLTrainer.evaluate_test (third_party/SnaTCHer @ 335c424, unchanged).
Validation only: the first 50 mini valA_fsor episodes per shot. GPU.

    python scripts/fidelity/snatcher_fidelity.py [--episodes 50]            # -> results/generated/fidelity_snatcher.json
    python scripts/fidelity/snatcher_fidelity.py --record                   # -> results/checks/snatcher_fidelity_end_to_end.json (2.0.0 check)

Their path: evaluate_test on our episodes laid out in their query-major 10-class order (unknown-class support slots filled with dummy images
that their code drops); per-query values captured by wrapping calc_auroc / count_acc. Our path: (i) end to end, our encoder batch of 5K + 150
images; (ii) the same scoring code on encoder outputs of their exact batch, isolating the code path. Recorded result (docs/results/implementation_checks.md): max |diff| 7.6e-6
on the same encoder batch at both shots; end to end at 5-shot up to 3.2e-3 (cuDNN batch composition). Per comparison the output also holds
the mean AUROC of S over the episodes (ours and theirs), their difference and the largest per-episode AUROC difference.
--record writes the result as a check of this release (results/checks/, listed with its sha256 in results/PROVENANCE.md, "2.0.0 checks";
python scripts/provenance.py add-check <file>), with the git HEAD, the sha256 of this script and the software versions; the 50 episodes are
the registered default and --record refuses any other number.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ccp.data.images import open_split  # noqa: E402
from ccp.episodes import Episodes  # noqa: E402
from ccp.hosts import build_host  # noqa: E402
from ccp.hosts.base import checked_ckpt, normalize_u8  # noqa: E402
from ccp.utils.paths import third_party_root  # noqa: E402


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--record", action="store_true", help="write results/checks/snatcher_fidelity_end_to_end.json")
    a = ap.parse_args()
    if a.record and a.episodes != 50:
        raise SystemExit("--record uses the 50 registered fidelity episodes")
    n_eps = a.episodes
    sys.path.insert(0, str(third_party_root() / "SnaTCHer"))
    sys.path.append(str(REPO / "third_party" / "shims"))  # tensorboardX stand-in (imported by their logger, unused here)
    import model.trainer.fsl_trainer_SnaTCHerF as FT
    from model.models.SnaTCHerF import SnaTCHerF
    val = open_split("mini", "val", "eval84")
    img = lambda ids: np.array(val.arr[np.asarray(ids) - 38400])  # noqa: E731
    out = {}
    for sh in (1, 5):
        host = build_host("snatcher_f_mini", shot=sh)
        ck = checked_ckpt(host.cfg["checkpoints"][f"{sh}shot"])
        eps = Episodes("mini", f"valA_fsor_5way_{sh}shot")
        batches = []
        for e in range(n_eps):
            rows = [[eps.support[e][c][r] for c in range(5)] + [eps.unk_query[e][c][r] for c in range(5)] for r in range(sh)]
            rows += [[eps.query[e][c][r] for c in range(5)] + [eps.unk_query[e][c][r] for c in range(5)] for r in range(15)]
            ids = np.array(rows).reshape(-1)
            batches.append((normalize_u8(torch.from_numpy(img(ids)).cuda()).cpu(), torch.zeros(len(ids), dtype=torch.long)))
        args = argparse.Namespace(backbone_class="Res12", closed_way=5, open_way=5, way=10, eval_way=10, shot=sh, eval_shot=sh, query=15,
                                  eval_query=15, save_path=str(ck.parent), weight_name=ck.name)
        model = SnaTCHerF(args).cuda()
        rec = {"auroc": [], "logits": []}
        orig_auc, orig_acc = FT.calc_auroc, FT.count_acc
        FT.calc_auroc = lambda k, u: (rec["auroc"].append((np.array(k), np.array(u))), orig_auc(k, u))[1]
        FT.count_acc = lambda lg, lb: (rec["logits"].append(lg.detach().cpu().numpy()), orig_acc(lg, lb))[1]
        fake = types.SimpleNamespace(args=args, emb_dim=640, model=model, para_model=model, test_loader=batches)
        try:
            FT.FSLTrainer.evaluate_test(fake)
        finally:
            FT.calc_auroc, FT.count_acc = orig_auc, orig_acc
        their_acc = fake.record[:n_eps, 1] * 100
        kmap = np.array([r * 5 + c for c in range(5) for r in range(15)])
        res = {}
        from ccp.metrics import auroc
        for mode in ("end_to_end", "same_encoder_batch"):
            md = {"S": 0.0, "logits": 0.0, "Dist": 0.0, "acc": 0.0}
            au_ours, au_theirs = [], []
            for e in range(n_eps):
                if mode == "end_to_end":
                    E = host.pooled(img(eps.indices(e)))
                    Es, Eq = E[:5 * sh].view(5, sh, -1), E[5 * sh:]
                else:
                    with torch.no_grad():
                        Et = host.enc(batches[e][0].cuda()).mean((2, 3)).view(sh + 15, 10, -1)
                    Es = Et[:sh, :5].permute(1, 0, 2)
                    Eq = torch.cat([Et[sh:, :5].permute(1, 0, 2).reshape(75, -1), Et[sh:, 5:].permute(1, 0, 2).reshape(75, -1)])
                logits, S, Dist, acc = host.episode(Es.contiguous(), Eq.contiguous())
                logits, S, Dist = logits.cpu().numpy(), S.cpu().numpy(), Dist.cpu().numpy()
                tk_S, tu_S = rec["auroc"][3 * e + 2]
                tk_D, tu_D = rec["auroc"][3 * e + 1]
                md["logits"] = max(md["logits"], float(np.abs(logits[:75] - rec["logits"][e][kmap]).max()))
                md["S"] = max(md["S"], float(np.abs(S[:75] - tk_S[kmap]).max()), float(np.abs(S[75:] - tu_S[kmap]).max()))
                md["Dist"] = max(md["Dist"], float(np.abs(Dist[:75] - tk_D[kmap]).max()), float(np.abs(Dist[75:] - tu_D[kmap]).max()))
                md["acc"] = max(md["acc"], float(abs(acc - their_acc[e])))
                au_ours.append(auroc(S))
                au_theirs.append(auroc(np.r_[tk_S[kmap], tu_S[kmap]]))
            res[mode] = {"max_abs_diff": md, "pass_1e-4": all(v <= 1e-4 for v in md.values()),
                         "S_auroc_ours": float(np.mean(au_ours)), "S_auroc_theirs": float(np.mean(au_theirs)),
                         "S_auroc_mean_abs_diff": float(np.mean(au_ours) - np.mean(au_theirs)),
                         "S_auroc_max_abs_episode_diff": float(np.abs(np.array(au_ours) - np.array(au_theirs)).max())}
        out[f"{sh}shot"] = {**res, "n_episodes": n_eps}
        print(sh, json.dumps(out[f"{sh}shot"]), flush=True)
        del model, host
        torch.cuda.empty_cache()
    p = REPO / "results" / "generated" / "fidelity_snatcher.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    if a.record:
        import hashlib
        import subprocess
        import time
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "ccp", "scripts", "configs"], cwd=REPO, capture_output=True,
                                    text=True).stdout.strip())
        rec = {"check": "SnaTCHer-F adapter vs SnaTCHer's FSLTrainer.evaluate_test, end to end and on the same encoder batch",
               "episodes": f"first {n_eps} of mini valA_fsor_5way_{{1,5}}shot (validation)", "results": out,
               "script": "scripts/fidelity/snatcher_fidelity.py",
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "git_head": head, "uncommitted_code_changes": dirty, "third_party": "SnaTCHer @ 335c42469f0a7ad72c5c3480c8effc8c293823e0",
               "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
               "when": time.strftime("%Y-%m-%d %H:%M:%S")}
        q = REPO / "results" / "checks" / "snatcher_fidelity_end_to_end.json"
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"recorded {q.relative_to(REPO).as_posix()} (sha256 {hashlib.sha256(q.read_bytes()).hexdigest()}); list it with "
              f"python scripts/provenance.py add-check {q.relative_to(REPO).as_posix()}")


if __name__ == "__main__":
    main()
