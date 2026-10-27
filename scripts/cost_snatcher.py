"""SnaTCHer-F per-episode cost on mini valA (timing only, no metric is computed). GPU. Writes results/paper_prep/cost_snatcher_valA.json.

    python scripts/cost_snatcher.py [--episodes 200] [--warmup 20]

Same protocol as scripts/cost.py (the recorded Table 5 measurement, results/recorded/validation/fsor_cost.json): validation manifests
mini valA_fsor_5way_{1,5}shot only, mean over `--episodes` episodes per shot after `--warmup` warm-up episodes, torch.cuda.synchronize around
each timed block, extra memory = torch.cuda.max_memory_allocated inside the block minus the memory allocated before it.
Timed blocks per episode (images already in host memory, as in cost.py):
  SnaTCHer-F own score: its fp32 encoder on the 5K supports + 150 queries, then the transformer score S of ccp/hosts/snatcher_f.py episode();
  CCP-A on SnaTCHer-F features (extra over SnaTCHer-F): the registered flip-view GAP forward of the same encoder + CCP-A with its released bank.
The recorded cost file is not modified; this measurement is a separate, generated file (added after the recorded Table 5 measurement).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ccp.cohort import m_topk  # noqa: E402
from ccp.episodes import Episodes  # noqa: E402
from ccp.hosts import build_host  # noqa: E402
from ccp.hosts.base import gap, normalize_u8  # noqa: E402
from ccp.pipeline import host_images, load_bank  # noqa: E402

N_BASE = 38400  # mini val images follow the 38,400 base images in the global index


def timed(fn):
    import torch
    torch.cuda.synchronize(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter(); r = fn(); torch.cuda.synchronize()
    return r, time.perf_counter() - t, torch.cuda.max_memory_allocated() - base


def main():
    import torch
    import torch.nn.functional as F
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    a = ap.parse_args()
    res = {"gpu": torch.cuda.get_device_name(0), "protocol": __doc__.split("\n\n")[1].strip(), "episodes": a.episodes, "warmup": a.warmup}
    per = {}
    for sh in (1, 5):
        host = build_host("snatcher_f_mini", shot=sh)
        B = load_bank(host).to_device().B
        val = host_images(host, "val")
        eps = Episodes("mini", f"valA_fsor_5way_{sh}shot")
        assert eps.split != "test"
        ns = 5 * sh
        Ts, Tc, ms, mc = [], [], 0, 0
        for e in range(a.warmup + a.episodes):
            x = torch.from_numpy(np.array(val[eps.indices(e) - N_BASE])).cuda()

            def own():
                E = host.enc(normalize_u8(x)).mean((2, 3))
                return host.episode(E[:ns].view(5, sh, -1), E[ns:])[1]

            def ccp():
                with torch.no_grad():
                    q = gap(host.encoder_forward, x) + gap(host.encoder_forward, x.flip(2))
                    P = F.normalize(q[:ns].view(5, sh, -1).mean(1), dim=1)
                    U = F.normalize(q[ns:], dim=1)
                    return ((U @ P.T) - 0.5 * (m_topk(P, B)[None] + m_topk(U, B)[:, None])).max(1).values
            with torch.no_grad():
                _, ts, mss = timed(own)
            _, tc, mcc = timed(ccp)
            if e >= a.warmup:
                Ts.append(ts); Tc.append(tc); ms = max(ms, mss); mc = max(mc, mcc)
        per[f"{sh}shot"] = {
            "SnaTCHer-F (own score S, incl. its fp32 forward)": {"ms_per_episode": float(np.mean(Ts) * 1000), "peak_extra_MB": ms / 2**20},
            "CCP-A on SnaTCHer-F features (extra over SnaTCHer-F: flip-view forward + CCP-A)": {"ms_per_episode": float(np.mean(Tc) * 1000),
                                                                                                 "peak_extra_MB": mc / 2**20}}
        host = B = None
        torch.cuda.empty_cache()
    res["per_episode"] = per
    res["when"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = REPO / "results" / "paper_prep" / "cost_snatcher_valA.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(per, indent=1))


if __name__ == "__main__":
    main()
