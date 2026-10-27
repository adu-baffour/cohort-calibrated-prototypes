"""Rebuild the Figure 2 / 4 validation score cache (scripts/figures/cache.py) with the release pipeline (tier 2: GPU, validation only).

    python scripts/figures/build_cache.py            # -> results/generated/figures/fig_cache_frozen_feat_mini_valB.npz and its sha256

Host frozen FEAT (mini-ImageNet) with the features stored in fp16, as in the feature cache the paper figures were computed from (evaluate.py
--feature-dtype fp16 reproduces that cache's numbers exactly). valB only. The rebuilt file is written next to the generated figures and
compared with the release asset by sha256; the release asset itself is never overwritten.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ccp import assets  # noqa: E402
from ccp.episodes import Episodes  # noqa: E402
from ccp.hosts import build_host, load_host_config  # noqa: E402
from ccp.pipeline import Target, load_bank  # noqa: E402
from ccp.scores import EpisodeContext  # noqa: E402
from ccp.utils.io import save_npz  # noqa: E402
from ccp.utils.paths import sha256_file  # noqa: E402
from scripts.figures import cache, style as S  # noqa: E402


def main():
    import torch
    cfg = copy.deepcopy(load_host_config("frozen_feat_mini"))
    cfg["feature_dtype"] = "fp16"
    cfg["bank"]["file"] = cfg["bank"]["file"].replace(".npz", "_fp16compat.npz")
    cfg["bank"]["sha256"] = None
    host = build_host(cfg)
    host.shot = None
    bank = load_bank(host, allow_build=True)
    tgt = Target(host, "val", bank)
    out = {}
    for sh in (1, 5):
        eps = Episodes("mini", f"valB_fsor_5way_{sh}shot")
        X, M, CF, CA = (np.zeros((eps.n, 150)) for _ in range(4))
        NC = np.zeros((eps.n, 150), np.int64)
        for e in range(eps.n):
            ep = EpisodeContext(tgt.tf, tgt.rows(eps.indices(e)), sh)
            c = ep.cos(True)
            mq = ep.m_q(1000)
            MA = c - 0.5 * (ep.m_p(1000)[None, :] + mq[:, None])          # CCP-A, Eq. 4
            X[e], M[e] = c.max(1).values.cpu().numpy(), mq.cpu().numpy()
            CF[e], CA[e], NC[e] = (-c.max(1).values).cpu().numpy(), (-MA.max(1).values).cpu().numpy(), c.argmax(1).cpu().numpy()
        out.update({f"x_{sh}shot": X, f"m_{sh}shot": M, f"cosf_{sh}shot": CF, f"ccpa_{sh}shot": CA, f"nearest_{sh}shot": NC})
    B = tgt.tf.bank.B
    nn = (tgt.tf.uf @ B.T).topk(5, dim=1)
    out["nn_idx"] = nn.indices.cpu().numpy().astype(np.int64)   # bank rows = global train indices 0..38,399
    out["nn_cos"] = nn.values.cpu().numpy()
    p = S.GEN / "fig_cache_frozen_feat_mini_valB.npz"
    save_npz(p, **out)
    h, want = sha256_file(p), assets.FILES[cache.KEY][1]
    print(f"{p}: sha256 {h} ({'identical to' if h == want else 'DIFFERS from'} the release asset {want})")
    del tgt, host
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
