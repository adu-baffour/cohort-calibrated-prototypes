"""Test round (vii), the final round (paper Table 1; docs/preregistration/final_round.md): (A) CIFAR-FS, (B) mini-ImageNet -> CUB transfer,
(C) OSCR / FPR95 for the configurations of rounds (i)-(iv), (D) SCG on test. No method choice: CCP-A and CCP-QW (lambda 1, beta 1/2) fixed.

One scorer, `score_host`, is shared by the validation steps, the bundle dry run and the registered test pass (scripts/rounds/final_round.py).
For each (host, shot, manifest) it embeds the target split once and returns, per row, the per-query unknown score u (n, 150; higher =
unknown), the per-known-query correctness (n, 75) and the per-episode accuracy computed exactly as in the recorded passes (torch float32
mean x 100), so the recorded AUROC and accuracy are reproduced bit for bit.

Rows (paper names; ccp.naming gives the recorded keys):
  COS, COSF, CCP-A                         registry scores (frozen hosts: own labels; trained hosts: as rows of the recorded passes)
  CCP-QW                                   u = 1.0 m(q) - max_k [c_k - 0.5 m(p_k)], CCP-A's label (Eq. 6)
  host                                     frozen host: COS (part C) or COSF (parts A, B); trained host: its own score S / D, own label
  H_fix                                    D + w u_CCP-A (frozen weight of configs/hosts)
  CCP-QW fused                             D + w_QW u_QW (w_QW frozen from results/variants/selection_valA.json)
  <baseline>                               a baseline at its valA-selected setting: matrix -> own label (frozen); evidence -> COSF + w e
                                           (COSF label); trained -> fused[<baseline>] = D + w u_b
  CCP-A, target bank / H_fix, target bank  the separate target-bank analysis (Section 5.5.3): a cohort bank of CUB base images, K = 1000
  ..., target bank, K=154                  the same with K matched to the registered fraction, round(1000 x 5,903 / 38,400) = 154
"""
from __future__ import annotations

import numpy as np

from ccp import fusion, metrics

QW_LAMBDA, QW_BETA = 1.0, 0.5
WAY, NQ = 5, 15
BASE_BANK_SIZE = 38400        # the registered K = 1000 is 1000 / 38,400 of the mini-ImageNet base bank
TARGET_ROWS = ("CCP-A, target bank", "CCP-A, target bank, K=154")
TARGET_WEIGHT = "H_fix, target bank"


def matched_k(n_bank: int) -> int:
    """K matched to the registered fraction 1000 / 38,400 (CUB target bank: round(1000 x 5,903 / 38,400) = 154)."""
    return int(round(1000 * n_bank / BASE_BANK_SIZE))


def _u_pred(M):
    return -M.max(1).values, M.argmax(1)


def episode_rows(ep, names: list[str], evidence: set, ep_target=None, k_target: dict | None = None) -> dict:
    """{row: (u (150,) tensor, pred (150,) tensor | None)} for one episode of a frozen-feature pass. Evidence rows return (e, None)."""
    from ccp.cohort import REGISTERED_K, ccp_a_matrix, ccp_qw_matrix
    from ccp.scores import get_evidence, get_score
    out = {}
    for n in names:
        if n in ("COS", "COSF", "CCP-A") or (n not in evidence and n != "CCP-QW" and n not in TARGET_ROWS):
            out[n] = _u_pred(get_score(n)(ep))
        elif n in evidence:
            out[n] = (get_evidence(n)(ep), None)
    if "CCP-QW" in names:
        c, mq, mp = ep.cos(True), ep.m_q(REGISTERED_K), ep.m_p(REGISTERED_K)
        lab = ccp_a_matrix(c, mq, mp).argmax(1)
        out["CCP-QW"] = (-ccp_qw_matrix(c, mq, mp, QW_LAMBDA, QW_BETA).max(1).values, lab)
    for n in TARGET_ROWS:
        if n in names:
            k = k_target[n]
            out[n] = _u_pred(ccp_a_matrix(ep_target.cos(True), ep_target.m_q(k), ep_target.m_p(k)))
    return out


def with_bank(tf, device_bank):
    """A TargetFeatures sharing tf's target features (the same tensors, bit for bit) with another bank (the target bank)."""
    from ccp.scores import TargetFeatures
    t = TargetFeatures.__new__(TargetFeatures)
    t.bank, t.q0, t.qf, t.u0, t.uf, t._cache = device_bank, tf.q0, tf.qf, tf.u0, tf.uf, {}
    return t


def score_host(host_name: str, shot: int, manifest: str, dataset: str | None = None, rows: list[str] | None = None,
               evidence: set = frozenset(), target_bank=None, need_scg: bool = False, allow_test: bool = False, log=print) -> dict:
    """GPU. {'rows': {row: {'u': (n,150), 'correct': (n,75) bool, 'acc': (n,)}}, 'own': {...} | None, 'scg': {'x','m'} | None, 'n': n}.
    `rows` lists the frozen-feature rows (registry scores, 'CCP-QW', baselines, TARGET_ROWS); a trained host's own score is added as 'own'.
    With a target bank and need_scg, 'scg' also holds m_K(q) against the target bank at K = 1000 ('m_t1000') and at the matched K
    ('m_tK154'). dataset: the target dataset (default: the host's; 'cub' for the transfer part)."""
    import torch

    from ccp.episodes import Episodes
    from ccp.hosts import build_host, load_host_config
    from ccp.pipeline import Target, load_bank, score_host_own, target_bank_k
    from ccp.scores import EpisodeContext
    cfg = load_host_config(host_name)
    trained = cfg["kind"] in ("snatcher_f", "dnpg")
    host = build_host(cfg, shot=shot if trained else None)
    host.shot = shot
    bank = load_bank(host, allow_build=False, log=log)
    ds = dataset or cfg["dataset"]
    eps = Episodes(ds, manifest, allow_test=allow_test)
    split = "test" if eps.split == "test" else "val"
    target = Target(host, split, bank, need_pooled=cfg["kind"] == "snatcher_f", dataset=ds)
    tf_t = k_t = None
    rows = list(rows or [])
    if target_bank is not None:
        tf_t = with_bank(target.tf, target_bank.to_device(host.device))
        k_t = {TARGET_ROWS[0]: target_bank_k(target_bank), TARGET_ROWS[1]: matched_k(target_bank.raw_noflip.shape[0])}
    out = {"own": None, "n": eps.n, "scg": None}
    if trained:
        own, acc = score_host_own(host, target, eps)
        out["own"] = {"u": own["S" if cfg["kind"] == "snatcher_f" else "D"], "correct": own["correct"], "acc": acc}
    R = {n: {"u": np.zeros((eps.n, 150)), "correct": np.zeros((eps.n, 75), bool), "acc": np.zeros(eps.n)} for n in rows}
    X = np.zeros((eps.n, 150)) if need_scg else None
    Mq = np.zeros((eps.n, 150)) if need_scg else None
    Mt = {k: np.zeros((eps.n, 150)) for k in ("m_t1000", "m_tK154")} if need_scg and tf_t is not None else {}
    tgt = torch.arange(WAY, device=host.device).repeat_interleave(NQ)
    with torch.no_grad():
        for e in range(eps.n):
            r = target.rows(eps.indices(e))
            ep = EpisodeContext(target.tf, r, eps.shot)
            ep_t = EpisodeContext(tf_t, r, eps.shot) if tf_t is not None else None
            for n, (u, pred) in episode_rows(ep, rows, set(evidence), ep_t, k_t).items():
                R[n]["u"][e] = u.cpu().numpy()
                if pred is not None:
                    hit = pred[:75] == tgt
                    R[n]["correct"][e] = hit.cpu().numpy()
                    R[n]["acc"][e] = hit.float().mean().item() * 100
            if need_scg:
                X[e], Mq[e] = ep.cos(True).max(1).values.cpu().numpy(), ep.m_q().cpu().numpy()
                if Mt:
                    Mt["m_t1000"][e] = ep_t.m_q(k_t[TARGET_ROWS[0]]).cpu().numpy()
                    Mt["m_tK154"][e] = ep_t.m_q(k_t[TARGET_ROWS[1]]).cpu().numpy()
    out["rows"] = R
    if need_scg:
        out["scg"] = {"x": X, "m": Mq, **Mt}
    log(f"{host_name} shot {shot} {ds}/{manifest}: {len(rows)} rows{' + own' if trained else ''}")
    del target, host
    torch.cuda.empty_cache()
    return out


def assemble(raw: dict, trained: bool, part: str, weights: dict, evidence: set, host_kind: str = "") -> dict:
    """Reported rows {name: {'u', 'correct', 'acc'}} from a score_host result.
    weights: {'H_fix': w, 'CCP-QW fused': w_QW, <baseline>: w_b, 'H_fix, target bank': w} (trained) or {<evidence baseline>: w_b} (frozen).
    part 'C' (the configurations of rounds (i)-(iv)): frozen host row = COS; parts 'A', 'B': frozen host row = COSF.
    Label convention of the recorded trained-host passes, kept for exact reproduction: DNPG reports DNPG's own accuracy on every row (CCP-A
    and COSF alone included); SnaTCHer-F reports CCP-A / COSF with their own argmax labels. Fused rows use the host's labels."""
    R = raw["rows"]
    out = {}
    if not trained:
        out["host"] = R["COS" if part == "C" else "COSF"]
        for n, x in R.items():
            if n in evidence:
                out[n] = {"u": R["COSF"]["u"] + weights[n] * x["u"], "correct": R["COSF"]["correct"], "acc": R["COSF"]["acc"]}
            else:
                out[n] = x
        return out
    own = raw["own"]
    out["host"] = own
    for n in ("COSF", "CCP-A"):                        # unfused rows of the recorded trained passes; everything else enters fused with D
        if n in R:
            x = R[n]
            out[n] = {"u": x["u"], "correct": own["correct"], "acc": own["acc"]} if host_kind == "dnpg" else x

    def fused(u, w):
        return {"u": fusion.h_fix(own["u"], u, w), "correct": own["correct"], "acc": own["acc"]}
    if "CCP-A" in R:
        out["H_fix"] = fused(R["CCP-A"]["u"], weights["H_fix"])
    if "CCP-QW" in R:
        out["CCP-QW fused"] = fused(R["CCP-QW"]["u"], weights["CCP-QW fused"])
    for n in R:
        if n in weights and n not in ("H_fix", "CCP-QW fused", TARGET_WEIGHT) and n not in TARGET_ROWS:
            out[f"fused[{n}]"] = fused(R[n]["u"], weights[n])
    for n in TARGET_ROWS:
        if n in R:
            out["H_fix" + n[len("CCP-A"):]] = fused(R[n]["u"], weights[TARGET_WEIGHT])
    return out


def per_episode(rows: dict) -> dict:
    """{row: {'auroc', 'acc', 'oscr', 'fpr95'}} per-episode arrays."""
    return {n: {"auroc": metrics.episode_auroc(x["u"]), "acc": np.asarray(x["acc"], dtype=np.float64),
                "oscr": metrics.episode_oscr(x["u"], x["correct"]), "fpr95": metrics.episode_fpr95(x["u"])} for n, x in rows.items()}


def summary(pe: dict) -> dict:
    """Mean and 95% CI of every metric (AUROC / accuracy summarized exactly as the recorded passes: metrics.summarise)."""
    s = metrics.summarise(pe["auroc"], pe["acc"])
    for k in ("oscr", "fpr95"):
        s[k], s[f"{k}_ci95"] = float(np.mean(pe[k])), metrics.ci95(pe[k])
    return s


def paired_vs(pe_rows: list[dict], pe_refs: list[dict]) -> dict:
    """Paired per-episode differences row - reference for each metric, averaged over the hosts of a configuration (DNPG seeds) per episode.
    For FPR95 lower is better, so a negative difference is an improvement ('frac_gain' counts differences < 0 for FPR95)."""
    out = {}
    for k in ("auroc", "acc", "oscr", "fpr95"):
        d = np.mean([a[k] - b[k] for a, b in zip(pe_rows, pe_refs)], axis=0)
        p = metrics.paired(d, np.zeros_like(d))
        if k == "fpr95":
            p["frac_gain"] = float((d < 0).mean())
        out[k] = p
    return out
