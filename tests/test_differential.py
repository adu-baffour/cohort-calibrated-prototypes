"""Differential tests: every ported function gives bit-identical outputs to the 1.5.0 package of the authors' earlier release repository (private) on random synthetic inputs.

The old package runs in a separate process (tests/oldpkg.py); scores are compared under ccp.naming.recorded_key (the old registry used the
recorded names). Tolerance: none; every comparison is exact (np.array_equal / ==), NaN equal to NaN."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import OLD_REASON, OLD_REPO
from oldpkg import OldJobs

from ccp import baselines, cohort, embedding, fusion, metrics, naming, scg, stats, variants
from ccp.cohort import Bank, build_bank
from ccp.rounds import final_round as FR
from ccp.scores import EpisodeContext, TargetFeatures, get_evidence, get_score, registered_evidence, registered_scores

OLD = OldJobs(OLD_REPO)
pytestmark = [pytest.mark.old, pytest.mark.skipif(not OLD.available, reason=OLD_REASON)]


def same(a, b, path="") -> None:
    """Exact equality of nested results (dicts, lists, tuples, arrays, floats); NaN == NaN."""
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), (path, list(a)[:5], list(b)[:5] if isinstance(b, dict) else b)
        for k in a:
            same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype, (path, a.shape, b.shape, a.dtype, b.dtype)
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), (path, np.nanmax(np.abs(a.astype(float) - b.astype(float))))
    elif isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        pass
    else:
        assert a == b and type(a) is type(b), (path, a, b)


# ---------------------------------------------------------------------------------------------------------------- synthetic inputs
rng = np.random.RandomState(20261026)
D, N_BANK, N_TARGET = 24, 3200, 700
raw0_bank = rng.randn(N_BANK, D).astype(np.float32)
rawf_bank = (raw0_bank + 0.3 * rng.randn(N_BANK, D)).astype(np.float32)
labels_bank = rng.randint(0, 40, N_BANK).astype(np.int64)
BANK = build_bank(torch.from_numpy(raw0_bank), torch.from_numpy(rawf_bank), labels_bank, np.arange(N_BANK), {"host": "synthetic"}, "fp32",
                  with_noflip_stats=True)
BANK_FIELDS = {k: getattr(BANK, k) for k in ("raw_noflip", "raw_flip", "bbar", "C", "class_means", "bank_index", "meta", "bbar_noflip", "C_noflip")}
raw0_t = rng.randn(N_TARGET, D).astype(np.float32)
rawf_t = (raw0_t + 0.3 * rng.randn(N_TARGET, D)).astype(np.float32)


def episodes(shot: int, n: int, seed: int) -> list[np.ndarray]:
    r = np.random.RandomState(seed)
    return [r.permutation(N_TARGET)[: 5 * shot + 150] for _ in range(n)]


ROWS = {1: episodes(1, 3, 1), 5: episodes(5, 3, 5)}


def contexts(shot: int):
    tf = TargetFeatures(torch.from_numpy(raw0_t), torch.from_numpy(rawf_t), Bank(**BANK_FIELDS).to_device("cpu"))
    return [EpisodeContext(tf, r, shot) for r in ROWS[shot]]


# ---------------------------------------------------------------------------------------------------------------- jobs: cohort and scores
V = torch.from_numpy(rng.randn(50, D).astype(np.float32))
V = torch.nn.functional.normalize(V, dim=1)
B_unit = embedding.unit_flip(torch.from_numpy(raw0_bank), torch.from_numpy(rawf_bank))
c = torch.from_numpy(rng.rand(150, 5).astype(np.float32))
mq, mp = torch.from_numpy(rng.rand(150).astype(np.float32)), torch.from_numpy(rng.rand(5).astype(np.float32))
sq, sp = torch.from_numpy(rng.rand(150).astype(np.float32) + 0.1), torch.from_numpy(rng.rand(5).astype(np.float32) + 0.1)
idx_q = torch.from_numpy(rng.randint(0, 300, (150, 40)).astype(np.int32))
idx_p = torch.from_numpy(rng.randint(0, 300, (5, 40)).astype(np.int32))
ov = cohort.cohort_overlap(idx_q, idx_p, 300)

COHORT = {  # new callable, its arguments, old function name
    "unit_flip": (embedding.unit_flip, (torch.from_numpy(raw0_t), torch.from_numpy(rawf_t)), "ccp.cohort.unit_flip"),
    "unit_flip_single": (embedding.unit_flip, (torch.from_numpy(raw0_t),), "ccp.cohort.unit_flip"),
    "m_topk": (cohort.m_topk, (V, B_unit, 100), "ccp.cohort.m_topk"),
    "m_topk_chunked": (lambda a, b: cohort.m_topk(a, b, 50, chunk=7), (V, B_unit), None),
    "topk_stats": (cohort.topk_stats, (V, B_unit, 100), "ccp.cohort.topk_stats"),
    "m_all": (cohort.m_all, (V, B_unit), "ccp.cohort.m_all"),
    "closed_form_stats": (cohort.closed_form_stats, (B_unit,), "ccp.cohort.closed_form_stats"),
    "s_norm_stats": (cohort.s_norm_stats, (V, torch.from_numpy(BANK.bbar), torch.from_numpy(BANK.C)), "ccp.cohort.s_norm_stats"),
    "ccp_a_matrix": (cohort.ccp_a_matrix, (c, mq, mp), "ccp.cohort.ccp_a_matrix"),
    "ccp_qw_matrix": (cohort.ccp_qw_matrix, (c, mq, mp, 1.0, 0.5), "ccp.cohort.qw_matrix"),
    "full_as_norm_matrix": (cohort.full_as_norm_matrix, (c, mq, sq, mp, sp), "ccp.cohort.ask_matrix"),
    "cohort_overlap": (cohort.cohort_overlap, (idx_q, idx_p, 300), "ccp.cohort.cohort_overlap"),
    "shared_cohort_matrix": (cohort.shared_cohort_matrix, (c, mq, mp, ov, 0.25), "ccp.cohort.shc_matrix"),
    "ccp_s_matrix": (cohort.ccp_s_matrix, (c, mq, sq, mp, sp), "ccp.cohort.ccp_s_matrix"),
}
for k, (_, args, old) in COHORT.items():
    if old:
        OLD.add(f"cohort:{k}", old, *args)
OLD.add("cohort:m_topk_chunked", "ccp.cohort.m_topk", V, B_unit, 50, chunk=7)
OLD.add("build_bank", "ccp.cohort.build_bank", torch.from_numpy(raw0_bank), torch.from_numpy(rawf_bank), labels_bank, np.arange(N_BANK),
        {"host": "synthetic"}, "fp32", with_noflip_stats=True)

NEW_SCORES = registered_scores() + registered_evidence()
for sh in (1, 5):
    OLD.add(f"scores:{sh}", battery="scores", bank=BANK_FIELDS, raw0=raw0_t, rawf=rawf_t, rows=ROWS[sh], shot=sh,
            names=[naming.recorded_key(n) for n in NEW_SCORES])
    OLD.add(f"score_episode:{sh}", battery="score_episode", bank=BANK_FIELDS, raw0=raw0_t, rawf=rawf_t, rows=ROWS[sh], shot=sh,
            names=["COS", "COSF", "CCP-A", "CCP-S", "B4-CL2N", "V2-ASK"])
    OLD.add(f"variants:{sh}", battery="variants", bank=BANK_FIELDS, raw0=raw0_t, rawf=rawf_t, rows=ROWS[sh], shot=sh)
FR_NAMES = ["COS", "COSF", "CCP-A", "CCP-QW", "query cohort term", "kNN/k=1000/sign=-", "RMD/delta=1.0", "NNN/alpha=0.75/k=1000",
            "QB-Norm DIS/temperature=20", "episode CSLS"]
FR_EVIDENCE = {"query cohort term", "kNN/k=1000/sign=-"}
OLD.add("final_round_rows", battery="final_round_rows", bank=BANK_FIELDS, raw0=raw0_t, rawf=rawf_t, rows=ROWS[5], shot=5,
        names=[naming.recorded_key(n) for n in FR_NAMES], evidence=[naming.recorded_key(n) for n in FR_EVIDENCE])

# ---------------------------------------------------------------------------------------------------------------- jobs: metrics, fusion, statistics
S = rng.randn(20, 150)
S[3, :10] = S[3, 10]                                  # ties
S[4] = np.round(S[4], 1)                              # many ties
CORRECT = rng.rand(20, 75) > 0.3
A1, A2 = rng.rand(600) * 100, rng.rand(600) * 100
LABELS = [rng.permutation(np.r_[np.zeros(75), np.ones(75)]) for _ in range(5)]
for i in range(5):
    OLD.add(f"auroc:{i}", "ccp.metrics.auroc", S[i])
    OLD.add(f"auroc_labels:{i}", "ccp.metrics.auroc", S[i], LABELS[i])
    OLD.add(f"oscr:{i}", "ccp.metrics.oscr", S[i], CORRECT[i])
    OLD.add(f"fpr95:{i}", "ccp.metrics.fpr95", S[i])
OLD.add("episode_auroc", "ccp.metrics.episode_auroc", S)
OLD.add("episode_oscr", "ccp.metrics.episode_oscr", S, CORRECT)
OLD.add("episode_fpr95", "ccp.metrics.episode_fpr95", S)
OLD.add("ci95", "ccp.metrics.ci95", A1)
OLD.add("summarise", "ccp.metrics.summarise", A1, A2)
OLD.add("paired", "ccp.metrics.paired", A1, A2)
OLD.add("paired_zero", "ccp.metrics.paired", A1, A1.copy())
OLD.add("mean_std", "ccp.metrics.mean_std", [73.1, 72.4, 74.0])
OLD.add("accuracy", "ccp.metrics.accuracy", rng.randint(0, 5, 75))
Dv, Uv, Cv = rng.randn(30, 150), rng.randn(30, 150) * 0.1, rng.randn(30, 150) * 0.2
OLD.add("fit_weight", "ccp.fusion.fit_weight", Dv, Uv)
OLD.add("h_fix", "ccp.fusion.h_fix", Dv, Uv, 2.5)
OLD.add("z_sum", "ccp.fusion.z_sum", Dv, Uv)
OLD.add("fused_rows", "ccp.fusion.fused_rows", Dv, Uv, Cv, 2.5, 1.7)
X, M, Y = rng.rand(40, 150), rng.rand(40, 150), np.tile(np.r_[np.zeros(75), np.ones(75)], (40, 1))
X[:, :5] = 0.5                                        # values on bin edges
OLD.add("scg_bins", "ccp.variants.scg_bins", X, M, Y)
OLD.add("scg_bins_sparse", "ccp.variants.scg_bins", X[:1, :30], M[:1, :30], np.r_[np.zeros(28), np.ones(2)])
OLD.add("scg_per_shot", "ccp.final_round.scg_per_shot", X, M)
OLD.add("scg_reading", "ccp.variants.scg_reading", 0.55, 0.42)
P = rng.rand(32) / 10
P[[3, 7]] = P[5]
OLD.add("holm", "ccp.foundation.holm", P.tolist())
OLD.add("wilcoxon_greater", "ccp.foundation.wilcoxon_greater", A1 - A2 + 3)
OLD.add("bonferroni_ci", "ccp.foundation.bonferroni_ci", A1 - A2, 32)
xs, ys = rng.rand(32), rng.rand(32)
grp = np.repeat(np.arange(8), 4)
OLD.add("spearman_perm", "ccp.foundation.spearman_perm", xs, ys, 2000, 20261006)
OLD.add("stratified", "ccp.foundation.stratified_spearman_perm", xs, ys, grp, 300, 20261007)
OLD.add("headroom_paired", "ccp.foundation.headroom_paired", np.r_[A1[:50], 100.0], np.r_[A2[:50], 100.0])
OLD.add("headroom_cell", "ccp.foundation.headroom_cell", 90.1, 88.7)
CELLS = {f"c{i}": rng.randn(100) + 0.1 * i for i in range(6)}
OLD.add("h1_table", "ccp.foundation.h1_table", CELLS)
OLD.add("make_episodes", "ccp.episodes.make_fsor_episodes", {c_: list(range(c_ * 30, c_ * 30 + 30)) for c_ in range(16)}, 5, 12, 20260961)

# ---------------------------------------------------------------------------------------------------------------- jobs: selection rules
VALA = {naming.recorded_key(v): float(rng.rand()) for v in baselines.ALL_SETTINGS}
VALA[naming.recorded_key("RMD/delta=0.5")] = VALA[naming.recorded_key("RMD/delta=0.1")]   # a tie: the earlier setting wins
OLD.add("baselines.choose", "ccp.baselines.choose", VALA)
VMEANS = {naming.recorded_key(v): float(rng.rand()) for v in variants.CCP_QW}
OLD.add("variants.choose", "ccp.variants.choose", VMEANS, [naming.recorded_key(v) for v in variants.CCP_QW])
CELL_AUC = {naming.recorded_key(v): {cell: float(rng.rand()) for cell in variants.CELLS} for v in variants.FUSION_WEIGHT_FACTOR}
OLD.add("variants.select_v4", "ccp.variants.select_v4", CELL_AUC)
GAIN = {cell: float(rng.randn() * 0.3 + 0.3) for cell in variants.CELLS}
ACC = {cell: float(rng.randn() * 0.1) for cell in variants.CELLS}
OLD.add("gates_B", "ccp.variants.gates_B", GAIN, ACC, variants.CELLS)
OLD.add("gate_C", "ccp.variants.gate_C", GAIN, variants.CELLS)
RAW_A = {"own": Dv, "own_acc": rng.rand(30) * 100, "COSF": (Cv, rng.rand(30) * 100), "COS": (Cv * 0.9, rng.rand(30) * 100),
         "CCP-A": (Uv, rng.rand(30) * 100), "kNN/k=10/sign=+": (Uv * 3, None), "CL2N": (Uv * 2, rng.rand(30) * 100)}
RAW_A_OLD = {naming.recorded_key(k): v for k, v in RAW_A.items()}
BL = ["kNN/k=10/sign=+", "CL2N"]
for kind in ("feat_frozen", "dnpg"):
    OLD.add(f"baselines.fit_weights:{kind}", "ccp.baselines.fit_weights", kind, RAW_A_OLD, [naming.recorded_key(v) for v in BL])
    OLD.add(f"baselines.rows:{kind}", "ccp.baselines.rows", kind, RAW_A_OLD, [naming.recorded_key(v) for v in BL],
            {naming.recorded_key(v): 1.5 for v in BL}, 2.0)


# ---------------------------------------------------------------------------------------------------------------- tests
@pytest.mark.parametrize("name", list(COHORT))
def test_cohort_functions(name):
    fn, args, _ = COHORT[name]
    same(fn(*args), OLD.result(f"cohort:{name}"))


def test_build_bank():
    old = OLD.result("build_bank")
    for k, v in BANK_FIELDS.items():
        same(v, getattr(old, k) if not isinstance(old, dict) else old[k], k)


@pytest.mark.parametrize("shot", [1, 5])
def test_every_registered_score(shot):
    old = OLD.result(f"scores:{shot}")
    for ep, o in zip(contexts(shot), old):
        assert sorted(naming.recorded_key(n) for n in NEW_SCORES) == sorted(o)
        for n in NEW_SCORES:
            new = get_score(n)(ep) if n in registered_scores() else get_evidence(n)(ep)
            same(new, o[naming.recorded_key(n)], n)


@pytest.mark.parametrize("shot", [1, 5])
def test_score_episode(shot):
    from ccp.scores import score_episode
    names = ["COS", "COSF", "CCP-A", "CCP-S", "CL2N", "full AS-norm"]
    for ep, o in zip(contexts(shot), OLD.result(f"score_episode:{shot}")):
        new = {naming.recorded_key(k): (v[0], v[1]) for k, v in score_episode(ep, names).items()}
        same(new, o)


@pytest.mark.parametrize("shot", [1, 5])
def test_variant_episode_scores(shot):
    for ep, o in zip(contexts(shot), OLD.result(f"variants:{shot}")):
        new = {naming.recorded_key(k): v for k, v in variants.episode_scores(ep).items()}
        same(new, o)


def test_final_round_episode_rows():
    for ep, o in zip(contexts(5), OLD.result("final_round_rows")):
        new = {naming.recorded_key(k): v for k, v in FR.episode_rows(ep, FR_NAMES, FR_EVIDENCE).items()}
        same(new, o)


def test_metrics():
    for i in range(5):
        same(metrics.auroc(S[i]), OLD.result(f"auroc:{i}"))
        same(metrics.oscr(S[i], CORRECT[i]), OLD.result(f"oscr:{i}"))
        same(metrics.fpr95(S[i]), OLD.result(f"fpr95:{i}"))
    same(metrics.episode_auroc(S), OLD.result("episode_auroc"))
    same(metrics.episode_oscr(S, CORRECT), OLD.result("episode_oscr"))
    same(metrics.episode_fpr95(S), OLD.result("episode_fpr95"))
    same(metrics.ci95(A1), OLD.result("ci95"))
    same(metrics.summarise(A1, A2), OLD.result("summarise"))
    same(metrics.paired(A1, A2), OLD.result("paired"))
    same(metrics.paired(A1, A1.copy()), OLD.result("paired_zero"))
    same(metrics.mean_std([73.1, 72.4, 74.0]), OLD.result("mean_std"))


def test_auroc_with_labels():
    for i in range(5):
        same(metrics.auroc(S[i], LABELS[i]), OLD.result(f"auroc_labels:{i}"))


def test_fusion():
    same(fusion.fit_weight(Dv, Uv), OLD.result("fit_weight"))
    same(fusion.h_fix(Dv, Uv, 2.5), OLD.result("h_fix"))
    same(fusion.episode_standardization(Dv, Uv), OLD.result("z_sum"))
    inv = {v: k for k, v in naming.EPISODE_STANDARDIZATION.items()}
    new = {inv.get(k, k): v for k, v in fusion.fused_rows(Dv, Uv, Cv, 2.5, 1.7).items()}
    old = OLD.result("fused_rows")
    same({k: new[k] for k in old}, old)


def test_scg():
    same(scg.scg_bins(X, M, Y), OLD.result("scg_bins"))
    same(scg.scg_bins(X[:1, :30], M[:1, :30], np.r_[np.zeros(28), np.ones(2)]), OLD.result("scg_bins_sparse"))
    same(scg.scg_per_shot(X, M), OLD.result("scg_per_shot"))
    same(scg.scg_reading(0.55, 0.42), OLD.result("scg_reading"))


def test_stats():
    same(stats.holm(P.tolist()), OLD.result("holm"))
    same(stats.wilcoxon_greater(A1 - A2 + 3), OLD.result("wilcoxon_greater"))
    same(stats.bonferroni_ci(A1 - A2, 32), OLD.result("bonferroni_ci"))
    same(stats.spearman_perm(xs, ys, 2000, 20261006), OLD.result("spearman_perm"))
    same(stats.within_group_spearman_perm(xs, ys, grp, 300, 20261007), OLD.result("stratified"))
    same(stats.headroom_paired(np.r_[A1[:50], 100.0], np.r_[A2[:50], 100.0]), OLD.result("headroom_paired"))
    same(stats.headroom_cell(90.1, 88.7), OLD.result("headroom_cell"))
    same(stats.h1_table(CELLS), OLD.result("h1_table"))


def test_episode_generator():
    from ccp.episodes import make_fsor_episodes
    new = make_fsor_episodes({c_: list(range(c_ * 30, c_ * 30 + 30)) for c_ in range(16)}, 5, 12, 20260961)
    same(new, OLD.result("make_episodes"))


def test_selection_rules():
    old = OLD.result("baselines.choose")
    new = baselines.choose({naming.paper_name(k): v for k, v in VALA.items()})
    assert [naming.recorded_key(v) for v in new.values()] == list(old.values())
    assert new["RMD"] in ("RMD/delta=0.1", "RMD/delta=0.5", "RMD/delta=1.0")
    assert naming.recorded_key(variants.choose({naming.paper_name(k): v for k, v in VMEANS.items()}, variants.CCP_QW)) == OLD.result("variants.choose")
    new_v4 = variants.select_factor({naming.paper_name(k): v for k, v in CELL_AUC.items()})
    assert {g: naming.recorded_key(v) for g, v in new_v4.items()} == OLD.result("variants.select_v4")
    old_b, new_b = OLD.result("gates_B"), variants.criteria_valB(GAIN, ACC, variants.CELLS)
    assert [old_b[k] for k in ("mean_gain", "min_cell_gain", "min_frozen_acc_change", "G1", "G2", "G3", "pass_G1_G3")] == \
           [new_b[k] for k in ("mean_gain", "min_cell_gain", "min_frozen_acc_change", "(i)", "(ii)", "(iii)", "pass_i_iii")]
    old_c, new_c = OLD.result("gate_C"), variants.criterion_valC(GAIN, variants.CELLS)
    assert [old_c[k] for k in ("mean_gain", "min_cell_gain", "G4")] == [new_c[k] for k in ("mean_gain", "min_cell_gain", "(iv)")]


@pytest.mark.parametrize("kind", ["feat_frozen", "dnpg"])
def test_baseline_rows(kind):
    w_new = baselines.fit_weights(kind, RAW_A, BL)
    same({naming.recorded_key(k): v for k, v in w_new.items()}, OLD.result(f"baselines.fit_weights:{kind}"))
    rows_new = baselines.rows(kind, RAW_A, BL, {v: 1.5 for v in BL}, 2.0)
    same({naming.recorded_key(k): v for k, v in rows_new.items()}, OLD.result(f"baselines.rows:{kind}"))


# ---------------------------------------------------------------------------------------------------------------- final-round assembly
def _raw(trained: bool, seed: int):
    r = np.random.RandomState(seed)
    rows = ["COSF", "CCP-A", "CCP-QW", "query cohort term", "CL2N", "CCP-A, target bank", "CCP-A, target bank, K=154"] + ([] if trained else ["COS"])
    R = {n: {"u": r.randn(12, 150), "correct": r.rand(12, 75) > 0.3, "acc": r.rand(12) * 100} for n in rows}
    own = {"u": r.randn(12, 150), "correct": r.rand(12, 75) > 0.4, "acc": r.rand(12) * 100} if trained else None
    return {"rows": R, "own": own, "n": 12}


def _old(x):
    """Paper names -> the recorded names the old package used (rows and weights)."""
    if isinstance(x, dict):
        return {naming.recorded_key(k) if isinstance(k, str) else k: _old(v) for k, v in x.items()}
    return x


ASSEMBLE = []
for _tr, _kind, _part in ((False, "", "A"), (False, "", "C"), (True, "dnpg", "B"), (True, "snatcher_f", "B")):
    _r = _raw(_tr, len(ASSEMBLE))
    _w = ({"H_fix": 2.1, "CCP-QW fused": 1.7, "H_fix, target bank": 2.1, "query cohort term": 0.9, "CL2N": 1.3} if _tr else {"query cohort term": 0.8})
    _ev = {"query cohort term"}
    ASSEMBLE.append((_r, _tr, _part, _w, _ev, _kind))
    OLD.add(f"assemble:{len(ASSEMBLE) - 1}", "ccp.final_round.assemble", _old(_r), _tr, _part, _old(_w), {naming.recorded_key(v) for v in _ev}, _kind)
_PE = FR.per_episode(FR.assemble(*ASSEMBLE[2]))
OLD.add("per_episode", "ccp.final_round.per_episode", _old(FR.assemble(*ASSEMBLE[2])))
OLD.add("summary", "ccp.final_round.summary", _PE["H_fix"])
OLD.add("paired_vs", "ccp.final_round.paired_vs", [_PE["H_fix"], _PE["CCP-QW fused"]], [_PE["host"], _PE["host"]])


@pytest.mark.parametrize("i", range(4))
def test_final_round_assemble(i):
    new = FR.assemble(*ASSEMBLE[i])
    old = OLD.result(f"assemble:{i}")
    same({naming.recorded_key(k): v for k, v in new.items()}, old)


def test_final_round_metrics():
    new = FR.per_episode(FR.assemble(*ASSEMBLE[2]))
    same({naming.recorded_key(k): v for k, v in new.items()}, OLD.result("per_episode"))
    same(FR.summary(_PE["H_fix"]), OLD.result("summary"))
    same(FR.paired_vs([_PE["H_fix"], _PE["CCP-QW fused"]], [_PE["host"], _PE["host"]]), OLD.result("paired_vs"))
