"""Final round (docs/preregistration/final_round.md) on synthetic data (CPU): rows, the target-domain bank row, label conventions, paired
statistics, SCG (D1), the RFS ResNet-12 port and recipe schedule, the CUB split rule and the CIFAR-FS class lists."""
import numpy as np
import pytest
import torch

from ccp import metrics
from ccp.rounds import final_round as FR
from ccp.cohort import build_bank, ccp_a_matrix, ccp_qw_matrix
from ccp.scores import EpisodeContext, TargetFeatures, get_score
from conftest import make_episodes


@pytest.fixture(scope="module")
def ep(synth, bank):
    tf = TargetFeatures(synth["tgt"], synth["tgt_f"], bank.to_device("cpu"))
    eps = make_episodes(synth["tgt_lab"], 5, 2, seed=5)
    rows = np.concatenate([eps["support"][0].reshape(-1), eps["query"][0].reshape(-1), eps["unk_query"][0].reshape(-1)])
    return tf, rows


def test_episode_rows_v1_and_registry(ep):
    tf, rows = ep
    e = EpisodeContext(tf, rows, 5)
    out = FR.episode_rows(e, ["COS", "COSF", "CCP-A", "CCP-QW", "CL2N", "query cohort term"], {"query cohort term"})
    A = get_score("CCP-A")(e)
    assert torch.equal(out["CCP-A"][0], -A.max(1).values) and torch.equal(out["CCP-QW"][1], A.argmax(1))
    assert torch.equal(out["CCP-QW"][0], -ccp_qw_matrix(e.cos(True), e.m_q(), e.m_p(), 1.0, 0.5).max(1).values)
    assert out["query cohort term"][1] is None and torch.equal(out["query cohort term"][0], e.m_q(1000))
    assert torch.equal(out["CL2N"][0], -get_score("CL2N")(e).max(1).values)


def test_target_bank_row_uses_the_other_bank_and_the_same_features(ep, synth):
    tf, rows = ep
    tgt_bank = build_bank(synth["tgt"][:200], synth["tgt_f"][:200], synth["tgt_lab"][:200], np.arange(200), {"host": "t"}, "fp32")
    t2 = FR.with_bank(tf, tgt_bank.to_device("cpu"))
    assert t2.uf is tf.uf and t2.bank is not tf.bank
    e, e2 = EpisodeContext(tf, rows, 5), EpisodeContext(t2, rows, 5)
    k = 150
    out = FR.episode_rows(e, ["CCP-A, target bank"], set(), e2, {"CCP-A, target bank": k, "CCP-A, target bank, K=154": 5})
    B = t2.bank.B
    mq = (tf.uf[e.r_q] @ B.T).topk(k, 1).values.mean(1)
    mp = (e.pf @ B.T).topk(k, 1).values.mean(1)
    torch.testing.assert_close(out["CCP-A, target bank"][0], -ccp_a_matrix(e.cos(True), mq, mp).max(1).values, rtol=0, atol=1e-6)
    assert not torch.allclose(out["CCP-A, target bank"][0], -get_score("CCP-A")(e).max(1).values)
    both = FR.episode_rows(e, list(FR.TARGET_ROWS), set(), e2, {"CCP-A, target bank": k, "CCP-A, target bank, K=154": 30})
    mq30 = (tf.uf[e.r_q] @ B.T).topk(30, 1).values.mean(1)
    mp30 = (e.pf @ B.T).topk(30, 1).values.mean(1)
    torch.testing.assert_close(both["CCP-A, target bank, K=154"][0], -ccp_a_matrix(e.cos(True), mq30, mp30).max(1).values, rtol=0, atol=1e-6)


def test_matched_k_is_the_registered_fraction():
    assert FR.matched_k(5903) == 154 and FR.matched_k(38400) == 1000


def _row(rng, n=3):
    return {"u": rng.randn(n, 150), "correct": rng.rand(n, 75) < 0.7, "acc": rng.rand(n) * 100}


def test_assemble_conventions():
    rng = np.random.RandomState(0)
    R = {k: _row(rng) for k in ("COS", "COSF", "CCP-A", "CCP-QW", "query cohort term", "CL2N", "CCP-A, target bank", "CCP-A, target bank, K=154")}
    fz = FR.assemble({"rows": R, "own": None}, False, "C", {"query cohort term": 2.0}, {"query cohort term"})
    assert fz["host"] is R["COS"] and FR.assemble({"rows": R, "own": None}, False, "A", {"query cohort term": 2.0}, {"query cohort term"})["host"] is R["COSF"]
    np.testing.assert_array_equal(fz["query cohort term"]["u"], R["COSF"]["u"] + 2.0 * R["query cohort term"]["u"])
    assert fz["query cohort term"]["correct"] is R["COSF"]["correct"] and fz["CL2N"] is R["CL2N"]
    own = _row(rng)
    w = {"H_fix": 1.5, "CCP-QW fused": 2.5, "H_fix, target bank": 1.5, "CL2N": 3.0}
    for k in ("dnpg", "snatcher_f"):
        tr = FR.assemble({"rows": R, "own": own}, True, "B", w, set(), k)
        np.testing.assert_array_equal(tr["H_fix"]["u"], own["u"] + 1.5 * R["CCP-A"]["u"])
        np.testing.assert_array_equal(tr["CCP-QW fused"]["u"], own["u"] + 2.5 * R["CCP-QW"]["u"])
        np.testing.assert_array_equal(tr["fused[CL2N]"]["u"], own["u"] + 3.0 * R["CL2N"]["u"])
        np.testing.assert_array_equal(tr["H_fix, target bank"]["u"], own["u"] + 1.5 * R["CCP-A, target bank"]["u"])
        np.testing.assert_array_equal(tr["H_fix, target bank, K=154"]["u"], own["u"] + 1.5 * R["CCP-A, target bank, K=154"]["u"])
        assert all(tr[r]["correct"] is own["correct"] for r in ("host", "H_fix", "CCP-QW fused", "fused[CL2N]", "H_fix, target bank"))
        assert "CCP-QW" not in tr and "CL2N" not in tr
        # recorded label convention: DNPG reports its own accuracy on COSF / CCP-A alone, SnaTCHer-F their own labels
        assert (tr["CCP-A"]["correct"] is own["correct"]) == (k == "dnpg")


def test_per_episode_summary_and_paired_directions():
    rng = np.random.RandomState(2)
    r = _row(rng, 4)
    pe = FR.per_episode({"x": r})["x"]
    s = FR.summary(pe)
    ref = metrics.summarise(metrics.episode_auroc(r["u"]), r["acc"])
    assert s["auroc"] == ref["auroc"] and s["acc"] == ref["acc"] and s["auroc_ci95"] == ref["auroc_ci95"]
    np.testing.assert_allclose(pe["oscr"], metrics.episode_oscr(r["u"], r["correct"]))
    a = {"auroc": np.array([1.0, 2.0]), "acc": np.zeros(2), "oscr": np.zeros(2), "fpr95": np.array([10.0, 20.0])}
    b = {"auroc": np.array([0.0, 2.0]), "acc": np.zeros(2), "oscr": np.zeros(2), "fpr95": np.array([15.0, 20.0])}
    p = FR.paired_vs([a], [b])
    assert p["auroc"]["mean"] == 0.5 and p["auroc"]["frac_gain"] == 0.5
    assert p["fpr95"]["mean"] == -2.5 and p["fpr95"]["frac_gain"] == 0.5        # FPR95: lower is better


def test_scg_per_shot_is_the_registered_statistic():
    from ccp import scg
    rng = np.random.RandomState(4)
    x, m = rng.rand(30, 150), rng.rand(30, 150)
    d = scg.scg_per_shot(x, m)
    bins = scg.scg_bins(x, m, np.tile(np.r_[np.zeros(75), np.ones(75)], (30, 1)))
    assert d["std_gap"] == scg.scg_gap(bins) and d["raw_gap"] == scg.scg_gap(bins, False)


def test_rfs_resnet12_from_rfs_code():
    """RFS's own resnet12 (third_party/RFS @ f8c837b; skipped when not cloned, as in CI)."""
    from ccp.utils.paths import third_party_root
    if not (third_party_root() / "RFS" / "models" / "resnet.py").exists():
        pytest.skip("RFS not cloned (python third_party/clone.py RFS)")
    from ccp.hosts.rfs_resnet12 import layer4_map, resnet12_cifarfs
    torch.manual_seed(0)
    net = resnet12_cifarfs()
    keys = set(net.state_dict())
    assert {"layer1.0.conv1.weight", "layer4.0.downsample.0.weight", "layer4.0.bn3.running_mean", "classifier.weight"} <= keys
    assert net.layer1[0].drop_rate == 0.1 and not net.layer1[0].drop_block and net.layer4[0].drop_block and net.layer4[0].block_size == 2
    x = torch.randn(4, 3, 32, 32)
    net.eval()
    f = layer4_map(net, x)
    assert f.shape == (4, 640, 2, 2)
    torch.testing.assert_close(net(x), net.classifier(f.mean((2, 3))))
    assert net.layer4[0].num_batches_tracked == 2                 # every forward counts, eval mode included (RFS)
    if torch.cuda.is_available():                                  # RFS's DropBlock masks are created with .cuda()
        net = net.cuda().train()
        out = net(x.cuda())
        assert out.shape == (4, 64) and torch.isfinite(out).all()


def test_rfs_recipe_schedule():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("trs", Path(__file__).resolve().parents[1] / "scripts" / "train" / "train_rfs_cifarfs.py")
    trs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trs)
    lr = [trs.lr_at(e) for e in range(1, 91)]
    assert lr[0] == lr[44] == 0.05 and lr[45] == pytest.approx(0.005) and lr[59] == pytest.approx(0.005)
    assert lr[60] == pytest.approx(5e-4) and lr[75] == pytest.approx(5e-5) and lr[89] == pytest.approx(5e-5)
    assert trs.RECIPE["epochs"] == 90 and trs.RECIPE["batch_size"] == 64 and trs.RECIPE["weight_decay"] == 5e-4


def test_cub_split_rule_and_index(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("cubprep", Path(__file__).resolve().parents[1] / "scripts" / "prepare_data" / "cub.py")
    cub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cub)
    splits = [cub.split_of(i) for i in range(1, 201)]
    assert [splits.count(s) for s in ("train", "val", "test")] == [100, 50, 50]
    assert splits[:8] == ["val", "train", "test", "train", "val", "train", "test", "train"]
    (tmp_path / "classes.txt").write_text("1 001.A\n2 002.B\n3 003.C\n")
    (tmp_path / "images.txt").write_text("1 001.A/a1.jpg\n2 002.B/b1.jpg\n3 003.C/c1.jpg\n4 002.B/b2.jpg\n5 001.A/a2.jpg\n")
    (tmp_path / "image_class_labels.txt").write_text("1 1\n2 2\n3 3\n4 2\n5 1\n")
    idx = cub.read_index(tmp_path)
    assert idx["image_ids"] == [2, 4, 1, 5, 3] and list(idx["split"]) == ["train", "train", "val", "val", "test"]
    assert list(idx["labels"]) == [1, 1, 0, 0, 2]


def test_cifarfs_class_lists_are_the_pinned_ones():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("cfsprep", Path(__file__).resolve().parents[1] / "scripts" / "prepare_data" / "cifarfs.py")
    cfs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfs)
    lists = cfs.class_lists()
    assert lists["val"][:3] == ["otter", "motorcycle", "television"] and lists["test"][0] == "baby" and lists["train"][0] == "train"
