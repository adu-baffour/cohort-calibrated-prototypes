"""Foundation round (docs/preregistration/foundation_round.md), CPU: the input rule, the (n, D, 1, 1) -> registered gap() path, the BGR -> RGB
conversion of the tiered arrays, tiny random-init CLIP / DINOv2 ViTs from the pinned code (skipped when not cloned), the registered statistics,
the host configs and the CUB eval224 view."""
from pathlib import Path

import numpy as np
import pytest
import torch

from ccp import stats as FD
from ccp.hosts.base import gap
from ccp.hosts.vit_input import SIZE, FoundationHost, to_model_input
from ccp.utils.paths import load_yaml, third_party_root

REPO = Path(__file__).resolve().parents[1]
BACKBONES = ("clip_b16", "dinov2_s14", "dinov2_b14", "dinov2_l14")
DATASETS = ("mini", "tiered", "cifarfs", "cub")


def cfg(name="t", ds="mini", view="eval84", order="RGB"):
    return {"name": name, "dataset": ds, "view": view, "channel_order": order, "feature_dtype": "fp32", "channels_last": False}


def test_input_rule_constant_image_and_no_resample_at_224():
    mean, std = (0.5, 0.4, 0.3), (0.2, 0.25, 0.3)
    x = torch.full((2, 84, 84, 3), 128, dtype=torch.uint8)
    y = to_model_input(x, mean, std)
    assert y.shape == (2, 3, SIZE, SIZE)
    want = torch.tensor([(128 / 255 - m) / s for m, s in zip(mean, std)]).view(1, 3, 1, 1)
    torch.testing.assert_close(y, want.expand_as(y), rtol=0, atol=1e-5)          # bicubic of a constant image is the constant
    z = torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8)
    direct = ((z.permute(0, 3, 1, 2).float() / 255) - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    assert torch.equal(to_model_input(z, mean, std), direct)                     # 224 x 224 arrays are not resampled


def test_input_rule_matches_torch_bicubic_antialias_and_clamps():
    import torch.nn.functional as F
    x = torch.zeros((1, 32, 32, 3), dtype=torch.uint8)
    x[:, :, 16:] = 255                                                           # a hard edge: bicubic overshoots, clamp(0, 1) cuts it
    y = to_model_input(x, (0, 0, 0), (1, 1, 1))
    ref = F.interpolate(x.permute(0, 3, 1, 2).float() / 255, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
    assert ref.min() < 0 or ref.max() > 1
    torch.testing.assert_close(y, ref.clamp(0, 1), rtol=0, atol=0)
    assert y.min() >= 0 and y.max() <= 1


class Dummy(FoundationHost):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def __init__(self, c):
        super().__init__(c, device="cpu")
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(3, 7)

    def features(self, x):
        return self.lin(x.mean((2, 3)))


def test_feature_convention_n_d_1_1_through_registered_gap():
    h = Dummy(cfg())
    x = torch.randint(0, 256, (5, 84, 84, 3), dtype=torch.uint8)
    fm = h.encoder_forward(x)
    assert fm.shape == (5, 7, 1, 1)
    with torch.no_grad():
        g = gap(h.encoder_forward, x, batch=2)
        f = h.features(to_model_input(x, h.MEAN, h.STD))
    torch.testing.assert_close(g, f.half().float(), rtol=0, atol=0)              # the 1 x 1 map rounded to fp16, fp32 mean


def test_rgb_assertion_and_tiered_bgr_to_rgb(monkeypatch):
    with pytest.raises(ValueError, match="RGB"):
        Dummy(cfg(order="BGR"))
    from ccp import pipeline
    from ccp.data.images import ImageSet
    arr = np.random.RandomState(0).randint(0, 256, (4, 84, 84, 3)).astype(np.uint8)
    monkeypatch.setattr(pipeline, "open_split", lambda ds, split, view: ImageSet(arr, "BGR", 0, split, ds, view))
    h = Dummy(cfg(ds="tiered", view="bgr84"))
    out = pipeline.host_images(h, "train")
    np.testing.assert_array_equal(out, arr[..., ::-1])                           # stored BGR -> RGB for the foundation hosts
    for b in BACKBONES:
        c = load_yaml(REPO / "configs" / "hosts" / f"tiered_{b}.yaml")
        assert c["channel_order"] == "RGB" and c["view"] == "bgr84" and c["bank"]["embed"] == "subset_only"


def test_host_configs():
    seen = 0
    for b in BACKBONES:
        for ds in DATASETS:
            c = load_yaml(REPO / "configs" / "hosts" / f"{ds}_{b}.yaml")
            assert c["name"] == f"{ds}_{b}" and c["dataset"] == ds and c["channel_order"] == "RGB" and c["feature_dtype"] == "fp32"
            assert c["kind"] == ("clip_frozen" if b == "clip_b16" else "dinov2_frozen")
            assert c["view"] == {"mini": "eval84", "tiered": "bgr84", "cifarfs": "eval32", "cub": "eval84"}[ds]
            seen += 1
    for b in ("clip_b16", "dinov2_b14"):
        assert load_yaml(REPO / "configs" / "hosts" / f"cub_{b}_eval224.yaml")["view"] == "eval224"
    assert seen == 16


def test_tiny_clip_vit_from_pinned_code():
    if not (third_party_root() / "CLIP" / "clip" / "model.py").exists():
        pytest.skip("CLIP not cloned (python third_party/clone.py CLIP)")
    from ccp.hosts.clip_frozen import ClipFrozen, clip_model_module
    torch.manual_seed(0)
    v = clip_model_module().VisionTransformer(input_resolution=224, patch_size=16, width=32, layers=1, heads=2, output_dim=16).float()
    h = ClipFrozen({**cfg(), "kind": "clip_frozen"}, device="cpu", visual=v)
    x = torch.randint(0, 256, (2, 84, 84, 3), dtype=torch.uint8)
    with torch.no_grad():
        fm = h.encoder_forward(x)
    assert fm.shape == (2, 16, 1, 1) and torch.isfinite(fm).all()


def test_tiny_dinov2_vit_from_pinned_code():
    if not (third_party_root() / "dinov2" / "hubconf.py").exists():
        pytest.skip("DINOv2 not cloned (python third_party/clone.py dinov2)")
    from ccp.hosts.dinov2_frozen import Dinov2Frozen, build_dinov2
    torch.manual_seed(0)
    m = build_dinov2("vits14")                                                    # random init, no weights
    h = Dinov2Frozen({**cfg(), "kind": "dinov2_frozen", "arch": "vits14"}, device="cpu", model=m)
    x = torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.uint8)
    with torch.no_grad():
        fm = h.encoder_forward(x)
        cls = m.forward_features(to_model_input(x, h.MEAN, h.STD))["x_norm_clstoken"]
    assert fm.shape == (2, 384, 1, 1)
    torch.testing.assert_close(fm.flatten(1), cls, rtol=0, atol=0)


def test_holm_bonferroni_and_wilcoxon():
    assert FD.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert FD.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    d = np.array([1.0, 2.0, 3.0, 4.0])
    from scipy.stats import norm
    m, lo, hi = FD.bonferroni_ci(d, m=32)
    h = norm.ppf(1 - 0.05 / 64) * d.std() / 2
    assert (m, lo, hi) == pytest.approx((2.5, 2.5 - h, 2.5 + h))
    assert FD.wilcoxon_greater(np.zeros(10)) == 1.0
    assert FD.wilcoxon_greater(np.arange(1, 21, dtype=float)) < 1e-4
    assert FD.wilcoxon_greater(-np.arange(1, 21, dtype=float)) > 0.99
    tab = FD.h1_table({"a": np.arange(1, 31, dtype=float), "b": np.r_[np.ones(15), -np.ones(15)]})
    assert tab["n_holm_significant"] == 1 and tab["cells"]["a"]["p_holm"] <= 2 * tab["cells"]["a"]["p_one_sided"]


def test_spearman_permutation():
    x = np.arange(8, dtype=float)
    r = FD.spearman_perm(x, x * 2 + 1, n_perm=2000, seed=1)
    assert r["rho"] == pytest.approx(1.0) and r["p_one_sided"] < 0.01
    r2 = FD.spearman_perm(x, -x, n_perm=2000, seed=1)
    assert r2["rho"] == pytest.approx(-1.0) and r2["p_one_sided"] > 0.99
    assert FD.spearman_perm(x, x, n_perm=500, seed=3) == FD.spearman_perm(x, x, n_perm=500, seed=3)


def test_cub_eval224_view(tmp_path):
    import importlib.util

    from PIL import Image
    spec = importlib.util.spec_from_file_location("cubprep", REPO / "scripts" / "prepare_data" / "cub.py")
    cub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cub)
    for w, h in ((300, 400), (500, 333), (256, 256)):
        p = tmp_path / f"{w}x{h}.jpg"
        Image.fromarray(np.random.RandomState(0).randint(0, 256, (h, w, 3)).astype(np.uint8)).save(p)
        assert cub.eval224_view(str(p)).shape == (224, 224, 3)
    from ccp.data.images import SIDE_BY_VIEW
    assert SIDE_BY_VIEW["eval224"] == 224


def test_stratified_spearman_hand_built():
    # two strata of four: perfectly concordant in both -> statistic 1; the chance that both permuted strata are again perfectly ordered is
    # (1/24)^2 = 1/576, so the Monte Carlo p is about 0.0017
    x = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=float)
    y = np.array([10, 20, 30, 40, 5, 6, 7, 8], dtype=float)
    s = ["a"] * 4 + ["b"] * 4
    r = FD.within_group_spearman_perm(x, y, s, n_perm=20000, seed=1)
    assert r["stat_mean_within_rho"] == pytest.approx(1.0) and r["per_stratum_rho"] == pytest.approx([1.0, 1.0])
    assert 0.0005 < r["p_one_sided"] < 0.004
    # concordant in one stratum, discordant in the other -> statistic 0, p about one half
    y2 = np.array([10, 20, 30, 40, 8, 7, 6, 5], dtype=float)
    r2 = FD.within_group_spearman_perm(x, y2, s, n_perm=20000, seed=1)
    assert r2["stat_mean_within_rho"] == pytest.approx(0.0) and 0.4 < r2["p_one_sided"] < 0.7
    # a level shift between strata does not matter (within-stratum ranks only)
    r3 = FD.within_group_spearman_perm(x, y + np.r_[np.zeros(4), np.full(4, 1000.0)], s, n_perm=2000, seed=1)
    assert r3["stat_mean_within_rho"] == pytest.approx(1.0)


def _ckpt_cfg(tmp_path, monkeypatch, kind, arch=None, sha="0" * 64):
    monkeypatch.setenv("CCP_CKPT", str(tmp_path))
    c = {**cfg(), "kind": kind, "checkpoint": {"path": "w.pth", "sha256": sha}}
    if arch:
        c["arch"] = arch
    return c


def _no_downloads(monkeypatch):
    import urllib.request

    def boom(*a, **k):
        raise AssertionError("implicit download attempted")
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", boom)
    monkeypatch.setattr(torch.hub, "download_url_to_file", boom)
    monkeypatch.setattr(urllib.request, "urlretrieve", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)


def test_hash_mismatch_refuses_to_load(tmp_path, monkeypatch):
    from ccp.hosts.clip_frozen import ClipFrozen
    from ccp.hosts.dinov2_frozen import Dinov2Frozen
    (tmp_path / "w.pth").write_bytes(b"not the pinned weights")
    _no_downloads(monkeypatch)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        ClipFrozen(_ckpt_cfg(tmp_path, monkeypatch, "clip_frozen"), device="cpu")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        Dinov2Frozen(_ckpt_cfg(tmp_path, monkeypatch, "dinov2_frozen", "vits14"), device="cpu")


def test_dinov2_reads_only_the_pinned_local_file(tmp_path, monkeypatch):
    if not (third_party_root() / "dinov2" / "hubconf.py").exists():
        pytest.skip("DINOv2 not cloned")
    from ccp.hosts.dinov2_frozen import Dinov2Frozen, build_dinov2
    from ccp.utils.paths import sha256_file
    torch.manual_seed(0)
    sd = build_dinov2("vits14").state_dict()
    torch.save(sd, tmp_path / "w.pth")
    _no_downloads(monkeypatch)
    h = Dinov2Frozen(_ckpt_cfg(tmp_path, monkeypatch, "dinov2_frozen", "vits14", sha256_file(tmp_path / "w.pth")), device="cpu")
    for k, v in h.net.state_dict().items():
        assert torch.equal(v, sd[k]), k


def test_clip_reads_only_the_pinned_local_file(tmp_path, monkeypatch):
    if not (third_party_root() / "CLIP" / "clip" / "model.py").exists():
        pytest.skip("CLIP not cloned")
    from ccp.hosts.clip_frozen import ClipFrozen, clip_model_module
    from ccp.utils.paths import sha256_file
    torch.manual_seed(0)
    m = clip_model_module().CLIP(embed_dim=16, image_resolution=224, vision_layers=1, vision_width=64, vision_patch_size=16, context_length=8,
                                 vocab_size=50, transformer_width=64, transformer_heads=1, transformer_layers=1)
    torch.save(m.state_dict(), tmp_path / "w.pth")
    _no_downloads(monkeypatch)
    h = ClipFrozen(_ckpt_cfg(tmp_path, monkeypatch, "clip_frozen", sha=sha256_file(tmp_path / "w.pth")), device="cpu")
    ref = {k[len("visual."):]: v for k, v in m.state_dict().items() if k.startswith("visual.")}
    for k, v in h.net.state_dict().items():   # build_model's convert_weights rounds conv / linear / attention weights to fp16 (LayerNorm stays
        assert v.dtype == torch.float32 and (torch.equal(v, ref[k]) or torch.equal(v, ref[k].half().float())), k   # fp32); .float() restores fp32
    x = torch.randint(0, 256, (2, 84, 84, 3), dtype=torch.uint8)
    with torch.no_grad():
        assert h.encoder_forward(x).shape == (2, 16, 1, 1)


def test_batch_override_is_per_host_and_defaults_to_500():
    h = Dummy(cfg())
    assert int(h.cfg.get("batch", 500)) == 500
    h2 = Dummy({**cfg(), "batch": 250})
    x = torch.randint(0, 256, (3, 84, 84, 3), dtype=torch.uint8)
    with torch.no_grad():
        a, b = h.gap_pair(x)[0], h2.gap_pair(x)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.data
def test_clip_loader_matches_official_clip_load():
    """F1: the clip_frozen host path equals clip.load(<pinned local file>, jit=False).float().encode_image bit for bit (500 mini base images)."""
    import sys
    sys.path.insert(0, str(REPO))
    pytest.importorskip("ftfy")
    pytest.importorskip("regex")
    from checks.clip_loader_fidelity import compare
    r = compare(500)
    assert r["plain"]["bitwise_identical"] and r["flip"]["bitwise_identical"], r


def test_headroom_normalised_gain():
    # episodes: COSF 80 -> row 90 gives (90 - 80) / 20 = 0.5; COSF 90 -> row 85 gives -5 / 10 = -0.5; COSF 100 has no headroom (excluded)
    r = FD.headroom_paired([90.0, 85.0, 100.0], [80.0, 90.0, 100.0])
    assert r["n_used"] == 2 and r["n_excluded"] == 1
    assert r["mean"] == pytest.approx(0.0) and r["frac_gain"] == pytest.approx(0.5)
    from ccp import metrics
    assert r["ci95"] == pytest.approx(metrics.ci95(np.array([0.5, -0.5])))
    assert FD.headroom_cell(95.0, 90.0) == pytest.approx(0.5) and FD.headroom_cell(70.0, 80.0) == pytest.approx(-0.5)
