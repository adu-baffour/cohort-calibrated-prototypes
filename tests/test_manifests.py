"""Episode manifests: sha256 checks fail loudly; test manifests need allow_test; regeneration from the recorded seeds (data)."""
import shutil

import numpy as np
import pytest

from ccp.episodes import MANIFEST_DIR, Episodes, ManifestError, SplitAccessError, make_fsor_episodes, sha256sums


def test_all_listed_manifests_load_and_match():
    sums = sha256sums()
    assert len(sums) == 30   # 14 paper + 4 valC (prereg_variants.md) + 10 final round (final_round.md) + 2 CUB valB (foundation_round.md)
    for rel in sums:
        ds, name = rel.split("/")
        if "_fsor_" not in name:
            continue
        e = Episodes(ds, name, allow_test=True)
        assert e.support.shape == (e.n, 5, e.shot) and e.query.shape == (e.n, 5, 15) and e.unk_query.shape == (e.n, 5, 15)
        assert e.n == (600 if e.split == "test" else 1000)
        assert len(set(e.known[0]) & set(e.unknown[0])) == 0


def test_test_manifest_refused_without_flag():
    with pytest.raises(SplitAccessError):
        Episodes("mini", "test_fsor_5way_1shot")


def test_tampered_manifest_fails_loudly(tmp_path):
    root = tmp_path / "manifests"
    shutil.copytree(MANIFEST_DIR, root)
    p = root / "mini" / "valA_fsor_5way_1shot.json"
    txt = p.read_text(encoding="utf-8")
    p.write_text(txt.replace('"seed": 20260951', '"seed": 20260952'), encoding="utf-8", newline="\n")
    with pytest.raises(ManifestError):
        Episodes("mini", "valA_fsor_5way_1shot", root=root)
    Episodes("mini", "valA_fsor_5way_5shot", root=root)  # untouched files still load


def test_generator_is_deterministic():
    c2i = {c: np.arange(c * 30, c * 30 + 30) for c in range(12)}
    a, _ = make_fsor_episodes(c2i, 1, 5, 7)
    b, _ = make_fsor_episodes(c2i, 1, 5, 7)
    assert all(np.array_equal(a[k], b[k]) for k in a)


@pytest.mark.data
def test_manifests_regenerate_from_recorded_seeds():
    """Every FSOR manifest equals the registered generator's output (needs the prepared labels, $CCP_DATA)."""
    from ccp.data.images import class_to_indices
    for ds, sets in (("mini", ("valA", "valB", "test")), ("tiered", ("tvA", "tvB", "test"))):
        for sh in (1, 5):
            keys_a = None
            for st in sets:
                e = Episodes(ds, f"{st}_fsor_5way_{sh}shot", allow_test=True)
                c2i = class_to_indices(ds, "test" if st == "test" else "val")
                if ds == "tiered" and st != "test":
                    c2i = {c: v for c, v in c2i.items() if c in set(e.meta["classes"])}
                forbidden = keys_a if (ds == "mini" and st == "valB") else frozenset()
                arrs, used = make_fsor_episodes(c2i, sh, e.n, e.meta["seed"], forbidden=frozenset(forbidden or ()))
                if st in ("valA",):
                    keys_a = used
                for k in ("known", "support", "query", "unknown", "unk_query"):
                    assert np.array_equal(arrs[k], getattr(e, k)), (ds, st, sh, k)


@pytest.mark.data
def test_valc_manifests_regenerate_from_header():
    """valC (docs/prereg_variants.md, Section 5): seed 20261000 + shot; mini all 16 val classes minus the valA / valB keys, tiered the tvB classes
    minus the tvB keys. Reads validation labels and validation manifests only."""
    from ccp.data.images import class_to_indices
    for ds, key_sets in (("mini", ("valA", "valB")), ("tiered", ("tvB",))):
        for sh in (1, 5):
            e = Episodes(ds, f"valC_fsor_5way_{sh}shot")
            assert e.meta["seed"] == 20261000 + sh and e.split == "val"
            c2i = {c: v for c, v in class_to_indices(ds, "val").items() if c in set(e.meta["classes"])}
            forbidden = set()
            for s in key_sets:
                o = Episodes(ds, f"{s}_fsor_5way_{sh}shot")
                forbidden |= {(tuple(sorted(o.known[i])), tuple(sorted(o.unknown[i]))) for i in range(o.n)}
            arrs, _ = make_fsor_episodes(c2i, sh, e.n, e.meta["seed"], forbidden=frozenset(forbidden))
            for k in ("known", "support", "query", "unknown", "unk_query"):
                assert np.array_equal(arrs[k], getattr(e, k)), (ds, sh, k)


@pytest.mark.data
def test_final_round_manifests_regenerate_from_header():
    """CIFAR-FS valA / valB / test and CUB val / test (docs/preregistration/final_round.md, Section 3): seed = base + shot, valB excludes
    the valA keys. Reads class labels and manifests only (the test files' episode indices are compared, no image is read)."""
    from ccp.data.images import class_to_indices
    plan = [("cifarfs", "valA", "val", None, 20261020), ("cifarfs", "valB", "val", "valA", 20261030), ("cifarfs", "test", "test", None, 20261040),
            ("cub", "val", "val", None, 20261050), ("cub", "test", "test", None, 20261060)]
    for sh in (1, 5):
        used = {}
        for ds, name, split, forbid, base in plan:
            e = Episodes(ds, f"{name}_fsor_5way_{sh}shot", allow_test=True)
            assert e.meta["seed"] == base + sh and e.n == (600 if split == "test" else 1000)
            arrs, used[(ds, name)] = make_fsor_episodes(class_to_indices(ds, split), sh, e.n, base + sh,
                                                        forbidden=frozenset(used[(ds, forbid)]) if forbid else frozenset())
            for k in ("known", "support", "query", "unknown", "unk_query"):
                assert np.array_equal(arrs[k], getattr(e, k)), (ds, name, sh, k)
