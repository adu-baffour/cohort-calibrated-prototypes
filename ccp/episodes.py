"""Fixed, hashed episode manifests (manifests/<dataset>/*.json; sha256 in manifests/SHA256SUMS).

FSOR episode (docs/protocol.md): 5 known classes (K support + 15 queries each) and 5 disjoint unknown classes (15 queries each, no
support): 75 known + 75 unknown queries. Arrays hold GLOBAL image indices of the dataset (docs/data.md, "Global index"):
  known (n, 5), support (n, 5, K), query (n, 5, 15), unknown (n, 5), unk_query (n, 5, 15)
Order used everywhere: support (class-major), known queries (class-major), unknown queries.

The JSON files are lossless conversions of the research repository's .npz episode files (their sha256 is kept as
`source_npz_sha256`; tests/test_manifests.py @data rebuilds them from the generator and the recorded seeds).
Test manifests (split = "test") load only with allow_test=True: the paper's test numbers come from single, pre-registered passes
(docs/protocol.md, "Single-pass testing"); re-evaluating them is the user's decision.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ccp.utils.paths import REPO, sha256_file

MANIFEST_DIR = REPO / "manifests"
FORMAT = "ccp-fsor-episodes/1"   # the format identifier written in the verbatim manifests (release 1.5.0); not a name
WAY, NQ = 5, 15


class ManifestError(ValueError):
    pass


class SplitAccessError(PermissionError):
    pass


def sha256sums(path: Path = MANIFEST_DIR / "SHA256SUMS") -> dict[str, str]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            h, name = line.split(maxsplit=1)
            out[name.strip().lstrip("*")] = h
    return out


def manifest_path(dataset: str, name: str) -> Path:
    return MANIFEST_DIR / dataset / (name if name.endswith(".json") else f"{name}.json")


class Episodes:
    def __init__(self, dataset: str, name: str, allow_test: bool = False, root: Path = MANIFEST_DIR):
        path = root / dataset / (name if name.endswith(".json") else f"{name}.json")
        rel = f"{dataset}/{path.name}"
        sums = sha256sums(root / "SHA256SUMS")
        if rel not in sums:
            raise ManifestError(f"{rel} is not listed in {root / 'SHA256SUMS'}")
        got = sha256_file(path)
        if got != sums[rel]:
            raise ManifestError(f"{rel}: sha256 {got} does not match SHA256SUMS {sums[rel]} (file modified or corrupted)")
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("format") != FORMAT:
            raise ManifestError(f"{rel}: unknown format {d.get('format')!r}")
        if d["split"] == "test" and not allow_test:
            raise SplitAccessError(f"{rel} is a TEST manifest; pass allow_test=True (scripts: --allow-test) only for a deliberate test evaluation")
        self.meta = {k: v for k, v in d.items() if k not in ("known", "support", "query", "unknown", "unk_query")}
        self.known = np.asarray(d["known"], dtype=np.int64)
        self.support = np.asarray(d["support"], dtype=np.int64)
        self.query = np.asarray(d["query"], dtype=np.int64)
        self.unknown = np.asarray(d["unknown"], dtype=np.int64)
        self.unk_query = np.asarray(d["unk_query"], dtype=np.int64)
        self.n, _, self.shot = self.support.shape
        self.dataset, self.name, self.split, self.sha256 = dataset, path.stem, d["split"], got
        if self.meta["n_episodes"] != self.n or self.meta["shot"] != self.shot:
            raise ManifestError(f"{rel}: header does not match the arrays")

    def indices(self, e: int) -> np.ndarray:
        """support (class-major), known queries (class-major), unknown queries: 5K + 150 global indices."""
        return np.concatenate([self.support[e].reshape(-1), self.query[e].reshape(-1), self.unk_query[e].reshape(-1)])

    def all_indices(self) -> np.ndarray:
        return np.concatenate([self.support.reshape(-1), self.query.reshape(-1), self.unk_query.reshape(-1)])

    def subset(self, n: int) -> "Episodes":
        import copy
        s = copy.copy(self)
        for k in ("known", "support", "query", "unknown", "unk_query"):
            setattr(s, k, getattr(self, k)[:n])
        s.n = n
        return s


def make_fsor_episodes(c2i: dict, shot: int, n: int, seed: int, forbidden=frozenset()):
    """The registered generator (research/fsor/fsor.py make_fsor_episodes). c2i: {class id: global indices}. The 10-class
    (known set, unknown set) key is unique within the file and absent from `forbidden`."""
    rng = np.random.RandomState(seed)
    classes = sorted(c2i)
    known = np.zeros((n, WAY), np.int64)
    unknown = np.zeros((n, WAY), np.int64)
    sup = np.zeros((n, WAY, shot), np.int64)
    qry = np.zeros((n, WAY, NQ), np.int64)
    uq = np.zeros((n, WAY, NQ), np.int64)
    used, e = set(), 0
    while e < n:
        perm = [classes[i] for i in rng.permutation(len(classes))[: 2 * WAY]]
        kc, uc = perm[:WAY], perm[WAY:]
        key = (tuple(sorted(kc)), tuple(sorted(uc)))
        if key in used or key in forbidden:
            continue
        used.add(key)
        for k, c in enumerate(kc):
            ids = rng.permutation(c2i[c])
            known[e, k], sup[e, k], qry[e, k] = c, ids[:shot], ids[shot: shot + NQ]
        for k, c in enumerate(uc):
            ids = rng.permutation(c2i[c])
            unknown[e, k], uq[e, k] = c, ids[:NQ]
        e += 1
    return {"known": known, "support": sup, "query": qry, "unknown": unknown, "unk_query": uq}, used


def write_manifest(path: Path, arrays: dict, header: dict) -> str:
    """Canonical JSON (sorted header keys, arrays as nested int lists, one line per array)."""
    body = {"format": FORMAT, **header}
    lines = ["{"]
    items = [(k, json.dumps(v, sort_keys=True)) for k, v in sorted(body.items())]
    items += [(k, json.dumps(np.asarray(arrays[k]).astype(int).tolist(), separators=(",", ":")))
              for k in ("known", "support", "query", "unknown", "unk_query")]
    lines.append(",\n".join(f"  {json.dumps(k)}: {v}" for k, v in items))
    lines.append("}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return sha256_file(path)


class ClosedEpisodes:
    """Closed-set 5-way episodes used only for FRN model selection (scripts/train/train_frn.py): classes (n, 5), support (n, 5, K),
    query (n, 5, 15)."""

    def __init__(self, dataset: str, name: str, root: Path = MANIFEST_DIR):
        path = root / dataset / (name if name.endswith(".json") else f"{name}.json")
        sums = sha256sums(root / "SHA256SUMS")
        rel = f"{dataset}/{path.name}"
        if sums.get(rel) != sha256_file(path):
            raise ManifestError(f"{rel}: sha256 mismatch or not listed")
        d = json.loads(path.read_text(encoding="utf-8"))
        self.classes = np.asarray(d["classes"], dtype=np.int64)
        self.support = np.asarray(d["support"], dtype=np.int64)
        self.query = np.asarray(d["query"], dtype=np.int64)
        self.n, self.way, self.shot = self.support.shape
        self.n_query = self.query.shape[2]

    def batch_indices(self, e: int) -> np.ndarray:
        return np.concatenate([self.support[e].reshape(-1), self.query[e].reshape(-1)])
