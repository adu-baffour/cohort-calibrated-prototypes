"""The one loader for recorded files (recorded results, round records, registration-era JSON).

Every recorded file is a verbatim copy whose sha256 is listed in results/PROVENANCE.md; `load` checks it before reading. With
paper_names=True (the default) the keys of a JSON file are translated to the paper's names (ccp.naming); the files themselves never change.

Translation rules (applied to dict keys recursively, and to string values that are score keys or baseline / variant group names):
  - score and row keys: ccp.naming.paper_name (for example B2-KNN/k=1000/sign=- -> kNN/k=1000/sign=-, H[B4-CL2N] -> fused[CL2N]);
  - composite keys joined with '|' are translated part by part (A|cifarfs_rfs_seed0|1|test|B1-QTERM);
  - SCG keys d1, d1_valA, d1_valB, d1_test -> scg, scg_valA, scg_valB, scg_test; record field names (ccp.naming.FIELD);
  - baseline and variant group names (B1 QTERM, V1 QW, ...);
  - per-file rules for keys that are too generic to translate everywhere: the hypothesis labels of test_hypotheses.json (H2b -> H2,
    H2 -> H3, ...), the acceptance criteria G1-G4 of the variants records, and the episode-wise standardization rows H / K of the
    trained-host records.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from functools import lru_cache
from pathlib import Path

from ccp import naming
from ccp.utils.paths import REPO

PROVENANCE = REPO / "results" / "PROVENANCE.md"
_SCORE_VALUE = re.compile(r"^(H\[)?(" + "|".join(re.escape(f) for f in naming.FAMILY) + r")(/|\]|$)")


class RecordedFileError(ValueError):
    pass


@lru_cache(maxsize=1)
def provenance() -> dict[str, str]:
    """{repository path: sha256} of every verbatim file (results/PROVENANCE.md)."""
    txt = PROVENANCE.read_text(encoding="utf-8").split("\n## Line endings")[0].split("\n## Release assets")[0]
    return dict(re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \| `[^`]+` \|$", txt, re.M))


def read_bytes(rel: str) -> bytes:
    """The bytes of a recorded file after checking its sha256 against results/PROVENANCE.md."""
    rel = Path(rel).as_posix()
    want = provenance().get(rel)
    if want is None:
        raise RecordedFileError(f"{rel} is not a verbatim file listed in results/PROVENANCE.md")
    data = (REPO / rel).read_bytes()
    if hashlib.sha256(data).hexdigest() != want:
        raise RecordedFileError(f"{rel}: sha256 differs from results/PROVENANCE.md (the recorded file was modified)")
    return data


def _profile(rel: str) -> dict:
    name = Path(rel).name
    extra = {}
    if name == "test_hypotheses.json":
        extra.update(naming.HYPOTHESIS)
    if rel.startswith("results/variants/"):
        extra.update(naming.CRITERION)
    if "dnpg" in name.lower() or "snatcher" in name.lower():
        extra.update(naming.EPISODE_STANDARDIZATION)
    return extra


def translate_key(k: str, extra: dict | None = None) -> str:
    if extra and k in extra:
        return extra[k]
    if k in naming.SCG_KEY:
        return naming.SCG_KEY[k]
    if k in naming.FIELD:
        return naming.FIELD[k]
    if k in naming.BASELINE_GROUP:
        return naming.BASELINE_GROUP[k]
    if k in naming.VARIANT_GROUP:
        return naming.VARIANT_GROUP[k]
    if "|" in k:
        return "|".join(translate_key(p, extra) for p in k.split("|"))
    return naming.paper_name(k)


def _translate_value(v: str) -> str:
    if v in naming.BASELINE_GROUP:
        return naming.BASELINE_GROUP[v]
    if v in naming.VARIANT_GROUP:
        return naming.VARIANT_GROUP[v]
    if _SCORE_VALUE.match(v):
        return naming.paper_name(v)
    return v


def translate(obj, extra: dict | None = None):
    """Recursively translate dict keys (and score-key string values) of a parsed recorded JSON object to the paper's names."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = translate_key(k, extra) if isinstance(k, str) else k
            if nk in out:
                raise RecordedFileError(f"two recorded keys translate to {nk!r}")
            out[nk] = translate(v, extra)
        return out
    if isinstance(obj, list):
        return [translate(v, extra) for v in obj]
    if isinstance(obj, str):
        return _translate_value(obj)
    return obj


def load(rel: str, paper_names: bool = True):
    """A recorded file, checked against results/PROVENANCE.md. JSON -> parsed object (keys translated unless paper_names=False);
    CSV -> list of dict rows (untranslated); anything else -> text."""
    data = read_bytes(rel)
    if rel.endswith(".json"):
        obj = json.loads(data.decode("utf-8"))
        return translate(obj, _profile(Path(rel).as_posix())) if paper_names else obj
    if rel.endswith(".csv"):
        return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    return data.decode("utf-8")
