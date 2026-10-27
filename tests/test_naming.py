"""The naming map (ccp.naming, docs/naming.md): every recorded score key translates to the paper's name and back, the registry uses the
paper's names, the hypotheses, rounds and SCG keys map as the paper states, and the recorded-file loader translates every recorded JSON."""
import json
import re

import pytest

from conftest import ROOT

from ccp import baselines, naming, recorded, variants
from ccp.scores import registered_evidence, registered_scores


def recorded_keys() -> set[str]:
    """Every score / row key that appears in the recorded JSON files (raw, untranslated)."""
    keys = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                for part in str(k).split("|"):
                    keys.add(part)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            keys.add(o)
    for rel in recorded.provenance():
        if rel.endswith(".json"):
            walk(recorded.load(rel, paper_names=False))
    return {k for k in keys if re.match(r"^(H\[)?(B\d[ab]?-|V\d-)", k)}


KEYS = sorted(recorded_keys())


def test_every_recorded_score_key_round_trips():
    assert len(KEYS) > 50
    for k in KEYS:
        p = naming.paper_name(k)
        assert p != k and not re.search(r"\b(B\d|V\d)", p), (k, p)
        assert naming.recorded_key(p) == k, (k, p)


def test_registry_and_grids_use_paper_names_and_round_trip():
    names = registered_scores() + registered_evidence() + baselines.ALL_SETTINGS + variants.CCP_QW + variants.SHARED_COHORT + \
        variants.FULL_AS_NORM + variants.FUSION_WEIGHT_FACTOR
    for n in names:
        assert not re.match(r"^(B\d|V\d)", n), n
        assert naming.paper_name(naming.recorded_key(n)) == n, n
    rec = {naming.recorded_key(v) for v in baselines.ALL_SETTINGS}
    assert {k for k in KEYS if k.startswith("B") and not k.startswith("H[")} <= rec


@pytest.mark.parametrize("rec,paper", [
    ("B1-QTERM", "query cohort term"), ("B2-KNN/k=1000/sign=-", "kNN/k=1000/sign=-"), ("B3-RMD/lambda=1.0", "RMD/delta=1.0"),
    ("B4-CL2N", "CL2N"), ("B5-ZN", "ZN"), ("B6-NNN/alpha=0.75/k=1000", "NNN/alpha=0.75/k=1000"),
    ("B7a-QB-IS/beta=10", "QB-Norm IS/temperature=10"), ("B7b-QB-DIS/beta=20", "QB-Norm DIS/temperature=20"), ("B8-TEEN", "TEEN"),
    ("B9-CSLS-episode", "episode CSLS"), ("V1-QW/lambda=1.0/beta=0.5", "CCP-QW/lambda=1.0/beta=0.5"), ("V2-ASK", "full AS-norm"),
    ("V3-SHC/gamma=0.25", "shared-cohort correction/gamma=0.25"), ("V4-FW/mu=4", "fusion-weight factor/factor=4"),
    ("V4-FW/alone", "fusion-weight factor/CCP-A alone"), ("H[B2-KNN/k=100/sign=-]", "fused[kNN/k=100/sign=-]"), ("V1", "CCP-QW"),
    ("V1fused", "CCP-QW fused"), ("CCP-A@target-K154", "CCP-A, target bank, K=154"), ("CCP-A/flip-off", "CCP-A/single view"),
    ("H_fix", "H_fix"), ("COSF", "COSF"),
])
def test_specific_names(rec, paper):
    assert naming.paper_name(rec) == paper and naming.recorded_key(paper) == rec


def test_hypotheses_rounds_scg_and_splits():
    assert naming.HYPOTHESIS["H2b"] == "H2" and naming.HYPOTHESIS["H2"] == "H3" and naming.HYPOTHESIS["H1"] == "H1"
    assert naming.HYPOTHESIS["H2b-s"] == "H2 within-group" and naming.HYPOTHESIS["H2-s"] == "H3 within-group"
    assert naming.SCG_KEY == {"d1": "scg", "d1_valA": "scg_valA", "d1_valB": "scg_valB", "d1_test": "scg_test",
                              "d1_valA_predictions": "scg_valA_predictions"}
    assert naming.SPLIT["tvA"] == "valA" and naming.SPLIT["tvB"] == "valB"
    assert [r[0] for r in naming.ROUNDS] == ["(i)", "(ii)", "(iii)", "(iv)", "(v)", "(vi)", "(vii)", "(viii)"]
    assert [r[3] for r in naming.ROUNDS] == sorted(r[3] for r in naming.ROUNDS)


def test_round_tokens_match_the_approval_files():
    toks = (ROOT / "results/recorded/test/FINAL_APPROVED").read_text() + (ROOT / "bundles/FINAL_APPROVED").read_text()
    for _, _, token, _ in naming.ROUNDS:
        assert token in toks


def test_loader_translates_the_hypotheses_file():
    h = recorded.load("results/foundation_round/test_hypotheses.json")
    raw = recorded.load("results/foundation_round/test_hypotheses.json", paper_names=False)
    assert h["H2"] == raw["H2b"] and h["H3"] == raw["H2"] and h["H2 within-group"] == raw["H2b-s"]
    assert json.dumps(h["H1"]) == json.dumps(recorded.translate(raw["H1"]))


def test_loader_refuses_a_modified_file(tmp_path, monkeypatch):
    rel = "results/foundation_round/test_hypotheses.json"
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_bytes((ROOT / rel).read_bytes() + b" ")
    monkeypatch.setattr(recorded, "REPO", tmp_path)
    with pytest.raises(recorded.RecordedFileError, match="modified"):
        recorded.load(rel)
    with pytest.raises(recorded.RecordedFileError, match="not a verbatim file"):
        recorded.load("results/generated/anything.json")
