"""The eight registered test rounds (paper Section 4.6, Table 1): their bundle blocks, approval tokens and STARTED / DONE markers.

A round's token is the first 16 hex characters of the sha256 of its YAML bundle block in the registration document. `check()` recomputes
every token from the registration documents and requires it to equal the token in the round's DONE marker, in the approval file and in
the round's test record (rounds (v)-(viii)), and the token listed in ccp.naming.ROUNDS. Nothing is written.
"""
from __future__ import annotations

import hashlib
import json
import re

from ccp import naming, recorded

# paper numeral -> (registration document, fenced block name, DONE marker, STARTED marker, approval file, test record or None)
ROUND_FILES = {
    "(i)": ("docs/preregistration/FINAL_BUNDLE_fsor.md", "yaml-fsor-mini", "results/recorded/test/FINAL_TEST_FSOR_MINI_DONE.json",
            "results/recorded/test/FINAL_TEST_FSOR_MINI_STARTED.json", "results/recorded/test/FINAL_APPROVED", None),
    "(ii)": ("docs/preregistration/FINAL_BUNDLE_fsor.md", "yaml-fsor-tiered", "results/recorded/test/FINAL_TEST_FSOR_TIERED_DONE.json",
             "results/recorded/test/FINAL_TEST_FSOR_TIERED_STARTED.json", "results/recorded/test/FINAL_APPROVED", None),
    "(iii)": ("docs/preregistration/FINAL_BUNDLE_fsor.md", "yaml-fsor-mini-dnpg", "results/recorded/test/FINAL_TEST_FSOR_MINI-DNPG_DONE.json",
              "results/recorded/test/FINAL_TEST_FSOR_MINI-DNPG_STARTED.json", "results/recorded/test/FINAL_APPROVED", None),
    "(iv)": ("docs/preregistration/FINAL_BUNDLE_fsor.md", "yaml-fsor-mini-snatcher", "results/recorded/test/FINAL_TEST_FSOR_MINI-SNATCHER_DONE.json",
             "results/recorded/test/FINAL_TEST_FSOR_MINI-SNATCHER_STARTED.json", "results/recorded/test/FINAL_APPROVED", None),
    "(v)": ("docs/preregistration/FINAL_BUNDLE_baselines.md", "yaml-fsor-baselines", "bundles/FINAL_TEST_BASELINES_DONE.json",
            "bundles/FINAL_TEST_BASELINES_STARTED.json", "bundles/FINAL_APPROVED", "results/baselines/test.json"),
    "(vi)": ("docs/preregistration/FINAL_BUNDLE_variants.md", "yaml-fsor-variants", "bundles/FINAL_TEST_VARIANTS_DONE.json",
             "bundles/FINAL_TEST_VARIANTS_STARTED.json", "bundles/FINAL_APPROVED", "results/variants/test.json"),
    "(vii)": ("docs/preregistration/FINAL_BUNDLE_final_round.md", "yaml-fsor-final-round", "bundles/FINAL_TEST_FINAL_ROUND_DONE.json",
              "bundles/FINAL_TEST_FINAL_ROUND_STARTED.json", "bundles/FINAL_APPROVED", "results/final_round/test.json"),
    "(viii)": ("docs/preregistration/FINAL_BUNDLE_foundation_round.md", "yaml-fsor-foundation-round", "bundles/FINAL_TEST_FOUNDATION_ROUND_DONE.json",
               "bundles/FINAL_TEST_FOUNDATION_ROUND_STARTED.json", "bundles/FINAL_APPROVED", "results/foundation_round/test.json"),
}


def block_token(doc: str, fence: str) -> str:
    txt = recorded.read_bytes(doc).decode("utf-8")
    m = re.search(rf"```{re.escape(fence)}\n(.*?)\n```", txt, re.S)
    if m is None:
        raise ValueError(f"{doc}: no {fence} block")
    return hashlib.sha256(m.group(1).encode("utf-8")).hexdigest()[:16]


def round_record(numeral: str) -> dict:
    """Token of the bundle block, the tokens of the DONE marker / approval file / test record, and the start and finish times."""
    doc, fence, done, started, approved, test = ROUND_FILES[numeral]
    d = json.loads(recorded.read_bytes(done))
    s = json.loads(recorded.read_bytes(started))
    out = {"round": numeral, "registered_name": next(r[1] for r in naming.ROUNDS if r[0] == numeral),
           "listed_token": next(r[2] for r in naming.ROUNDS if r[0] == numeral), "block_token": block_token(doc, fence),
           "done_token": d["token"], "approved": recorded.read_bytes(approved).decode("utf-8").split(),
           "started": s.get("started") or s.get("when"), "done": d.get("done") or d.get("when"), "registration_document": doc}
    if test is not None:
        out["test_record_token"] = json.loads(recorded.read_bytes(test))["token"]
    return out


def check() -> list[dict]:
    """Every round: block token == DONE token == listed token, in the approval file and (v-viii) == the test record's token."""
    out = []
    for numeral in ROUND_FILES:
        r = round_record(numeral)
        tok = r["block_token"]
        ok = tok == r["done_token"] == r["listed_token"] and tok in r["approved"] and r.get("test_record_token", tok) == tok
        out.append({**{k: v for k, v in r.items() if k != "approved"}, "in_approval_file": tok in r["approved"], "ok": ok})
    return out
