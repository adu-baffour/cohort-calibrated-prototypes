"""Tier 1 status reporting (scripts/reproduce.py): which steps count as passed, skipped or failed."""
from scripts import reproduce as R


def test_step_status():
    assert R.status("provenance", [0]) == "OK"
    assert R.status("numbers", [R.SKIPPED]) == "skipped"                     # needs the paper sources
    assert R.status("tables", [R.SKIPPED]) == "OK against paper_assets/; paper-source comparison skipped"
    assert R.status("tables", [1]) == "FAILED"
    assert R.status("figures", [0, 1]) == "FAILED"


def test_summary_line_without_paper_sources():
    res = {"provenance": "OK", "tables": R.status("tables", [R.SKIPPED]), "numbers": "skipped", "paper-pins": "skipped"}
    assert R.summary(res) == "tier 1: all OK (skipped: numbers, paper-pins: paper sources not extracted)"
    assert R.summary({"provenance": "OK", "tables": "OK"}) == "tier 1: all OK"
    assert R.summary({"tables": "FAILED", "numbers": "skipped"}) == "tier 1: FAILED: tables"
