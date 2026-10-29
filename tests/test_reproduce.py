"""Tier 1 status reporting (scripts/reproduce.py): which steps count as passed or failed."""
from scripts import reproduce as R


def test_step_status():
    assert R.status([0]) == "OK"
    assert R.status([1]) == "FAILED"
    assert R.status([0, 1]) == "FAILED"


def test_summary_line():
    assert R.summary({"provenance": "OK", "tables": "OK"}) == "tier 1: all OK"
    assert R.summary({"provenance": "OK", "tables": "FAILED", "figures": "FAILED"}) == "tier 1: FAILED: tables, figures"
