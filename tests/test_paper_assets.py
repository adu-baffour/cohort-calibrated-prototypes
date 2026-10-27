"""The paper's tables, figures and numbers and the detailed result documents regenerate from the recorded results.

Default: tables against paper_assets/ (and against the paper's own tables when paper/ is extracted), ranking marks, fonts of the committed
PDFs, docs/results/ up to date. With CCP_TEST_DATA=1 (release assets needed): the derived paired file, the plotted data of every figure and
the number-by-number match of the paper's text (also needs paper/).
"""
import pytest

from scripts import make_figures, make_results_docs, make_tables

PAPER = make_tables.PAPER


@pytest.mark.parametrize("tid", list(make_tables.TABLES))
def test_table_regenerates(tid):
    body, src = make_tables.render(tid)
    assert (make_tables.OUT / f"{tid}.tex").read_text(encoding="utf-8") == make_tables.header(src) + body
    if PAPER.is_dir():
        assert body == (PAPER / f"{tid}.tex").read_text(encoding="utf-8")


def test_ranking_marks_ties_and_second():
    r = make_tables.ranks(["1.20", "1.30", "1.30", "1.10", None])
    assert r == ["underline", "bold", "bold", "", ""]                  # ties share bold; the next distinct value is second
    assert make_tables.ranks(["$0.35^{\\circ}$", "$-0.96^{\\downarrow}$", "$0.49$"], best="min") == ["underline", "bold", ""]
    assert make_tables.marked("$+$7.83", "bold") == r"\textbf{\boldmath $+$7.83}"
    assert make_tables.marked("70.28", "underline") == r"\underline{70.28}"


def test_committed_figures_embed_fonts():
    """The tracked figures: Figures 2 and 3 and the three repository figures (Figures 1 and 4 contain dataset images, never tracked)."""
    pdfs = [make_figures.PAPER_OUT / f"{make_figures.FIGURES[f][1]}.pdf" for f in ("Fig2", "Fig3")]
    pdfs += [make_figures.DOCS_OUT / f"{f}.pdf" for f in ("episode_gains", "sensitivity", "gain_vs_host")]
    for p in pdfs:
        n, missing = make_figures.fonts_embedded(p)
        assert n > 0 and not missing, (p.name, missing)


def test_results_docs_up_to_date():
    for p, s in make_results_docs.render().items():
        assert p.read_text(encoding="utf-8") == s, p


@pytest.mark.data
def test_derived_paired_file_recomputes():
    from scripts import derive_paired
    assert derive_paired.check()


@pytest.mark.data
@pytest.mark.parametrize("fid", ["Fig1", "Fig2", "Fig3", "Fig4", "episode_gains", "sensitivity", "gain_vs_host"])
def test_figure_data_match_recorded(fid, monkeypatch, tmp_path):
    monkeypatch.setattr(make_figures, "PAPER_OUT", tmp_path / "paper")
    monkeypatch.setattr(make_figures, "DOCS_OUT", tmp_path / "docs")
    if fid == "Fig1" and not any(p.exists() for p in make_figures.FIG1_SOURCES):
        pytest.skip("Figure 1 is not redistributed (dataset images); not available locally")
    make_figures.run(fid, no_images=True)               # Figure 4: selection and plotted data, no image panels
    assert not make_figures.check([fid], no_images=True)


@pytest.mark.data
def test_every_paper_number_matches():
    if not PAPER.is_dir():
        pytest.skip("paper/ not extracted (python scripts/paper_pin.py extract)")
    from scripts import paper_numbers
    rows, problems = paper_numbers.match()
    assert len(rows) > 400 and not problems, problems[:5]


def test_alternative_anchor_numbers_are_wildcards():
    from scripts import paper_numbers
    rx = paper_numbers.fuzzy(r"the mean AUROC by at most $4\times10^{-4}$ points")
    assert rx.search(r"the mean AUROC by at most $3\times10^{-4}$ points")       # a changed number is found, then reported as a difference
    assert not rx.search(r"the mean accuracy by at most $4\times10^{-4}$ points")
