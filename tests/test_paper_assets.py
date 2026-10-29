"""The paper's tables and figures and the detailed result documents regenerate from the recorded results.

Default: tables against paper_assets/, ranking marks, fonts of the committed PDFs, docs/results/ up to date. With CCP_TEST_DATA=1 (release
assets needed): the derived paired file and the plotted data of Figures 2-4.
"""
import pytest

from scripts import make_figures, make_results_docs, make_tables


@pytest.mark.parametrize("tid", list(make_tables.TABLES))
def test_table_regenerates(tid):
    body, src = make_tables.render(tid)
    assert (make_tables.OUT / f"{tid}.tex").read_text(encoding="utf-8") == make_tables.header(src) + body


def test_ranking_marks_ties_and_second():
    r = make_tables.ranks(["1.20", "1.30", "1.30", "1.10", None])
    assert r == ["underline", "bold", "bold", "", ""]                  # ties share bold; the next distinct value is second
    assert make_tables.ranks(["$0.35^{\\circ}$", "$-0.96^{\\downarrow}$", "$0.49$"], best="min") == ["underline", "bold", ""]
    assert make_tables.marked("$+$7.83", "bold") == r"\textbf{\boldmath $+$7.83}"
    assert make_tables.marked("70.28", "underline") == r"\underline{70.28}"


def test_committed_figures_embed_fonts():
    """The tracked figures: Figures 2 and 3 (Figures 1 and 4 contain dataset images, never tracked)."""
    for p in [make_figures.PAPER_OUT / f"{make_figures.FIGURES[f][1]}.pdf" for f in ("Fig2", "Fig3")]:
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
@pytest.mark.parametrize("fid", list(make_figures.FIGURES))
def test_figure_data_match_recorded(fid, monkeypatch, tmp_path):
    monkeypatch.setattr(make_figures, "PAPER_OUT", tmp_path / "paper")
    make_figures.run(fid, no_images=True)               # Figure 4: selection and plotted data, no image panels
    assert not make_figures.check([fid], no_images=True)
