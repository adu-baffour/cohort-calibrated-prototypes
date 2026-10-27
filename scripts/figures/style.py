"""Shared figure style (Neurocomputing / Elsevier cas-dc) for every rendered figure (Figures 2-4 and the repository figures).

Font: Liberation Sans 2.1.5 (SIL OFL 1.1), vendored in third_party/fonts/liberation-sans-2.1.5/ and registered with matplotlib at setup();
Arial (metric-compatible) is the fallback if the files are missing. Mathtext uses the same family. Text is at least MIN_PT = 7 pt everywhere
(save() draws the figure and refuses to write it if any visible text is smaller); mathtext sub- and superscripts are scaled by
SCRIPT_SCALE = 0.86 instead of matplotlib's 0.7, so they are at least 6 pt at final size.
Widths (cas-dc, measured with the class, 10 pt base): full width W2 = \\textwidth = 494.50888 pt, single column W1 = \\columnwidth =
238.25444 pt (TeX points, 72.27 per inch); figures are made at their final size and placed at 100%.
Output: vector PDF with embedded TrueType fonts (pdf.fonttype 42, no creation date, so the PDFs are byte-reproducible) and a PNG. No in-figure
titles, light gray grid, no top/right spines. Known = #2a78d6 (circle), unknown = #eb6834 (triangle); third series #1baf7a only with a
direct label.
"""
from __future__ import annotations

import contextlib
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib._mathtext as _mathtext  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "results" / "generated" / "figures"         # data files of every figure (git-ignored), compared by make_figures.py --check
FONT_DIR = ROOT / "third_party" / "fonts" / "liberation-sans-2.1.5"
FONT_FILES = ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf", "LiberationSans-BoldItalic.ttf")
KNOWN, UNKNOWN, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
GRID = "#e1e0d9"
MIN_PT = 7.0
SCRIPT_SCALE = 0.86                      # 7 pt x 0.86 = 6.02 pt
MM = 1 / 25.4
TEX_PT = 1 / 72.27                       # inches per TeX point
W2 = 494.50888 * TEX_PT                  # cas-dc \textwidth  (173.80 mm)
W1 = 238.25444 * TEX_PT                  # cas-dc \columnwidth (83.74 mm)
FAMILY = None


def font_family() -> str:
    """Register the vendored Liberation Sans files (once) and return the family name in use."""
    global FAMILY
    if FAMILY is None:
        import matplotlib.font_manager as fm
        files = [FONT_DIR / f for f in FONT_FILES if (FONT_DIR / f).exists()]
        for f in files:
            fm.fontManager.addfont(str(f))
        FAMILY = "Liberation Sans" if len(files) == len(FONT_FILES) else "Arial"
    return FAMILY


def setup():
    fam = font_family()
    _mathtext.SHRINK_FACTOR = SCRIPT_SCALE          # sub- / superscript size relative to the base text (matplotlib default 0.7)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [fam, "Arial", "DejaVu Sans"], "font.size": MIN_PT, "axes.labelsize": MIN_PT,
        "axes.titlesize": MIN_PT, "xtick.labelsize": MIN_PT, "ytick.labelsize": MIN_PT, "legend.fontsize": MIN_PT, "legend.frameon": False,
        "mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold", "mathtext.sf": fam,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "errorbar.capsize": 1.5, "savefig.dpi": 600,
    })
    return fam


@contextlib.contextmanager
def isolated():
    """Apply the figure style only inside the block: matplotlib's rcParams and mathtext script scale are restored afterwards."""
    shrink = _mathtext.SHRINK_FACTOR
    with matplotlib.rc_context():
        try:
            yield
        finally:
            _mathtext.SHRINK_FACTOR = shrink


def small_text(fig) -> list[tuple[str, float]]:
    """Visible, non-empty text artists below MIN_PT (after a draw, so tick labels exist)."""
    from matplotlib.text import Text
    fig.canvas.draw()
    return [(t.get_text()[:40], t.get_fontsize()) for t in fig.findobj(Text)
            if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_PT - 1e-9]


def final_bbox(fig, pad: float = 0.01):
    """The tight bounding box, widened symmetrically to exactly the figure width, so the saved PDF is the final width (W1 or W2) and is
    placed at 100% in the manuscript. Content wider than the figure width is an error (the layout must fit)."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer()).padded(pad)
    w = fig.get_figwidth()
    if bb.width > w + 1e-6:
        raise ValueError(f"content is {bb.width / TEX_PT:.1f} pt wide, more than the figure width {w / TEX_PT:.1f} pt")
    dx = (w - bb.width) / 2
    return Bbox.from_extents(bb.x0 - dx, bb.y0, bb.x1 + dx, bb.y1)


def save(fig, name: str, out: Path, png_dpi: int = 600, png_out: Path | None = None):
    """<out>/<name>.pdf (vector, fonts embedded) and <png_out or out>/<name>.png, after the minimum-text-size check."""
    bad = small_text(fig)
    if bad:
        raise ValueError(f"{name}: text below {MIN_PT} pt: {bad[:8]}")
    out.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": final_bbox(fig)}
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None}, **kw)
    (png_out or out).mkdir(parents=True, exist_ok=True)
    fig.savefig((png_out or out) / f"{name}.png", dpi=png_dpi, metadata={"Software": None}, **kw)
    plt.close(fig)


def write_csv(name: str, header, rows, out: Path = GEN):
    """The plotted values of a figure: <out>/<name>_data.csv (the recorded versions are in results/recorded/figures_v0/)."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}_data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def panel_label(ax, x, y, text):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
