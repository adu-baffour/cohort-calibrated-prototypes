"""The paper's Figures 2-4, from recorded results and the sha256-pinned release assets (CPU).

    python scripts/make_figures.py [Fig2 Fig3 Fig4 | all] [--check] [--no-images]

Outputs:
  paper_assets/figures/fig{2-4}_*.pdf     the paper's figures; PNGs in results/generated/figures/
  results/generated/figures/              the plotted data of every figure (git-ignored)
Figure 1 is a diagram drawn by hand, not generated. Figures 1 and 4 contain mini-ImageNet images and are not redistributed (git-ignored);
Figure 4 is drawn from the prepared mini-ImageNet arrays ($CCP_DATA).
--check compares, for every figure made in this run: the plotted data with the recorded data (results/recorded/figures_v0/*_data.csv; Figure 3:
the sha256 of the data the paper's figure was drawn from), and that every font of every written PDF is embedded (text of at least 7 pt is
enforced when a figure is saved). It fails on any difference.
--no-images (tier 1) recomputes and checks Figure 4's selection and plotted data without drawing the image panels (no image is read).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import re
import shutil
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figures import style as S  # noqa: E402

PAPER_OUT = ROOT / "paper_assets" / "figures"
REC = ROOT / "results" / "recorded" / "figures_v0"
# id -> (module, output file name, recorded data file)
FIGURES = {"Fig2": ("fig2_mechanism", "fig2_mechanism", "fig2_data.csv"),
           "Fig3": ("fig3_scg_gain", "fig3_scg_gain", None),
           "Fig4": ("fig4_examples", "fig4_examples", "fig3_data.csv")}
GENERATED_DATA = {"Fig2": "fig2_data.csv", "Fig4": "fig4_data.csv"}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(fid: str, no_images: bool = False):
    """Make one figure."""
    mod, name, _ = FIGURES[fid]
    m = importlib.import_module(f"scripts.figures.{mod}")
    with S.isolated():
        if fid == "Fig4":
            m.main(PAPER_OUT, draw_images=not no_images)
        else:
            m.main(PAPER_OUT)
    png = PAPER_OUT / f"{name}.png"     # PNG previews go with the generated files, not with the paper's PDFs
    if png.exists():
        S.GEN.mkdir(parents=True, exist_ok=True)
        shutil.move(png, S.GEN / png.name)
    print(f"{fid}: paper_assets/figures/{name}.pdf")


def pdf_objects(data: bytes) -> list[bytes]:
    """The raw PDF and every inflated FlateDecode stream (fonts can sit in compressed object streams)."""
    out = [data]
    for m in re.finditer(rb"stream\r?\n", data):
        start = m.end()
        end = data.find(b"endstream", start)
        try:
            out.append(zlib.decompress(data[start:end]))
        except zlib.error:
            continue
    return out


def fonts_embedded(pdf: Path) -> tuple[int, list[str]]:
    """(number of font descriptors, names of fonts without an embedded font file). Type 3 fonts carry their glyphs and count as embedded."""
    parts = pdf_objects(pdf.read_bytes())
    missing, n = [], 0
    for part in parts:
        for m in re.finditer(rb"<<(?:(?!>>).)*?/Type\s*/FontDescriptor.*?>>", part, re.S):
            n += 1
            if not re.search(rb"/FontFile[23]?\b", m.group(0)):
                name = re.search(rb"/FontName\s*/([^\s/>]+)", m.group(0))
                missing.append(name.group(1).decode("latin-1") if name else "?")
    return n, missing


def check(fids, no_images: bool = False) -> list[str]:
    bad = []
    for f in fids:
        _, name, rec = FIGURES[f]
        pdf = PAPER_OUT / f"{name}.pdf"
        if f == "Fig3":
            from scripts.figures.fig3_scg_gain import DATA_SHA256
            gen = S.GEN / "fig3_scg_gain_data.json"
            ok = gen.exists() and sha(gen) == DATA_SHA256
            print(f"Fig3: plotted data {'identical to' if ok else 'DIFFER from'} the data of the paper's figure (sha256 {DATA_SHA256[:16]})")
            bad += [] if ok else ["Fig3 data"]
        else:
            a, b = S.GEN / GENERATED_DATA[f], REC / rec
            if not a.exists():
                print(f"{f}: no plotted data written in this run")
                bad.append(f"{f} data missing")
            else:
                ra = list(csv.reader(open(a, encoding="utf-8")))
                rb = list(csv.reader(open(b, encoding="utf-8")))
                diff = [i for i, (x, y) in enumerate(zip(ra, rb)) if x != y] + ([-1] if len(ra) != len(rb) else [])
                print(f"{f}: plotted data {'identical to' if not diff else f'{len(diff)} rows differ from'} results/recorded/figures_v0/{rec} "
                      f"({len(ra)} rows)")
                for i in diff[:3]:
                    if i >= 0:
                        print("   generated:", ra[i], "\n   recorded: ", rb[i])
                bad += [f"{f} data"] if diff else []
        if f == "Fig4" and no_images:
            print("   image panels not drawn (--no-images): selection and plotted data checked")
        elif pdf.exists():
            n, missing = fonts_embedded(pdf)
            print(f"   {rel(pdf)}: {n} font descriptors, " + (f"NOT embedded: {missing}" if missing else "all embedded"))
            bad += [f"{f} fonts"] if missing or n == 0 else []
        else:
            print(f"   {rel(pdf)} missing")
            bad.append(f"{f} pdf missing")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", default=["all"])
    ap.add_argument("--check", action="store_true", help="compare the plotted data with the recorded data; check fonts")
    ap.add_argument("--no-images", action="store_true", help="Figure 4: check selection and plotted data without drawing images ($CCP_DATA)")
    a = ap.parse_args()
    fids = list(FIGURES) if a.ids in (["all"], []) else a.ids
    unknown = [f for f in fids if f not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure ids {unknown}; choose from {list(FIGURES)}")
    for f in fids:
        run(f, a.no_images)
    if a.check:
        bad = check(fids, a.no_images)
        if bad:
            raise SystemExit(f"figure check failed: {bad}")
        print("figures: all OK")


if __name__ == "__main__":
    main()
