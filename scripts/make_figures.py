"""The paper's Figures 1-4 and the repository figures, from recorded results and the sha256-pinned release assets (CPU).

    python scripts/make_figures.py [Fig1 Fig2 Fig3 Fig4 episode_gains sensitivity gain_vs_host | all] [--check] [--no-images]

Outputs:
  paper_assets/figures/fig{1-4}_*.pdf     the paper's figures (file names as in the paper's figs/ folder); PNGs in results/generated/figures/
  docs/results/figures/<name>.{pdf,png}   repository figures (former Supplementary Figures S1-S3) and their captions (README.md)
  results/generated/figures/              the plotted data of every figure (git-ignored)
Figures 1 and 4 contain mini-ImageNet images and are not redistributed (git-ignored; scripts/release_scan.py fails on a tracked file
with dataset images). Figure 1 is drawn in draw.io; its PDF (the paper's figs/fig1_method.pdf) and editable diagram (.drawio) are kept
locally in figures/src/ or taken from the extracted paper (paper/figs/fig1_method.pdf) and copied unchanged after their sha256 is checked
against configs/paper_pins.json; without either, Figure 1 is reported as skipped. Figure 4 is drawn from the prepared mini-ImageNet arrays
($CCP_DATA).
--check compares, for every figure made in this run: the plotted data with the recorded data (results/recorded/figures_v0/*_data.csv; Figure 3:
the sha256 of the data the paper's figure was drawn from), the Figure 1 copy with the paper's file (configs/paper_pins.json), and that every
font of every written PDF is embedded (text of at least 7 pt is enforced when a figure is saved). It fails on any difference.
--no-images (tier 1) recomputes and checks Figure 4's selection and plotted data without drawing the image panels (no image is read).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import re
import shutil
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figures import style as S  # noqa: E402

PAPER_OUT = ROOT / "paper_assets" / "figures"
DOCS_OUT = ROOT / "docs" / "results" / "figures"
REC = ROOT / "results" / "recorded" / "figures_v0"
FIG1_SOURCES = [ROOT / "figures" / "src" / "Figure1_CCP-A_overview.pdf", ROOT / "paper" / "figs" / "fig1_method.pdf"]
DATASET_IMAGES = {"Fig1", "Fig4"}   # figures that contain dataset images: generated locally, never tracked
PINS = ROOT / "configs" / "paper_pins.json"
# id -> (module, output file name, kind, recorded data file)
FIGURES = {"Fig1": (None, "fig1_method", "paper", None),
           "Fig2": ("fig2_mechanism", "fig2_mechanism", "paper", "fig2_data.csv"),
           "Fig3": ("fig3_scg_gain", "fig3_scg_gain", "paper", None),
           "Fig4": ("fig4_examples", "fig4_examples", "paper", "fig3_data.csv"),
           "episode_gains": ("episode_gains", "episode_gains", "docs", "fig5_data.csv"),
           "sensitivity": ("sensitivity", "sensitivity", "docs", "fig6_data.csv"),
           "gain_vs_host": ("gain_vs_host", "gain_vs_host", "docs", "fig4_data.csv")}
GENERATED_DATA = {"Fig2": "fig2_data.csv", "Fig4": "fig4_data.csv", "episode_gains": "episode_gains_data.csv",
                  "sensitivity": "sensitivity_data.csv", "gain_vs_host": "gain_vs_host_data.csv"}
DOC_TITLES = {"episode_gains": "Per-episode test gains of fixed-weight fusion (former Supplementary Figure S1)",
              "sensitivity": "Sensitivity to the cohort size and the fusion weight (former Supplementary Figure S2)",
              "gain_vs_host": "Test gain against the host's own AUROC (former Supplementary Figure S3)"}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fig1_pin() -> str:
    return json.loads(PINS.read_text(encoding="utf-8"))["checked"]["figs/fig1_method.pdf"]


def run(fid: str, no_images: bool = False):
    """Make one figure; returns the caption of a repository figure (None for the paper's figures, whose captions are in the paper)."""
    mod, name, kind, _ = FIGURES[fid]
    if fid == "Fig1":
        src = next((p for p in FIG1_SOURCES if p.exists()), None)
        if src is None:
            print("Fig1: skipped (contains dataset images, not redistributed; put the paper's figs/fig1_method.pdf in figures/src/ or extract "
                  "the paper)")
            return None
        if sha(src) != fig1_pin():
            raise SystemExit(f"Fig1: {rel(src)} differs from its pinned sha256 (configs/paper_pins.json)")
        PAPER_OUT.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, PAPER_OUT / f"{name}.pdf")
        print(f"Fig1: {rel(src)} -> paper_assets/figures/{name}.pdf (copied unchanged; git-ignored)")
        return None
    m = importlib.import_module(f"scripts.figures.{mod}")
    with S.isolated():
        return _make(m, fid, name, kind, no_images)


def _make(m, fid, name, kind, no_images):
    if kind == "paper":
        out = PAPER_OUT
        if fid == "Fig4":
            m.main(out, draw_images=not no_images)
        else:
            m.main(out)
        for png in [out / f"{name}.png"]:     # PNG previews go with the generated files, not with the paper's PDFs
            if png.exists():
                S.GEN.mkdir(parents=True, exist_ok=True)
                shutil.move(png, S.GEN / png.name)
        print(f"{fid}: paper_assets/figures/{name}.pdf")
        return None
    cap = m.main(DOCS_OUT)
    print(f"{fid}: docs/results/figures/{name}.pdf / .png")
    return cap


def write_doc_captions(caps: dict):
    """docs/results/figures/README.md: one section per repository figure (sections of figures not made in this run are kept)."""
    p = DOCS_OUT / "README.md"
    old = {}
    if p.exists():
        for block in p.read_text(encoding="utf-8").split("\n## ")[1:]:
            old[block.split("\n", 1)[0]] = block
    L = ["# Repository figures", "",
         "Generated by `python scripts/make_figures.py episode_gains sensitivity gain_vs_host` from recorded results (former Supplementary "
         "Figures S1-S3; they are not part of the paper). The plotted data equal the recorded data in `results/recorded/figures_v0/` "
         "(`--check`).", ""]
    for fid in ("episode_gains", "sensitivity", "gain_vs_host"):
        title = DOC_TITLES[fid]
        if fid in caps:
            L += [f"## {title}", "", f"![{title}]({fid}.png)", "", f"[PDF]({fid}.pdf)", "", caps[fid], ""]
        elif title in old:
            L += ["## " + old[title].rstrip("\n"), ""]
    DOCS_OUT.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L), encoding="utf-8", newline="\n")


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
        _, name, kind, rec = FIGURES[f]
        pdf = (PAPER_OUT if kind == "paper" else DOCS_OUT) / f"{name}.pdf"
        if f == "Fig1":
            if not pdf.exists():
                print("Fig1: skipped (not available locally: contains dataset images, not redistributed)")
                continue
            ok = sha(pdf) == fig1_pin()
            print(f"Fig1: {'identical to' if ok else 'DIFFERS from'} the paper's figs/fig1_method.pdf (pinned sha256)")
            bad += [] if ok else ["Fig1 copy"]
        elif f == "Fig3":
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
    caps = {}
    for f in fids:
        cap = run(f, a.no_images)
        if cap:
            caps[f] = cap
    if caps:
        write_doc_captions(caps)
    if a.check:
        bad = check(fids, a.no_images)
        if bad:
            raise SystemExit(f"figure check failed: {bad}")
        print("figures: all OK")


if __name__ == "__main__":
    main()
