"""Numbers of the paper's text: every numeric token of the abstract, the sections and the figure captions (paper/), with its position.

Used by scripts/paper_numbers.py. The tables (paper/tables/T*.tex) are not scanned: scripts/make_tables.py regenerates them byte for byte
from the recorded results. Commented-out text (highlights, end matter), citation / reference / label / URL arguments, and the inside of
display-math, algorithm and figure-placement commands are not scanned (equation constants are notation, not results). Names that contain
digits (ResNet-12, ViT-B/16, ImageNet-1k, CUB-200-2011, RTX 4090, ...) are masked before scanning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FILES = ["manuscript.tex", "sections/01_introduction.tex", "sections/02_related.tex", "sections/03_method.tex",
         "sections/04_experiments.tex", "sections/05_results.tex", "sections/06_conclusion.tex"]
NAMES = [r"ResNet-1[28]", r"ViT-[SBL]/1[46]", r"ImageNet-(?:1k|22k)", r"LVD-142M", r"CIFAR-100", r"CUB-200-2011", r"DINOv2", r"RTX\s*4090",
         r"Core i9", r"\bH[123]b?\b", r"\bG[1-4]\b", r"fp16", r"bfloat16", r"\bL2\b", r"S-norm", r"(?<![\\\w])[A-Za-z]+\d+[A-Za-z]*\d*",  # words with digits (not \times10)
         r"\\(?:begin|end)\{[^}]*\}", r"\\(?:cite|cref|Cref|ref|label|eqref|url|input|includegraphics|bibliography\w*|shorttitle)"
         r"(?:\[[^\]]*\])?\{[^}]*\}", r"\\affiliation\[\d\]", r"\\author\[\d\]", r"\\cormark\[\d\]", r"\\cortext\[\d\]",
         r"orcid=[0-9X-]+", r"\\documentclass\[[^\]]*\]\{[^}]*\}", r"\\tfrac12", r"\\frac12"]
SKIP_ENVS = ["equation", "equation*", "align", "align*", "algorithm", "algorithmic"]
NUMBER = re.compile(r"(?P<sign>(?:\$-\$|\$-|(?<![-\w])-(?!-)|−)\s*)?(?<![\w.^])(?<!\^\{)(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
                    r"(?P<exp>\s*\\times\s*10\^\{(?P<e>-?\d+)\})?")
POW = re.compile(r"(?<![\d.])10\^\{(?P<e>-?\d+)\}")


@dataclass
class Token:
    file: str
    line: int
    text: str           # as printed (sign and exponent included)
    value: float
    pos: int            # offset in the cleaned file text
    context: str


def clean(text: str) -> str:
    """Comments and skipped environments blanked out with spaces (offsets preserved); digit-bearing names masked."""
    out = []
    for line in text.split("\n"):
        m = re.search(r"(?<!\\)%", line)
        out.append(line if m is None else line[:m.start()] + " " * (len(line) - m.start()))
    t = "\n".join(out)
    for env in SKIP_ENVS:
        e = re.escape(env)
        t = re.sub(r"\\begin\{" + e + r"\}.*?\\end\{" + e + r"\}", lambda m: re.sub(r"[^\n]", " ", m.group(0)), t, flags=re.S)
    t = re.sub(r"\\\[.*?\\\]", lambda m: re.sub(r"[^\n]", " ", m.group(0)), t, flags=re.S)
    t = re.sub(r"\\(le|ge|pm|approx)(?=\d)", lambda m: "\\" + " " * len(m.group(1)), t)     # \pm0.21: the number starts after the command
    for pat in NAMES:
        t = re.sub(pat, lambda m: re.sub(r"[^\n]", "#", m.group(0)), t)
    return t


def tokens(rel: str) -> list[Token]:
    raw = (PAPER / rel).read_text(encoding="utf-8")
    if rel == "manuscript.tex":            # only the abstract carries results; the rest is front and end matter (titles, addresses)
        i, j = raw.index(r"\begin{abstract}"), raw.index(r"\end{abstract}")
        raw = re.sub(r"[^\n]", " ", raw[:i]) + raw[i:j] + re.sub(r"[^\n]", " ", raw[j:])
    t = clean(raw)
    out = []
    taken = []
    for m in POW.finditer(t):               # bare powers such as p = 10^{-5}
        if re.search(r"\\times\s*$", t[:m.start()]):
            continue
        out.append(Token(rel, t.count("\n", 0, m.start()) + 1, m.group(0), 10.0 ** int(m.group("e")), m.start(), ""))
        taken.append((m.start(), m.end()))
    for m in NUMBER.finditer(t):
        if any(a <= m.start("num") < b for a, b in taken):
            continue
        num = m.group("num")
        v = float(num.replace(",", ""))
        if m.group("exp"):
            v *= 10.0 ** int(m.group("e"))
        if m.group("sign"):
            v = -v
        start = m.start() if m.group("sign") else m.start("num")
        out.append(Token(rel, t.count("\n", 0, start) + 1, m.group(0).strip(), v, start, ""))
    out.sort(key=lambda k: k.pos)
    for k in out:
        a, b = max(0, k.pos - 70), min(len(t), k.pos + len(k.text) + 40)
        k.context = re.sub(r"\s+", " ", raw[a:b].replace("\n", " "))
    return out


def all_tokens() -> list[Token]:
    return [k for f in FILES for k in tokens(f)]


if __name__ == "__main__":
    ks = all_tokens()
    for k in ks:
        print(f"{k.file}:{k.line} [{k.text}] {k.value!r} :: {k.context}")
    print(len(ks), "tokens")
