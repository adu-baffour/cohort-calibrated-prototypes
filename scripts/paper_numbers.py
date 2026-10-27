"""Every number in the paper's text, matched to its source: writes docs/paper_numbers.md.

    python scripts/paper_numbers.py            # -> docs/paper_numbers.md
    python scripts/paper_numbers.py --check    # exit 1 if a number is unmatched or differs, or if docs/paper_numbers.md is out of date

scripts/paper_numbers_extract.py finds every numeric token of the abstract, the sections and the figure captions (paper/). Each token is
bound to one claim below: an anchor (a unique piece of the source text) and one source per number inside it, in order. Sources:
  recorded   a recorded file (ccp.recorded: sha256-checked, keys in the paper's names) and the key path inside it, ` / `-separated;
  computed   a formula over recorded values (minimum, maximum, count, ratio, difference), with its inputs named;
  setting    a protocol constant (episode layout, registered settings, dataset sizes, hardware), with the file that fixes it (not recomputed);
  text       not a result (notation, a cited paper's number, a word-like count of items listed in the text).
Rules (how a printed number is compared with its source value v):
  =     v rounded to the printed decimals equals the printed number (sign included; `mag`: absolute values);
  pct   100 v rounded to the printed decimals;     sci   v in scientific notation with the printed mantissa digits;
  pd    paired-difference rule: |printed - v| <= 0.01, where v is a difference of means rounded to two decimals (the paper's text prints
        such differences from the rounded table values);
  le / lt / ge / gt    the printed number is a bound: v <= / < / >= / > printed.
The tables are not listed here: scripts/make_tables.py regenerates them byte for byte from the recorded results (`--check`). The test
per-episode files are read only through recorded summaries (results/derived/paired_derived.json, results/recorded/figures_v0/) and the
figure data; this script reads no per-episode file, manifest, image or score.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ccp import recorded  # noqa: E402
import paper_numbers_extract as X  # noqa: E402

OUT = ROOT / "docs" / "paper_numbers.md"
SH = (1, 5)

# ------------------------------------------------------------------------------------------------------------------- recorded files
MINI = "results/recorded/test/FINAL_TEST_FSOR_MINI_DONE.json"
TIERED = "results/recorded/test/FINAL_TEST_FSOR_TIERED_DONE.json"
DNPG = "results/recorded/test/FINAL_TEST_FSOR_MINI-DNPG_DONE.json"
SNATCHER = "results/recorded/test/FINAL_TEST_FSOR_MINI-SNATCHER_DONE.json"
PUB = "results/recorded/published/published_sota.csv"
SOTA = "results/recorded/published/sota.csv"
ABL = "results/recorded/validation/ccp_ablations.json"
COST = "results/recorded/validation/fsor_cost.json"
SN_VAL = "results/recorded/validation/snatcher.json"
DN_EVAL = "results/recorded/validation/dnpg_eval.json"
DN_TIERED = "results/recorded/validation/dnpg_tiered_eval.json"
DN_HFIX = "results/recorded/validation/dnpg_hfix.json"
FR = "results/final_round/test.json"
FR_SCG_VAL = "results/final_round/d1_val.json"
FM = "results/foundation_round/test.json"
FM_H = "results/foundation_round/test_hypotheses.json"
FM_SEL = "results/foundation_round/selection_valA.json"
FM_VALB = "results/foundation_round/validation_valB.json"
VAR_B = "results/variants/validation_valB.json"
VAR_C = "results/variants/validation_valC.json"
VAR_T = "results/variants/test.json"
DER = "results/derived/paired_derived.json"
GEOM = "results/derived/bank_geometry_exploratory.json"
COST_SN = "results/derived/cost_snatcher_valA.json"
ENV = "results/derived/environment.json"
FIG2 = "results/recorded/figures_v0/fig2_data.csv"
FIG_HOST = "results/recorded/figures_v0/fig4_data.csv"
FIG_SENS = "results/recorded/figures_v0/fig6_data.csv"

FM_HOSTS = ("clip_b16", "dinov2_s14", "dinov2_b14", "dinov2_l14")
FM_DATA = ("mini", "tiered", "cifarfs", "cub")
DINO = FM_HOSTS[1:]


@lru_cache(maxsize=None)
def J(rel: str):
    return recorded.load(rel)


def at(rel: str, path: str):
    node = J(rel)
    for part in path.split(" / "):
        node = node[int(part[1:-1])] if part.startswith("[") else node[part]
    return node


def csv_rows(rel: str) -> list[dict]:
    return J(rel) if rel.endswith(".csv") and isinstance(J(rel), list) else list(csv.DictReader(io.StringIO(J(rel))))


def published(bench: str, method: str, col: str) -> float:
    """Value part of a recorded published cell ('69.10±0.29' -> 69.10)."""
    rows = [r for r in csv_rows(PUB) if r["benchmark"] == bench and r["method"] == method]
    assert len(rows) == 1, (bench, method)
    return float(re.match(r"\s*([0-9.]+)", rows[0][col]).group(1))


def sota(prefix: str, col: str, src: str) -> float:
    rows = [r for r in csv_rows(SOTA) if r["benchmark"].startswith("CIFAR-FS") and r["method"].startswith(prefix) and src in r["source_url_and_table"]]
    assert len(rows) == 1, (prefix, len(rows))
    return float(re.match(r"\s*([0-9.]+)", rows[0][col]).group(1))


# ------------------------------------------------------------------------------------------------------------------- specs
@dataclass
class Spec:
    kind: str                       # recorded / computed / setting / text
    source: str
    fn: Callable | None = None
    rule: str = "="
    mag: bool = False
    note: str = ""


def R(rel, path, rule="=", mag=False, scale=1.0):
    return Spec("recorded", f"`{rel}` › `{path}`" + (f" × {scale:g}" if scale != 1 else ""), lambda: at(rel, path) * scale, rule, mag)


def F(fn, source, rule="=", mag=False, note=""):
    return Spec("computed", source, fn, rule, mag, note)


def S(source, value=None):
    """A protocol constant: the source names where it is fixed; it is not recomputed here (`value` documents the expected number)."""
    return Spec("setting", source)


def T(reason):
    return Spec("text", reason)


# ------------------------------------------------------------------------------------------------------------------- helpers over recorded values
def r2(x):
    return round(x, 2)


def der_resnet(host, sh, metric="auroc", field_="mean"):
    return at(DER, f"resnet_vs_cosf / {host}|{sh} / {metric} / {field_}")


def frc(host, sh, row, metric="auroc", field_="mean"):
    """Round (vii) re-evaluation of the round (i)-(iv) hosts: row - host score, paired over the test episodes."""
    return at(FR, f"tables / C / {host} / {sh} / paired / host / {row} / {metric} / {field_}")


def h1(ds, b, sh):
    return at(FM_H, f"H1 / cells / {ds}|{b}|{sh} / mean")


def fm_row(ds, b, sh, row, metric="auroc", key=FM):
    return at(key, f"summaries / {ds}_{b}|T|{sh}|{row} / {metric}")


def table_row(host: str, sh: int, row: str, metric: str) -> float:
    """A mean of Tables 2-3 as recorded (FRN: mean over seeds 0-2)."""
    if host == "FRN":
        return at(MINI, f"frn_seeds / {sh}shot / {row} / {metric} / mean")
    rel = TIERED if "tiered" in host else MINI
    return at(rel, f"rows / {host} / {sh}shot / {row} / {metric}")


def rounded_diff(host, sh, metric, a="CCP-A", b="COSF"):
    """Difference of two means of Tables 2-3 after rounding each to two decimals (as the text compares them)."""
    return r2(table_row(host, sh, a, metric)) - r2(table_row(host, sh, b, metric))


def RD(host, sh, metric, rule="pd", mag=False):
    return F(lambda: rounded_diff(host, sh, metric), f"CCP-A minus COSF {metric}, rounded means of Tables 2-3 (`{MINI}` / `{TIERED}` › "
             f"`rows / {host} / {sh}shot`; FRN: `frn_seeds`)", rule=rule, mag=mag)


def resnet_scg_band(lo, hi, n):
    k = sum(1 for p in fig3_points() if p["round"] == "final" and lo < p["scg_val"] < hi)
    return lo if k == n else float("nan")


def mini_gains():
    """CCP gains of the five mini-ImageNet ResNet-12 hosts (frozen: CCP-A - COSF; trained: H_fix - host score), both shots."""
    g = [der_resnet(h, sh) for h in ("frozen FEAT mini", "TANE mini", "FRN mini") for sh in SH]
    return g + [frc(h, sh, "H_fix") for h in ("SnaTCHer-F mini", "DNPG mini") for sh in SH]


def cub_gains(field_="mean"):
    return [at(DER, f"cub / {h}|{sh} / auroc / host / {field_}") for h in ("frozen FEAT mini", "SnaTCHer-F mini", "DNPG mini") for sh in SH]


def cells(hosts=FM_HOSTS):
    return [(ds, b, sh) for b in hosts for ds in FM_DATA for sh in SH]


INDUCTIVE = ("query cohort term", "kNN", "RMD", "CL2N", "ZN", "NNN", "QB-Norm IS", "QB-Norm DIS", "TEEN")


def fam(k):
    k = k[6:-1] if k.startswith("fused[") else k
    return k.split("/")[0]


def fm_baselines(ds, b, sh):
    """{row key: test AUROC} of the eight inductive baseline families' rows with the valA-selected settings."""
    pre = f"{ds}_{b}|T|{sh}|"
    return {k[len(pre):]: v["auroc"] for k, v in J(FM)["summaries"].items() if k.startswith(pre) and fam(k[len(pre):]) in INDUCTIVE}


def fm_best(ds, b, sh):
    rows = fm_baselines(ds, b, sh)
    return max(rows, key=rows.get)


def table5():
    import make_tables
    return make_tables.baseline_cells()


def t5_counts():
    n = [0, 0, 0]
    for *_, d in table5():
        d = {k: v for k, v in d.items() if k != "QB-Norm DIS"}
        for b, (m, e) in d.items():
            if b in ("episode CSLS",):
                continue
            n[0 if m - e > 0 else (2 if m + e < 0 else 1)] += 1
    return n


def t5_values(families, groups=None):
    out = []
    for g, _, _, d in table5():
        if groups is None or g in groups:
            out += [d[f][0] for f in families]
    return out


def ablation(host, key, sh, metric="auroc"):
    return at(ABL, f"hosts / {host} / rows / {key}|B{sh} / {metric}")


AM, AT = "frozen FEAT (mini)", "frozen FEAT tiered (BGR)"


def abl_diff(host, a, b, sh, metric="auroc"):
    return ablation(host, a, sh, metric) - ablation(host, b, sh, metric)


def retention(host, key, sh):
    return 100 * abl_diff(host, key, "COSF", sh) / abl_diff(host, "CCP-A (registered)", "COSF", sh)


def dnpg_abl(key, sh):
    import numpy as np
    return float(np.mean([at(ABL, f"hosts / DNPG seeds 0-2 (mini) / per_seed / seed{s}_{sh}shot / {key}|B / auroc") for s in range(3)]))


def dnpg_retention(key, sh):
    return 100 * (dnpg_abl(f"H_fix({key})", sh) - dnpg_abl("D", sh)) / (dnpg_abl("H_fix(CCP-A (registered))", sh) - dnpg_abl("D", sh))


def fig_rows(rel):
    return csv_rows(rel)


def fig2_bins(sh):
    rows = [r for r in fig_rows(FIG2) if r["panel"] == "b" and r["series"].startswith(f"{sh}shot ")]
    kn = [(float(r["value"]), float(r["ci95"])) for r in rows if " known " in r["series"]]
    un = [(float(r["value"]), float(r["ci95"])) for r in rows if " unknown " in r["series"]]
    return kn, un


def fig2_higher(sh):
    kn, un = fig2_bins(sh)
    return sum(1 for (a, e), (b, f) in zip(kn, un) if b - f > a + e)


def sens(series, sh, factor):
    rows = [r for r in fig_rows(FIG_SENS) if r["panel"] == "b" and r["series"].startswith(series) and r["shot"] == str(sh) and r["K_or_m"] == factor]
    assert len(rows) == 1, (series, sh, factor)
    return float(rows[0]["valB_auroc"])


PAIRED = "DNPG H_fix paired diff vs m=1"


def variants_test(v, cellset):
    return [c[v]["paired_auroc_vs_reference"]["mean"] for k, c in J(VAR_T)["cells_test"].items() if k in cellset and v in c]


VAR_MINI = {f"{h}|{sh}" for h in ("frozen FEAT mini", "SnaTCHer-F mini", "DNPG mini") for sh in SH}


def valb_gain(ds, b, sh):
    return at(FM_VALB, f"paired_vs_cosf / {ds}_{b}|{sh} / CCP-A / mean")


def geom(names, key):
    g = J(GEOM)["backbones"]
    return [g[n][key] for n in names]


CLIP_BANKS = [f"CLIP ViT-B/16 {d}" for d in FM_DATA]
DINO_BANKS = [f"DINOv2 ViT-{s}/14 {d}" for s in "SBL" for d in FM_DATA]
DINO_CUB = [f"DINOv2 ViT-{s}/14 cub" for s in "SBL"]
RESNET_BANKS = ["frozen FEAT mini", "frozen FEAT tiered", "RFS CIFAR-FS"]
LOSS_CELLS = lambda: [c for c in cells(DINO) if h1(*c) < 0]  # noqa: E731
SCG_VAL = lambda ds, b, sh: at(FM_SEL, f"scg_valA / {ds}|{b}|{sh} / std_gap")  # noqa: E731
FIG3_POINTS = "scripts/figures/fig3_scg_gain.py (plotted data, checked against the paper's figure data)"


def fig3_points():
    from scripts.figures.fig3_scg_gain import points
    return points()


# ------------------------------------------------------------------------------------------------------------------- claims
EPIS = "manifests/<dataset>/ (episode layout: 5 known and 5 unknown classes, 15 queries per class)"
DATA = "docs/data.md (dataset splits and sizes; ccp/data/)"
REG = "docs/preregistration/ (registered setting)"
NOTATION = "notation"
CLAIMS: list[tuple[str, str, list[Spec]]] = []


def C(file, anchor, *specs):
    CLAIMS.append((file, anchor, list(specs)))


ALTERNATIVES: list[list[tuple[str, str, list[Spec]]]] = []


def fuzzy(anchor: str):
    """A regex for the anchor in which every number may be any number."""
    import re as _re
    return _re.compile(_re.sub(r"\d+(?:\\\.\d+)?", lambda m: r"\d+(?:\.\d+)?", _re.escape(anchor)))


def ONE_OF(options):
    """Claims of which exactly one anchor must be in the paper (a sentence and its replacement listed in PAPER_EDITS.md)."""
    ALTERNATIVES.append(options)


A, S1, S2, S3, S4, S5, S6 = ("manuscript.tex", "sections/01_introduction.tex", "sections/02_related.tex", "sections/03_method.tex",
                             "sections/04_experiments.tex", "sections/05_results.tex", "sections/06_conclusion.tex")
DINO_LOSS = F(lambda: len(LOSS_CELLS()), f"count of DINOv2 configurations with H1 mean < 0 (`{FM_H}` › `H1 / cells`)")
DINO_N = F(lambda: len(cells(DINO)), "number of DINOv2 configurations (3 hosts x 4 datasets x 2 shots)")
H2_RHO = R(FM_H, "H2 / rho")
H2_WITHIN = R(FM_H, "H2 within-group / stat_mean_within_rho")
N_CELLS = R(FM_H, "H1 / n_cells")
CL2N_BEST = F(lambda: sum(1 for c in cells() if fm_row(*c, "CL2N") >= max([fm_row(*c, "CCP-A")] + list(fm_baselines(*c).values()))),
              f"count of configurations where CL2N's test AUROC is the highest among CCP-A and the eight inductive baselines (`{FM}` › `summaries`)")

# abstract
C(A, "evaluate 10 hosts", S("Section 4.2: hosts (i)-(x); configs/hosts/", 10))
C(A, "by 0.58--4.44 points", F(lambda: min(mini_gains()), "minimum of the mini-ImageNet ResNet-12 gains (Section 5.1)"),
  F(lambda: max(mini_gains()), "maximum of the same gains"))
C(A, "Gains are 1.84--5.50 points", F(lambda: min(cub_gains()), f"minimum CUB gain (`{DER}` › `cub / <host>|<shot> / auroc / host / mean`)"),
  F(lambda: max(cub_gains()), "maximum of the same gains"))
C(A, "0.86--8.03 points in all eight CLIP", F(lambda: min(h1(*c) for c in cells(("clip_b16",))), f"minimum CLIP H1 gain (`{FM_H}` › `H1 / cells`)"),
  F(lambda: max(h1(*c) for c in cells(("clip_b16",))), "maximum CLIP H1 gain"))
C(A, "reduces AUROC in 16 of 24 DINOv2", DINO_LOSS, DINO_N)
C(A, "is 0.78 overall and 0.70 on average", H2_RHO, H2_WITHIN)
# introduction
C(S1, "it reduces AUROC in 16 of 24 configurations", DINO_LOSS, DINO_N)
C(S1, "Over 32 foundation-model configurations, validation SCG has a Spearman rank correlation of 0.78 with test AUROC gains and 0.70 on",
  N_CELLS, H2_RHO, H2_WITHIN)
# related work
C(S2, "standard deviations to 1.", T("notation: AS-norm with unit standard deviations"))
C(S2, "uses the 1,000 nearest impostors", T("number reported by the cited work (VoxWatch)"))
C(S2, "Setting this weight to 0.5 recovers", T("notation: NNN weight that equals CCP-A's prototype term (Eq. 4)"))
# method
C(S3, "samples $N=5$ known classes", S(EPIS, 5))
C(S3, r"$S\in\{1,5\}$", S(EPIS, 1), S(EPIS, 5))
C(S3, "With 15 queries per class, each episode contains 75 known and 75 unknown queries", S(EPIS, 15), S(EPIS, 75), S(EPIS, 75))
C(S3, r"\in\{1,\dots,N\}", T(NOTATION))
C(S3, "For $S=1$, $p_k=z(x_{k,1})$", T(NOTATION), T(NOTATION))
C(S3, "We use $K=1000$ in every test round", S(f"ccp.cohort.REGISTERED_K; {REG}", 1000))
C(S3, "deviations fixed to 1. Equivalently", T(NOTATION))
C(S3, "(all 448,695 on tieredImageNet", S(DATA, 448695))
C(S3, r"We selected $\lambda=1$", F(lambda: float(at(VAR_B, "chosen / CCP-QW").split("lambda=")[1].split("/")[0]),
                                  f"`{VAR_B}` › `chosen / CCP-QW` (lambda)"))
C(S3, "within 0.02 points of fixed-weight fusion", F(lambda: max(abs(v) for v in hfix_vs_standardization()),
                                                    "maximum |episode-wise standardization - H_fix| validation AUROC difference, DNPG seeds 0-2, "
                                                    f"valA and valB (`{DN_HFIX}` › `summaries`)", rule="le"))
C(S3, "into 10 bins using the deciles", S("ccp.scg (10 decile bins; docs/protocol.md)", 10))
C(S3, "or with $s_j=0$, are excluded", T(NOTATION))
# experiments: datasets
C(S4, r"100 ImageNet classes~\cite{deng2009imagenet}, 600 images per class at $84\times84$ pixels, split 64/16/20",
  S(DATA, 100), S(DATA, 600), S(DATA, 84), S(DATA, 64), S(DATA, 16), S(DATA, 20))
C(S4, r"608 ImageNet classes in 34 super-categories, split 351/97/160 with super-categories separated across partitions; stored images are $84",
  S(DATA, 608), S(DATA, 34), S(DATA, 351), S(DATA, 97), S(DATA, 160), S(DATA, 84))
C(S4, r"the 100 CIFAR-100 classes~\cite{krizhevsky2009cifar}, 600 images per class at $32\times32$ pixels, split 64/16/20",
  S(DATA, 100), S(DATA, 600), S(DATA, 32), S(DATA, 64), S(DATA, 16), S(DATA, 20))
C(S4, "11,788 images of 200 bird species", S(DATA, 11788), S(DATA, 200))
C(S4, "gives 100/50/50 classes and 5,903/2,927/2,958 images", S(DATA, 100), S(DATA, 50), S(DATA, 50), S(DATA, 5903), S(DATA, 2927),
  S(DATA, 2958))
C(S4, r"to the hosts' $84\times84$ input", S("configs/hosts/ (input size of the ResNet-12 hosts)", 84))
C(S4, "two validation sets of 1,000 episodes each and 600 test episodes", S("manifests/ (valA, valB: 1,000 episodes; test: 600)", 1000),
  S("manifests/", 600))
C(S4, "a third set of 1,000 new validation episodes", S("manifests/ (valC)", 1000))
C(S4, "We evaluate 10 hosts in three groups", S("Section 4.2 hosts (i)-(x); configs/hosts/", 10))
C(S4, "Their stored epochs, 66 and 81", S("configs/hosts/tane_pre_mini.yaml (stored_epoch)", 66), S("configs/hosts/tane_pre_tiered.yaml (stored_epoch)", 81))
C(S4, "checkpoint of seed 0. With COSF, it reaches 71.14\\% / 86.64\\% accuracy at 1 / 5 shots; the published 71.5\\% / 86.0\\%",
  S("configs/hosts/cifarfs_rfs_seed0.yaml", 0), R(FR, "tables / A / CIFAR-FS / 1 / rows / COSF / acc"), R(FR, "tables / A / CIFAR-FS / 5 / rows / COSF / acc"),
  S(EPIS, 1), S(EPIS, 5), T("number reported by the cited work (RFS)"), T("number reported by the cited work (RFS)"))
# Section 4.2, SnaTCHer-F fidelity: the 2.0.0 check written by `python scripts/fidelity/snatcher_fidelity.py --record`
CHK = "results/checks/snatcher_fidelity_end_to_end.json"
CK = lambda path, **kw: R(CHK, f"results / {path}", **kw)  # noqa: E731
C(S4, r"On 50 validation episodes per shot, our implementation reproduces the scores of the original authors' evaluation code to within "
      r"$7.6\times10^{-6}$ when both use the same encoder outputs. End to end, where the encoder processes the images in different batches, "
      r"the scores differ by at most $7.6\times10^{-6}$ at 1 shot and $3.2\times10^{-3}$ at 5 shots, the mean AUROC by at most "
      r"$4\times10^{-4}$ points, and the accuracy is identical",
  CK("1shot / n_episodes"),
  F(lambda: max(at(CHK, f"results / {sh}shot / same_encoder_batch / max_abs_diff / S") for sh in SH),
    f"maximum of `{CHK}` › `results / <shot> / same_encoder_batch / max_abs_diff / S`", rule="sci",
    note="results/checks/snatcher_fidelity_end_to_end.json is a check of this release (2.0.0), written by `python "
         "scripts/fidelity/snatcher_fidelity.py --record` on validation episodes and listed with its sha256 in results/PROVENANCE.md, "
         "\"2.0.0 checks\"; it reproduces the recorded fidelity values of results/recorded/validation/snatcher.json and adds the AUROC "
         "differences."),
  CK("1shot / end_to_end / max_abs_diff / S", rule="sci"), S(EPIS, 1), CK("5shot / end_to_end / max_abs_diff / S", rule="sci"), S(EPIS, 5),
  F(lambda: max(abs(at(CHK, f"results / {sh}shot / end_to_end / S_auroc_mean_abs_diff")) for sh in SH),
    f"largest |mean AUROC difference| end to end, `{CHK}` › `results / <shot> / end_to_end / S_auroc_mean_abs_diff` (rounded to the "
    "printed digit; also a bound)", rule="sci"))
C(S4, "We select checkpoints on 600 validation episodes per epoch", S("configs/hosts/dnpg_mini_seed*.yaml; docs/checkpoints.md (DNPG selection)", 600))
C(S4, "Our reproduction is 0.46 / 0.52 AUROC points below",
  *[F(lambda sh=sh: published("mini-ImageNet", "DNPG", f"auroc_{sh}shot") - r2(at(DNPG, f"seed_summary / {sh}shot_T / D / auroc / mean")),
      f"published DNPG AUROC (`{PUB}`) minus `{DNPG}` › `seed_summary / {sh}shot_T / D / auroc / mean`", rule="pd") for sh in SH])
C(S4, "512-dimensional projected", T("model property (CLIP ViT-B/16 projection width)"))
C(S4, r"upsampled to $224\times224$", S("configs/hosts/*clip*, *dinov2* (input size)", 224))
C(S4, "CLIP's 400 million", T("number reported by the cited work (CLIP)"))
C(S4, "the bank contains all 38,400 base images. For tieredImageNet, we use a fixed random subset of 38,400 of its 448,695 base images",
  S("configs/hosts/ (bank)", 38400), S("configs/hosts/frozen_feat_tiered.yaml (bank subset)", 38400), S(DATA, 448695))
C(S4, "the bank contains all 5,903 base images", S(DATA, 5903))
C(S4, r"$m_K(q)$, $K=1000$, with greater", S(f"configs/baselines; {REG}", 1000))
C(S4, "cosine to the 10th, 100th, or 1,000th bank neighbor", S("ccp/baselines.py (kNN grid)", 10), S("ccp/baselines.py (kNN grid)", 100),
  S("ccp/baselines.py (kNN grid)", 1000))
C(S4, r"$\Sigma_\delta=(1-\delta)C", T(NOTATION))
C(S4, r"$\delta\in\{0.1,0.5,1.0\}$", S("ccp/baselines.py (RMD grid)", 0.1), S("ccp/baselines.py (RMD grid)", 0.5), S("ccp/baselines.py (RMD grid)", 1.0))
C(S4, "top-100 or top-1,000 bank neighbors", S("ccp/baselines.py (NNN grid)", 100), S("ccp/baselines.py (NNN grid)", 1000))
C(S4, r"$\alpha\in\{0.25,\allowbreak 0.5,\allowbreak 0.75,\allowbreak 1\}$", *[S("ccp/baselines.py (NNN grid)", v) for v in (0.25, 0.5, 0.75, 1)])
C(S4, "temperatures of 10, 20, and 50", *[S("ccp/baselines.py (QB-Norm grid)", v) for v in (10, 20, 50)])
C(S4, "$K=10$ with episode neighborhoods", S("ccp/baselines.py (episode CSLS)", 10))
C(S4, r"we report 95\% confidence intervals calculated as $\pm1.96", S("ccp.metrics.ci95", 95), S("ccp.metrics.ci95", 1.96))
C(S4, "over all 150 queries", S(EPIS, 150))
C(S4, r"at least 95\% of known queries, corresponding to the 72nd smallest score among 75 known", S("ccp.metrics.fpr95", 95),
  S("ccp.metrics.fpr95", 72), S(EPIS, 75))
C(S4, r"the mean difference and its 95\% confidence interval", S("ccp.metrics.ci95", 95))
C(S4, "Of the 22 paired ResNet comparisons", F(lambda: len(resnet_comparisons()), "number of host-dataset-shot comparisons with and without "
                                              "CCP-A (Tables 2-4: 7 hosts on mini / tiered x 2 shots, RFS x 2, 3 CUB hosts x 2)"))
C(S4, "on tieredImageNet ($p=0.27$ / $0.063$) have $p\\le3\\times10^{-5}$, below the Bonferroni threshold of $0.05/22$",
  R(DER, "resnet_vs_cosf / TANE tiered|1 / auroc / wilcoxon_p"), R(DER, "resnet_vs_cosf / TANE tiered|5 / auroc / wilcoxon_p"),
  F(lambda: max(p for k, p in resnet_comparisons().items() if not k.startswith("TANE tiered")),
    "maximum Wilcoxon p of the other 20 comparisons (`resnet_vs_cosf`, `tables / C` and `tables / A, B` of the recorded files)", rule="le"),
  S("Bonferroni level", 0.05), F(lambda: len(resnet_comparisons()), "number of comparisons"))
C(S4, "comprises 32 host--dataset--shot", N_CELLS)
C(S4, "over 32 one-sided Wilcoxon tests", N_CELLS)
C(S4, "permutation tests (100,000 permutations)", R(FM_H, "H2 / n_perm"))
C(S4, "its 5-shot valA gain was $+0.24$ points against the required $+0.50$", S(EPIS, 5),
  F(lambda: r2(at(DN_EVAL, "seed0_max_acc_5shot / cands_A / CCP-A / auroc")) - r2(at(DN_EVAL, "seed0_max_acc_5shot / DNPG_own / A / auroc")),
    f"`{DN_EVAL}` › `seed0_max_acc_5shot / cands_A / CCP-A / auroc` minus `DNPG_own / A / auroc` (rounded means)", rule="pd",
    note="The unrounded difference is 0.2347 (+0.23 at two decimals); +0.24 is the difference of the rounded means, 86.43 - 86.19, and is "
         "the value of the registration record (results/recorded/validation/reports/dnpg_seed0.md: \"the 5-shot valA gain (+0.24) is below "
         "+0.5\"). The paper keeps +0.24."),
  S(f"{REG} (DNPG replacement criterion)", 0.5))
C(S4, "PyTorch 2.14 (CUDA 13.0) under Windows 11", F(lambda: float(at(ENV, "torch")[:4]), f"`{ENV}` › `torch`"),
  F(lambda: float(at(ENV, "torch_cuda")), f"`{ENV}` › `torch_cuda`"), S(f"`{ENV}` › `os`", 11))
C(S4, "(24 GB), an Intel Core i9 processor, and 128 GB of RAM", F(lambda: round(int(at(ENV, "gpu_driver_memory").split(", ")[2].split()[0]) / 1024),
                                                                  f"`{ENV}` › `gpu_driver_memory`"), S("workstation specification (README)", 128))
C(S4, "fixed batches of 500 images and bank similarities in fixed chunks of 2,048 rows", S("ccp/hosts/base.py BATCH", 500),
  S("ccp/scores/base.py (top-K chunks)", 2048))
# results 5.1
C(S5, "given as 1-shot / 5-shot", S(EPIS, 1), S(EPIS, 5))
FROZEN_GAINS = [R(DER, f"resnet_vs_cosf / {h}|{sh} / auroc / mean") for h in ("frozen FEAT mini", "TANE mini", "FRN mini") for sh in SH]
C(S5, "+2.48 / +4.24 points on frozen FEAT, +2.54 / +4.44 on TANE pre-trained and +0.98 / +2.23 on FRN", *FROZEN_GAINS)
C(S5, "At 1 shot, classification accuracy increases by 1.62 points on frozen FEAT and 1.39 points on TANE pre-trained, while FRN's accuracy "
      "changes by less than 0.05 points", S(EPIS, 1), RD("frozen_FEAT", 1, "acc"), RD("TANE_DNPG_pretrain", 1, "acc"), RD("FRN", 1, "acc", rule="lt", mag=True))
C(S5, "COSF and COS differ by at most 0.18 AUROC points",
  F(lambda: max(abs(rounded_diff(h, sh, "auroc", "COSF", "COS")) for h in ("frozen_FEAT", "TANE_DNPG_pretrain", "FRN") for sh in SH),
    f"maximum |COSF - COS| AUROC of the frozen mini-ImageNet hosts, from the rounded means of Table 2 (`{MINI}` › `rows`, `frn_seeds`)"))
SNR = lambda sh, row: R(SNATCHER, f"rows / {sh}shot_T / {row} / auroc")  # noqa: E731
SNP = lambda sh, d, f: R(SNATCHER, f"paired / {sh}shot_T / {d} / {f}")  # noqa: E731
C(S5, "SnaTCHer-F's AUROC from 69.88 / 77.28 to 71.55 / 79.38. The paired gains are $+1.67\\pm0.21$ / $+2.10\\pm0.17$, with improvements in "
      "75\\% / 84\\% of episodes ($p=4.0\\times10^{-45}$ / $1.7\\times10^{-75}$). Plain-cosine fusion $K_{\\mathrm{fix}}$ reaches 70.14 / 77.49 "
      "AUROC, $1.41\\pm0.14$ / $1.89\\pm0.12$ points below $H_{\\mathrm{fix}}$. CCP-A alone reaches 72.08 / 80.07",
  SNR(1, "S"), SNR(5, "S"), SNR(1, "H_fix"), SNR(5, "H_fix"), SNP(1, "Hfix_minus_S", "mean"), SNP(1, "Hfix_minus_S", "ci95"),
  SNP(5, "Hfix_minus_S", "mean"), SNP(5, "Hfix_minus_S", "ci95"),
  R(FR, "tables / C / SnaTCHer-F mini / 1 / paired / host / H_fix / auroc / frac_gain", rule="pct"),
  R(FR, "tables / C / SnaTCHer-F mini / 5 / paired / host / H_fix / auroc / frac_gain", rule="pct"),
  R(FR, "tables / C / SnaTCHer-F mini / 1 / paired / host / H_fix / auroc / wilcoxon_p", rule="sci"),
  R(FR, "tables / C / SnaTCHer-F mini / 5 / paired / host / H_fix / auroc / wilcoxon_p", rule="sci"),
  SNR(1, "K_fix"), SNR(5, "K_fix"), SNP(1, "Hfix_minus_Kfix", "mean"), SNP(1, "Hfix_minus_Kfix", "ci95"), SNP(5, "Hfix_minus_Kfix", "mean"),
  SNP(5, "Hfix_minus_Kfix", "ci95"), SNR(1, "CCP-A"), SNR(5, "CCP-A"))
DNR = lambda sh, row: R(DNPG, f"seed_summary / {sh}shot_T / {row} / auroc / mean")  # noqa: E731
DNC = lambda sh, row, f: R(FR, f"tables / C / DNPG mini / {sh} / paired / host / {row} / auroc / {f}", rule="pct" if f == "frac_gain" else  # noqa: E731
                                 ("sci" if f == "wilcoxon_p" else "="))
C(S5, "DNPG's AUROC from 73.72 / 83.12 to 74.30 / 83.85. The paired gains are $+0.58\\pm0.13$ / $+0.73\\pm0.12$, with improvements in 66\\% / "
      "67\\% of episodes ($p=1.1\\times10^{-16}$ / $1.1\\times10^{-26}$). Both plain-cosine fusion $K_{\\mathrm{fix}}$ (73.31 / 81.91) and "
      "standalone CCP-A (73.51 / 82.21)",
  DNR(1, "D"), DNR(5, "D"), DNR(1, "H_fix"), DNR(5, "H_fix"), DNC(1, "H_fix", "mean"), DNC(1, "H_fix", "ci95"), DNC(5, "H_fix", "mean"),
  DNC(5, "H_fix", "ci95"), DNC(1, "H_fix", "frac_gain"), DNC(5, "H_fix", "frac_gain"), DNC(1, "H_fix", "wilcoxon_p"), DNC(5, "H_fix", "wilcoxon_p"),
  DNR(1, "K_fix"), DNR(5, "K_fix"), DNR(1, "CCP-A"), DNR(5, "CCP-A"))
C(S5, "significant only at 5 shots ($p=0.19$ / $1.5\\times10^{-11}$)", S(EPIS, 5), DNC(1, "CCP-A", "wilcoxon_p"), DNC(5, "CCP-A", "wilcoxon_p"))
C(S5, "frozen FEAT gains 2.36 / 3.74 OSCR points and reduces FPR95 by 3.04 / 5.67 points, while DNPG fusion gains 0.71 / 0.84 OSCR points and "
      "reduces FPR95 by 0.57 / 0.61 points",
  R(DER, "resnet_vs_cosf / frozen FEAT mini|1 / oscr / mean"), R(DER, "resnet_vs_cosf / frozen FEAT mini|5 / oscr / mean"),
  R(DER, "resnet_vs_cosf / frozen FEAT mini|1 / fpr95 / mean", mag=True), R(DER, "resnet_vs_cosf / frozen FEAT mini|5 / fpr95 / mean", mag=True),
  R(FR, "tables / C / DNPG mini / 1 / paired / host / H_fix / oscr / mean"), R(FR, "tables / C / DNPG mini / 5 / paired / host / H_fix / oscr / mean"),
  R(FR, "tables / C / DNPG mini / 1 / paired / host / H_fix / fpr95 / mean", mag=True),
  R(FR, "tables / C / DNPG mini / 5 / paired / host / H_fix / fpr95 / mean", mag=True))
C(S5, "Over the 10 host--shot configurations, the AUROC gain shows no detectable monotonic association with the AUROC of the reference score "
      "($\\rho=0.01$)", F(lambda: len(mini_gains()), "number of mini-ImageNet host-shot configurations (5 hosts x 2 shots)"),
  F(lambda: float([r for r in fig_rows(FIG_HOST) if r["host"] == "spearman_mini_10_points"][0]["test_gain"]),
    f"`{FIG_HOST}` (row spearman_mini_10_points; the data of the repository figure gain_vs_host)"))
C(S5, "remains 2.23 / 0.74 points below HDFA",
  *[F(lambda sh=sh: published("mini-ImageNet", "HDFA", f"auroc_{sh}shot") - r2(at(DNPG, f"seed_summary / {sh}shot_T / H_fix / auroc / mean")),
      f"published HDFA AUROC (`{PUB}`) minus `{DNPG}` › `seed_summary / {sh}shot_T / H_fix / auroc / mean`", rule="pd") for sh in SH])
# results 5.2
C(S5, "CCP-A improves frozen FEAT's AUROC by +0.42 / +0.36 points and its classification accuracy by 0.65 / 0.06 points. TANE pre-trained gains "
      "+0.08 / +0.26 AUROC points, and neither gain is statistically significant ($p=0.27$ / $0.063$)",
  *[R(DER, f"resnet_vs_cosf / frozen FEAT tiered|{sh} / auroc / mean") for sh in SH],
  *[RD("frozen_FEAT_tiered", sh, "acc") for sh in SH],
  *[R(DER, f"resnet_vs_cosf / TANE tiered|{sh} / auroc / mean") for sh in SH],
  *[R(DER, f"resnet_vs_cosf / TANE tiered|{sh} / auroc / wilcoxon_p") for sh in SH])
TFC = lambda sh: r2(at(TIERED, f"rows / frozen_FEAT_tiered / {sh}shot / CCP-A / auroc"))  # noqa: E731
C(S5, "It remains 1.14 / 0.15 points below published DNPG and about 2.0 / 1.3 points below OPP+HDE",
  *[F(lambda sh=sh: published("tieredImageNet", "DNPG", f"auroc_{sh}shot") - TFC(sh),
      f"published DNPG tieredImageNet AUROC (`{PUB}`) minus `{TIERED}` › `rows / frozen_FEAT_tiered / {sh}shot / CCP-A / auroc`", rule="pd")
    for sh in SH],
  *[F(lambda sh=sh: published("tieredImageNet", "OPP+HDE", f"auroc_{sh}shot") - TFC(sh),
      f"published OPP+HDE tieredImageNet AUROC (`{PUB}`) minus the same frozen FEAT CCP-A AUROC") for sh in SH])
C(S5, "DNPG's 5-shot fusion gain was $+0.37\\pm0.11$ points, below the required $+0.50$ points, and for SnaTCHer-F, the gain of "
      "$H_{\\mathrm{fix}}$ over $K_{\\mathrm{fix}}$ was $+0.20$ / $+0.23$ points, below the required $+0.30$ points",
  S(EPIS, 5), R(DN_TIERED, "summaries / seed0_max_acc_5shot_B / paired / Hfix_minus_D / mean"),
  R(DN_TIERED, "summaries / seed0_max_acc_5shot_B / paired / Hfix_minus_D / ci95"), S(f"{REG} (DNPG tieredImageNet criterion)", 0.5),
  R(SN_VAL, "val / rules / tiered / values / S2 / 1"), R(SN_VAL, "val / rules / tiered / values / S2 / 5"), S(f"{REG} (SnaTCHer-F criterion)", 0.3))
FRA = lambda sh, row, m="auroc", f=None: R(FR, f"tables / A / CIFAR-FS / {sh} / " + (f"rows / {row} / {m}" if f is None else  # noqa: E731
                                                                                 f"paired / host / {row} / {m} / {f}"), rule="pct" if f == "frac_gain" else "=")
C(S5, "CCP-A increases RFS AUROC from 76.42 / 85.10 to 76.89 / 86.62. The paired gains are $+0.47\\pm0.20$ / $+1.52\\pm0.21$ points, with "
      "improvements in 57\\% / 68\\% of episodes. Classification accuracy changes by +0.17 / $-0.56$ points, whereas no other frozen host in "
      "\\cref{tab:mini,tab:tiered-cifar} loses more than 0.05 accuracy points",
  FRA(1, "host"), FRA(5, "host"), FRA(1, "CCP-A"), FRA(5, "CCP-A"), FRA(1, "CCP-A", f="mean"), FRA(1, "CCP-A", f="ci95"), FRA(5, "CCP-A", f="mean"),
  FRA(5, "CCP-A", f="ci95"), FRA(1, "CCP-A", f="frac_gain"), FRA(5, "CCP-A", f="frac_gain"), FRA(1, "CCP-A", "acc", "mean"),
  FRA(5, "CCP-A", "acc", "mean"),
  F(lambda: max(-rounded_diff(h, sh, "acc") for h in ("frozen_FEAT", "TANE_DNPG_pretrain", "FRN", "frozen_FEAT_tiered", "TANE_tiered_pretrain")
                for sh in SH),
    f"largest accuracy loss of CCP-A against COSF of the other frozen hosts, from the rounded means of Tables 2-3 (`{MINI}`, `{TIERED}` › `rows`)",
    rule="le"))
C(S5, "Our 1-shot AUROC trails GEL, SEMAN-G, and the SnaTCHer-F result reported by GEL's authors, whereas our 5-shot result is within 0.25 "
      "points of GEL without its pixel-wise branch (86.84)", S(EPIS, 1), S(EPIS, 5),
  F(lambda: sota("GEL, Ours w/o Pix", "auroc_5shot", "2304.11855") - r2(at(FR, "tables / A / CIFAR-FS / 5 / rows / CCP-A / auroc")),
    f"GEL w/o Pix 5-shot AUROC (`{SOTA}`) minus `{FR}` › `tables / A / CIFAR-FS / 5 / rows / CCP-A / auroc`", rule="le"),
  F(lambda: sota("GEL, Ours w/o Pix", "auroc_5shot", "2304.11855"), f"`{SOTA}` (GEL, Ours w/o Pix, 5-shot AUROC; GEL Table 1)"))
C(S5, "CCP gains +1.90 / +3.79 AUROC points for frozen FEAT, +1.84 / +3.93 for SnaTCHer-F and +5.13 / +5.50 for DNPG. Between 74\\% and 94\\% of "
      "episodes improve in each configuration, and CCP outperforms all eight inductive baselines with 95\\%",
  *[R(DER, f"cub / {h}|{sh} / auroc / host / mean") for h in ("frozen FEAT mini", "SnaTCHer-F mini", "DNPG mini") for sh in SH],
  F(lambda: 100 * min(cub_gains("frac_gain")), f"minimum fraction of improved episodes (`{DER}` › `cub / <host>|<shot> / auroc / host / frac_gain`)"),
  F(lambda: 100 * max(cub_gains("frac_gain")), "maximum of the same fractions"), S("ccp.metrics.ci95", 95))
# results 5.3
C(S5, "Over 16 host--dataset--shot configurations and eight inductive baselines, its paired differences are positive with 95\\% confidence "
      "intervals entirely above zero in 121 of 128 comparisons",
  F(lambda: len(table5()), "number of Table 5 configurations (scripts/make_tables.py baseline_cells)"), S("ccp.metrics.ci95", 95),
  F(lambda: t5_counts()[0], "Table 5 count of CIs above zero (QB-Norm IS / DIS counted once)"), F(lambda: sum(t5_counts()), "Table 5 comparisons"))
NEG = {"CL2N|tieredImageNet|1": ("frozen FEAT tiered", 1), "RMD|mini-ImageNet|5": ("SnaTCHer-F mini", 5), "QB-Norm IS|CIFAR-FS|1": None}


def t5_cell(group, label, sh, fam_):
    for g, lab, s, d in table5():
        if g == group and lab == label and s == sh:
            return d[fam_]
    raise KeyError((group, label, sh))


C(S5, "frozen FEAT on tieredImageNet at 1 shot ($-0.65\\pm0.31$), RMD for SnaTCHer-F on mini-ImageNet at 5 shots ($-0.96\\pm0.15$) and QB-Norm "
      "for RFS on CIFAR-FS at 1 shot ($-0.36\\pm0.20$)",
  S(EPIS, 1), F(lambda: t5_cell("tieredImageNet", "Frozen FEAT", 1, "CL2N")[0], f"`{'results/baselines/test.json'}` › `paired_test / frozen FEAT tiered / 1 / CL2N / minus_CCP` (negated)"),
  F(lambda: t5_cell("tieredImageNet", "Frozen FEAT", 1, "CL2N")[1], "its 95% CI"),
  S(EPIS, 5), F(lambda: t5_cell("mini-ImageNet", "SnaTCHer-F", 5, "RMD")[0], "`results/baselines/test.json` › `paired_test / SnaTCHer-F mini / 5 / fused[RMD/...] / minus_CCP` (negated)"),
  F(lambda: t5_cell("mini-ImageNet", "SnaTCHer-F", 5, "RMD")[1], "its 95% CI"),
  S(EPIS, 1), F(lambda: t5_cell("CIFAR-FS", "RFS", 1, "QB-Norm IS")[0], f"`{FR}` › `tables / A / CIFAR-FS / 1 / paired / CCP-A / QB-Norm IS/... / auroc` (negated)"),
  F(lambda: t5_cell("CIFAR-FS", "RFS", 1, "QB-Norm IS")[1], "its 95% CI"))


def below_reference():
    """Query cohort term and kNN fused with the reference score, against the reference: (baseline - reference) AUROC, mini / tiered / CIFAR-FS."""
    bt = J("results/baselines/test.json")["paired_test"]
    out = [v["minus_host"]["mean"] for h in bt for sh in SH for k, v in bt[h][str(sh)].items() if fam(k) in ("query cohort term", "kNN")]
    out += [v["auroc"]["mean"] for sh in SH for k, v in at(FR, f"tables / A / CIFAR-FS / {sh} / paired / host").items()
            if fam(k) in ("query cohort term", "kNN")]
    return out


C(S5, "they fall below it in all 20 applicable", F(lambda: sum(1 for v in below_reference() if v < 0), "count of negative (baseline - reference) "
                                                  "AUROC differences, query cohort term and kNN (`results/baselines/test.json` › `minus_host`; "
                                                  f"`{FR}` › `tables / A / CIFAR-FS / <shot> / paired / host`)"))
C(S5, "Over all 16 configurations, they are 1.3--15.2 AUROC points below CCP",
  F(lambda: len(table5()), "number of Table 5 configurations"),
  F(lambda: min(t5_values(("query cohort term", "kNN"))), "minimum Table 5 difference CCP - (query cohort term, kNN)"),
  F(lambda: max(t5_values(("query cohort term", "kNN"))), "maximum of the same differences"))
C(S5, "NNN is below CCP in all 16 configurations and QB-Norm in 15",
  F(lambda: sum(1 for v in t5_values(("NNN",)) if v > 0), "Table 5 count of positive CCP - NNN"),
  F(lambda: sum(1 for v in t5_values(("QB-Norm IS",)) if v > 0), "Table 5 count of positive CCP - QB-Norm"))
C(S5, "in every configuration, by 0.6--3.5 AUROC points",
  F(lambda: min(-r2(v) for v in t5_values(("episode CSLS",))), "minimum episode CSLS - CCP, as printed in Table 5 (two decimals)"),
  F(lambda: max(-r2(v) for v in t5_values(("episode CSLS",))), "maximum episode CSLS - CCP, as printed in Table 5 (two decimals)"))
# results 5.4
C(S5, "two shot settings give 32 configurations", N_CELLS)
C(S5, "AUROC gains range from 0.86 to 8.03 points", F(lambda: min(h1(*c) for c in cells(("clip_b16",))), "minimum CLIP H1 gain"),
  F(lambda: max(h1(*c) for c in cells(("clip_b16",))), "maximum CLIP H1 gain"))
C(S5, "the largest FPR95 reduction, 22 points, on mini-ImageNet at 5 shots",
  F(lambda: -min(r2(fm_row(*c, "CCP-A", "fpr95")) - r2(fm_row(*c, "COSF", "fpr95")) for c in cells(("clip_b16",))),
    f"largest CLIP FPR95 reduction (Table 6: difference of rounded means, `{FM}` › `summaries / <cell>|CCP-A, COSF / fpr95`)"), S(EPIS, 5))
CL_RMD = [("tiered", 1), ("tiered", 5), ("cifarfs", 1), ("cifarfs", 5)]
C(S5, "CL2N or RMD outperforms it by 1.6--2.4 points",
  F(lambda: min(-at(DER, f"foundation / {d}|clip_b16|{sh} / CCPA_minus / {fm_best(d, 'clip_b16', sh)} / mean") for d, sh in CL_RMD),
    f"minimum best baseline - CCP-A, CLIP on tieredImageNet and CIFAR-FS (`{DER}` › `foundation / <cell> / CCPA_minus`)"),
  F(lambda: max(-at(DER, f"foundation / {d}|clip_b16|{sh} / CCPA_minus / {fm_best(d, 'clip_b16', sh)} / mean") for d, sh in CL_RMD),
    "maximum of the same differences"))
C(S5, "COSF reaches AUROC of 88--99", F(lambda: min(fm_row(*c, "COSF") for c in cells(DINO)), f"minimum DINOv2 COSF AUROC (`{FM}` › `summaries`)"),
  F(lambda: max(fm_row(*c, "COSF") for c in cells(DINO)), "maximum DINOv2 COSF AUROC"))
C(S5, "The losses occur in 16 of the 24 configurations, including all 12 on mini-ImageNet and tieredImageNet", DINO_LOSS, DINO_N,
  F(lambda: sum(1 for c in LOSS_CELLS() if c[0] in ("mini", "tiered")), "count of losses on mini-ImageNet and tieredImageNet"))
C(S5, "They range from 0.05 to 1.34 points, and 15 of the corresponding 95\\%",
  F(lambda: min(-h1(*c) for c in LOSS_CELLS()), "smallest DINOv2 loss (H1 mean)"), F(lambda: max(-h1(*c) for c in LOSS_CELLS()), "largest DINOv2 loss"),
  F(lambda: sum(1 for c in LOSS_CELLS() if at(FM_H, f"H1 / cells / {'|'.join(map(str, c))} / mean") + at(FM_H, f"H1 / cells / {'|'.join(map(str, c))} / ci95") < 0),
    f"count of losses whose 95% CI lies below zero (`{FM_H}` › `H1 / cells / <cell> / ci95`)"), S("ccp.metrics.ci95", 95))
C(S5, "OSCR also falls in all 16 configurations, and FPR95 rises in 15",
  F(lambda: sum(1 for c in LOSS_CELLS() if r2(fm_row(*c, "CCP-A", "oscr")) < r2(fm_row(*c, "COSF", "oscr"))), "count of OSCR decreases (Table 6 rule)"),
  F(lambda: sum(1 for c in LOSS_CELLS() if r2(fm_row(*c, "CCP-A", "fpr95")) > r2(fm_row(*c, "COSF", "fpr95"))), "count of FPR95 increases (Table 6 rule)"))
C(S5, "improves all six DINOv2 configurations by +0.07 to +1.67 points, and ViT-L on CIFAR-FS, with gains of +0.89 / +0.46 points",
  F(lambda: min(h1("cub", b, sh) for b in DINO for sh in SH), "minimum DINOv2 CUB gain (H1)"),
  F(lambda: max(h1("cub", b, sh) for b in DINO for sh in SH), "maximum DINOv2 CUB gain (H1)"),
  R(FM_H, "H1 / cells / cifarfs|dinov2_l14|1 / mean"), R(FM_H, "H1 / cells / cifarfs|dinov2_l14|5 / mean"))
C(S5, "In the 16 configurations with losses, CL2N changes AUROC by $-0.11$ to $+0.63$ points",
  F(lambda: len(LOSS_CELLS()), "count of DINOv2 losses"),
  F(lambda: min(at(DER, f"foundation / {'|'.join(map(str, c))} / minus_COSF / CL2N / mean") for c in LOSS_CELLS()),
    f"minimum CL2N - COSF over those configurations (`{DER}` › `foundation / <cell> / minus_COSF / CL2N`)"),
  F(lambda: max(at(DER, f"foundation / {'|'.join(map(str, c))} / minus_COSF / CL2N / mean") for c in LOSS_CELLS()), "maximum of the same"))


def exceeds_all(c):
    return all(at(DER, f"foundation / {'|'.join(map(str, c))} / CCPA_minus / {k} / mean") > 0 for k in fm_baselines(*c))


C(S5, "The Holm-corrected H1 test is significant in all 16 configurations with positive gains; the other 16 configurations show losses. CCP-A "
      "exceeds all eight inductive baselines in nine configurations: CLIP on mini-ImageNet and CUB, and DINOv2 on CUB except ViT-L at 5 shots, "
      "where RMD exceeds CCP-A by 0.01 points. CCP-QW is below CCP-A in 29 of the 32 configurations; the exceptions are CLIP on CUB at both shot "
      "settings and DINOv2 ViT-S on CUB at 5 shots",
  R(FM_H, "H1 / n_holm_significant"), F(lambda: sum(1 for c in cells() if h1(*c) < 0), "count of negative H1 means"),
  S(EPIS, 5), F(lambda: -at(DER, f"foundation / cub|dinov2_l14|5 / CCPA_minus / {fm_best('cub', 'dinov2_l14', 5)} / mean"),
                f"`{DER}` › `foundation / cub|dinov2_l14|5 / CCPA_minus / RMD/... / mean` (negated)"),
  F(lambda: sum(1 for c in cells() if at(DER, f"foundation / {'|'.join(map(str, c))} / CCPA_minus / CCP-QW / mean") > 0),
    f"count of CCP-A - CCP-QW > 0 (`{DER}` › `foundation / <cell> / CCPA_minus / CCP-QW`)"), N_CELLS, S(EPIS, 5))
C(S5, "The valB and test gains differ by less than 1 point in 26 of the 32 configurations",
  S("threshold stated in the text", 1),
  F(lambda: sum(1 for c in cells() if abs(valb_gain(*c) - h1(*c)) < 1), f"count of |valB gain - test gain| < 1 (`{FM_VALB}` › `paired_vs_cosf`; H1)"),
  N_CELLS)
C(S5, "valB gains of $+0.64$ / $+0.87$ become test losses of $-0.60$ / $-0.28$, and for ViT-B at 5 shots, a valB gain of $+0.35$ becomes a test "
      "loss of $-0.27$",
  *[F(lambda sh=sh: valb_gain("cifarfs", "dinov2_s14", sh), f"`{FM_VALB}` › `paired_vs_cosf / cifarfs_dinov2_s14|{sh} / CCP-A / mean`") for sh in SH],
  R(FM_H, "H1 / cells / cifarfs|dinov2_s14|1 / mean"), R(FM_H, "H1 / cells / cifarfs|dinov2_s14|5 / mean"), S(EPIS, 5),
  F(lambda: valb_gain("cifarfs", "dinov2_b14", 5), f"`{FM_VALB}` › `paired_vs_cosf / cifarfs_dinov2_b14|5 / CCP-A / mean`"),
  R(FM_H, "H1 / cells / cifarfs|dinov2_b14|5 / mean"))
G224 = lambda b, sh: F(lambda: fm_row("cub_eval224", b, sh, "CCP-A") - fm_row("cub_eval224", b, sh, "COSF"),  # noqa: E731
                       f"`{FM}` › `summaries / cub_eval224_{b}|T|{sh}|CCP-A` minus `...|COSF` (auroc)")
C(S5, r"directly at $224\times224$ gives AUROC gains of +5.59 / +3.76 points for CLIP and +0.49 / +0.22 for DINOv2 ViT-B. Upsampling the stored "
      r"$84\times84$ images gives +5.90 / +4.59 and +0.61 / +0.29",
  S("configs/hosts/cub_*_eval224.yaml", 224), G224("clip_b16", 1), G224("clip_b16", 5), G224("dinov2_b14", 1), G224("dinov2_b14", 5),
  S(DATA, 84), R(FM_H, "H1 / cells / cub|clip_b16|1 / mean"), R(FM_H, "H1 / cells / cub|clip_b16|5 / mean"),
  R(FM_H, "H1 / cells / cub|dinov2_b14|1 / mean"), R(FM_H, "H1 / cells / cub|dinov2_b14|5 / mean"))
# results 5.5 (figure captions and text)
FIGN = "scripts/figures/fig2_mechanism.py"
C(S5, "(a)~Queries from 40 episodes at 1 shot", S(f"{FIGN} N_PLOT", 40), S(EPIS, 1))
C(S5, "Mean $m_K(q)$ with 95\\% confidence intervals in 10 decile bins of raw maximum cosine, at 1 shot (top) and 5 shots", S("ccp.metrics.ci95", 95),
  S(f"{FIGN} N_BINS", 10), S(EPIS, 1), S(EPIS, 5))
C(S5, "The 95\\% confidence intervals do not overlap in 9 of 10 bins at 1 shot and in all 10 bins at 5 shots. At 1 shot, the mean cohort "
      "affinities of unknown and known queries are 0.710 and 0.677 in the lowest similarity bin and converge to 0.748 and 0.749 in the highest",
  S("ccp.metrics.ci95", 95), F(lambda: fig2_higher(1), f"count of non-overlapping bins (`{FIG2}`, panel b)"), S(f"{FIGN} N_BINS", 10), S(EPIS, 1),
  S(f"{FIGN} N_BINS", 10), S(EPIS, 5), S(EPIS, 1),
  F(lambda: fig2_bins(1)[1][0][0], f"`{FIG2}` › `b, 1shot unknown bin1`"), F(lambda: fig2_bins(1)[0][0][0], f"`{FIG2}` › `b, 1shot known bin1`"),
  F(lambda: fig2_bins(1)[1][-1][0], f"`{FIG2}` › `b, 1shot unknown bin10`"), F(lambda: fig2_bins(1)[0][-1][0], f"`{FIG2}` › `b, 1shot known bin10`"))
C(S5, "prototype-only correction mainly raises 1-shot accuracy", S(EPIS, 1))
C(S5, "Over the 32 configurations, its Spearman", N_CELLS)
C(S5, "for 8 ResNet-12 and 32 foundation-model configurations. Error bars are 95\\% confidence intervals over 600 test episodes",
  F(lambda: sum(1 for p in fig3_points() if p["round"] == "final"), f"ResNet-12 points of Figure 3 ({FIG3_POINTS})"),
  F(lambda: sum(1 for p in fig3_points() if p["round"] == "foundation"), "foundation-model points of Figure 3"), S("ccp.metrics.ci95", 95),
  S("manifests/ (test: 600 episodes)", 600))
C(S5, "gray rings mark the 4 frozen FEAT configurations", F(lambda: sum(1 for p in fig3_points() if p["round"] == "final" and p["dataset"] in ("mini", "tiered")
                                                                       and p["backbone"].startswith("frozen FEAT")), "late-SCG points of Figure 3"))
C(S5, "its Spearman rank correlation with test AUROC gains is $\\rho=0.78$", H2_RHO)
C(S5, "Test SCG correlates more strongly ($\\rho=0.91$; H3)", R(FM_H, "H3 / rho"))
C(S5, "Both one-sided permutation tests reach $p=10^{-5}$, the resolution floor for 100,000 permutations",
  F(lambda: max(at(FM_H, "H2 / p_one_sided"), at(FM_H, "H3 / p_one_sided")), f"`{FM_H}` › `H2 / p_one_sided`, `H3 / p_one_sided` (1 / (n_perm + 1))",
    rule="sci"), R(FM_H, "H2 / n_perm"))
C(S5, "the mean correlations are 0.70 for validation SCG and 0.68 for test SCG ($p=2.6\\times10^{-4}$ and $2.7\\times10^{-4}$",
  H2_WITHIN, R(FM_H, "H3 within-group / stat_mean_within_rho"), R(FM_H, "H2 within-group / p_one_sided", rule="sci"),
  R(FM_H, "H3 within-group / p_one_sided", rule="sci"))


def scg_rule(lo=None, hi=None):
    """The printed threshold if the statement holds: every DINOv2 configuration with validation SCG < lo loses (lo), every foundation
    configuration with validation SCG > hi gains (hi)."""
    if lo is not None:
        ok = all(h1(*c) < 0 for c in cells(DINO) if SCG_VAL(*c) < lo)
        return lo if ok else float("nan")
    ok = all(h1(*c) > 0 for c in cells() if SCG_VAL(*c) > hi)
    return hi if ok else float("nan")


C(S5, "with validation SCG below 0.5 show AUROC losses, all foundation-model configurations with SCG above 0.72 show gains",
  F(lambda: scg_rule(lo=0.5), f"statement checked: every DINOv2 configuration with valA SCG < 0.5 has a negative H1 mean (`{FM_SEL}` › `scg_valA`)"),
  F(lambda: scg_rule(hi=0.72), "statement checked: every foundation configuration with valA SCG > 0.72 has a positive H1 mean"))
C(S5, "On mini-ImageNet at 1 shot, DINOv2 ViT-L has the lowest SCG of the three model sizes (0.01) and the largest AUROC loss ($-0.88$)",
  S(EPIS, 1), R(FM_SEL, "scg_valA / mini|dinov2_l14|1 / std_gap"), R(FM_H, "H1 / cells / mini|dinov2_l14|1 / mean"))
C(S5, "including five with SCG between 0.1 and 0.4",
  F(lambda: resnet_scg_band(0.1, 0.4, 5), f"statement checked: five ResNet-12 points of Figure 3 have validation SCG between 0.1 and 0.4 ({FIG3_POINTS})"),
  S("bounds stated in the text", 0.4))
C(S5, "SCG falls from 0.30 / 0.53 on valA to 0.20 / 0.40 on test, and the AUROC gain falls from +1.35 / +1.41 to +0.42 / +0.36",
  R(FR_SCG_VAL, "scg / tiered|1 / std_gap"), R(FR_SCG_VAL, "scg / tiered|5 / std_gap"), R(FR, "scg / tiered|1 / std_gap"), R(FR, "scg / tiered|5 / std_gap"),
  R(FR_SCG_VAL, "scg / tiered|1 / ccpa_minus_cosf_auroc"), R(FR_SCG_VAL, "scg / tiered|5 / ccpa_minus_cosf_auroc"),
  R(FR, "scg / tiered|1 / ccpa_minus_cosf_auroc"), R(FR, "scg / tiered|5 / ccpa_minus_cosf_auroc"))
FRB = lambda h, sh, row: R(FR, f"tables / B / {h} / {sh} / rows / {row} / auroc")  # noqa: E731
DNB = lambda sh, row: F(lambda: at(FR, f"tables / B / DNPG mini / {sh} / rows / {row} / auroc / mean"),  # noqa: E731
                        f"`{FR}` › `tables / B / DNPG mini / {sh} / rows / {row} / auroc / mean`")
C(S5, "frozen FEAT's SCG on the same test queries from 0.21 / 0.43 to 0.34 / 0.62. Its AUROC gain rises from +1.90 / +3.79 to +4.73 / +7.28 points. "
      "With this target bank, AUROC reaches 63.45 / 70.13 for frozen FEAT and 65.13 / 73.10 for DNPG at $K=1000$",
  R(FR, "scg / CUB|1 / std_gap"), R(FR, "scg / CUB|5 / std_gap"), R(FR, "scg / CUB@target|1 / std_gap"), R(FR, "scg / CUB@target|5 / std_gap"),
  R(FR, "scg / CUB|1 / ccpa_minus_cosf_auroc"), R(FR, "scg / CUB|5 / ccpa_minus_cosf_auroc"),
  R(FR, "scg / CUB@target|1 / ccpa_minus_cosf_auroc"), R(FR, "scg / CUB@target|5 / ccpa_minus_cosf_auroc"),
  FRB("frozen FEAT mini", 1, "CCP-A, target bank"), FRB("frozen FEAT mini", 5, "CCP-A, target bank"),
  DNB(1, "H_fix, target bank"), DNB(5, "H_fix, target bank"), S(f"ccp.cohort.REGISTERED_K; {REG}", 1000))
C(S5, "Because the CUB bank contains only 5,903 images, we also evaluate $K=154$, which approximately matches the fraction selected by $K=1000$ from "
      "the 38,400-image mini-ImageNet bank. At $K=154$, AUROC decreases to 62.41 / 68.62 for frozen FEAT and 64.36 / 72.42 for DNPG",
  S(DATA, 5903), F(lambda: round(1000 * 5903 / 38400), "1000 x 5,903 / 38,400 (configs/final_round_target_banks.yaml)"),
  S(f"ccp.cohort.REGISTERED_K; {REG}", 1000), S("configs/hosts/ (bank)", 38400), S("configs/final_round_target_banks.yaml", 154),
  FRB("frozen FEAT mini", 1, "CCP-A, target bank, K=154"), FRB("frozen FEAT mini", 5, "CCP-A, target bank, K=154"),
  DNB(1, "H_fix, target bank, K=154"), DNB(5, "H_fix, target bank, K=154"))
MB = lambda b, sh: F(lambda: fm_row("cub@minibank", b, sh, "CCP-A") - fm_row("cub@minibank", b, sh, "COSF"),  # noqa: E731
                     f"`{FM}` › `summaries / cub@minibank_{b}|T|{sh}|CCP-A` minus `...|COSF` (auroc)")
C(S5, "reduces CLIP's gains on CUB from +5.90 / +4.59 to +3.32 / +1.46 points. DINOv2 gains then range from $-0.23$ to $+0.28$ points",
  R(FM_H, "H1 / cells / cub|clip_b16|1 / mean"), R(FM_H, "H1 / cells / cub|clip_b16|5 / mean"), MB("clip_b16", 1), MB("clip_b16", 5),
  F(lambda: min(fm_row("cub@minibank", b, sh, "CCP-A") - fm_row("cub@minibank", b, sh, "COSF") for b in DINO for sh in SH),
    f"minimum DINOv2 CUB gain with the mini-ImageNet bank (`{FM}` › `summaries / cub@minibank_<host>`)"),
  F(lambda: max(fm_row("cub@minibank", b, sh, "CCP-A") - fm_row("cub@minibank", b, sh, "COSF") for b in DINO for sh in SH), "maximum of the same"))
GM = lambda names, key, f, what: F(lambda: f(geom(names, key)), f"{what} of `{GEOM}` › `backbones / <bank> / {key}`")  # noqa: E731
C(S5, "CLIP banks are highly concentrated (0.80--0.90), and their cohort statistics largely follow the global direction ($R^2=0.60$--$0.85$)",
  GM(CLIP_BANKS, "mean_cos_to_bank_mean", min, "minimum (CLIP banks)"), GM(CLIP_BANKS, "mean_cos_to_bank_mean", max, "maximum (CLIP banks)"),
  GM(CLIP_BANKS, "r2_mK_on_cos_to_mean", min, "minimum (CLIP banks)"), GM(CLIP_BANKS, "r2_mK_on_cos_to_mean", max, "maximum (CLIP banks)"))
C(S5, "ResNet-12 banks show a much weaker association ($R^2=0.04$--$0.18$)",
  GM(RESNET_BANKS, "r2_mK_on_cos_to_mean", min, "minimum (ResNet-12 banks)"), GM(RESNET_BANKS, "r2_mK_on_cos_to_mean", max, "maximum (ResNet-12 banks)"))
C(S5, "DINOv2 banks are less concentrated (0.13--0.33), yet on CUB their cohort statistics follow the global direction ($R^2=0.70$--$0.80$)",
  GM(DINO_BANKS, "mean_cos_to_bank_mean", min, "minimum (DINOv2 banks)"), GM(DINO_BANKS, "mean_cos_to_bank_mean", max, "maximum (DINOv2 banks)"),
  GM(DINO_CUB, "r2_mK_on_cos_to_mean", min, "minimum (DINOv2 CUB banks)"), GM(DINO_CUB, "r2_mK_on_cos_to_mean", max, "maximum (DINOv2 CUB banks)"))
C(S5, "(mini-ImageNet valB, frozen FEAT, 1 shot; at most one query per episode)", S(EPIS, 1))
C(S5, "Rows 1--3: unknown queries with the largest increases; row 4: the known query with the largest decrease; row 5:",
  T("row numbers of Figure 4"), T("row numbers of Figure 4"), T("row numbers of Figure 4"), T("row numbers of Figure 4"))
# results 5.6
AB = lambda h, k, sh, m="auroc": R(ABL, f"hosts / {h} / rows / {k}|B{sh} / {m}")  # noqa: E731
AD = lambda h, a, b, sh, m="auroc": F(lambda: r2(ablation(h, a, sh, m)) - r2(ablation(h, b, sh, m)),  # noqa: E731
                                       f"`{ABL}` › `hosts / {h} / rows`: {a} minus {b} (valB {sh}-shot {m}; rounded means of Table 7)", rule="pd")
C(S5, "COSF reaches AUROC of 74.35 / 82.64 on mini-ImageNet and 68.85 / 77.19 on tieredImageNet, and the registered CCP-A configuration improves "
      "these by +1.74 / +2.54 and +1.03 / +0.84",
  AB(AM, "COSF", 1), AB(AM, "COSF", 5), AB(AT, "COSF", 1), AB(AT, "COSF", 5),
  AD(AM, "CCP-A (registered)", "COSF", 1), AD(AM, "CCP-A (registered)", "COSF", 5), AD(AT, "CCP-A (registered)", "COSF", 1),
  AD(AT, "CCP-A (registered)", "COSF", 5))
C(S5, "peaks at the largest finite neighborhood tested, $K=3000$. Compared with the registered $K=1000$, $K=3000$ adds 0.64 / 0.63 points on "
      "mini-ImageNet and 0.15 / 0.13 points on tieredImageNet. Smaller neighborhoods ($K=100$ and $K=300$)",
  S("ablation grid (scripts/fidelity/ablations)", 3000), S(f"ccp.cohort.REGISTERED_K; {REG}", 1000), S("ablation grid", 3000),
  AD(AM, "K=3000", "CCP-A (registered)", 1), AD(AM, "K=3000", "CCP-A (registered)", 5), AD(AT, "K=3000", "CCP-A (registered)", 1),
  AD(AT, "K=3000", "CCP-A (registered)", 5), S("ablation grid", 100), S("ablation grid", 300))
C(S5, "retains 83\\% / 55\\% of CCP-A's gain on mini-ImageNet but only about half / none of it on tieredImageNet. For DNPG fusion, the whole-bank "
      "mean retains just 32\\% / 15\\%",
  F(lambda: retention(AM, "K=all", 1), f"(K=all - COSF) / (CCP-A - COSF), `{ABL}` (frozen FEAT mini, valB 1-shot AUROC)"),
  F(lambda: retention(AM, "K=all", 5), "the same at 5 shots"),
  F(lambda: dnpg_retention("K=all", 1), f"(H_fix(K=all) - D) / (H_fix(K=1000) - D), `{ABL}` › `hosts / DNPG seeds 0-2 (mini) / per_seed` (seed means)"),
  F(lambda: dnpg_retention("K=all", 5), "the same at 5 shots"))
C(S5, "a bank of 64 class-mean vectors retains 88\\% / 78\\% of the AUROC gain of the 38,400 image embeddings on mini-ImageNet. The corresponding "
      "retention on tieredImageNet is 70\\% / none",
  S(DATA, 64), F(lambda: retention(AM, "class means (matched top-k)", 1), f"(class means, matched - COSF) / (CCP-A - COSF), `{ABL}`"),
  F(lambda: retention(AM, "class means (matched top-k)", 5), "the same at 5 shots"), S("configs/hosts/ (bank)", 38400),
  F(lambda: retention(AT, "class means (matched top-k)", 1), "the same on tieredImageNet, 1 shot"))
C(S5, "on mini-ImageNet (76.14 / 86.22) and at 5 shots on tieredImageNet (78.68)", AB(AM, "query-side only", 1), AB(AM, "query-side only", 5),
  S(EPIS, 5), AB(AT, "query-side only", 5))
C(S5, "increases 1-shot classification accuracy from 68.55\\% to 70.12\\% on mini-ImageNet and from 63.77\\% to 65.15\\% on tieredImageNet",
  S(EPIS, 1), AB(AM, "COSF", 1, "acc"), AB(AM, "prototype-side only", 1, "acc"), AB(AT, "COSF", 1, "acc"), AB(AT, "prototype-side only", 1, "acc"))
C(S5, "at 5 shots on tieredImageNet it falls below COSF. Both variants subtract their cohort term with a coefficient of 1", S(EPIS, 5), T(NOTATION))
C(S5, "by 0.57 / 0.73 AUROC points on mini-ImageNet and 0.20 / 0.23 points on tieredImageNet",
  AD(AM, "CCP-S (mean + std)", "CCP-A (registered)", 1), AD(AM, "CCP-S (mean + std)", "CCP-A (registered)", 5),
  AD(AT, "CCP-S (mean + std)", "CCP-A (registered)", 1), AD(AT, "CCP-S (mean + std)", "CCP-A (registered)", 5))


def ccps_minus_ccpa(sh):
    m, t = J(MINI)["rows"], J(TIERED)["rows"]
    import numpy as np
    out = [m[h][f"{sh}shot"]["CCP-S"]["auroc"] - m[h][f"{sh}shot"]["CCP-A"]["auroc"] for h in ("frozen_FEAT", "TANE_DNPG_pretrain")]
    out.append(float(np.mean([m[f"FRN_seed{s}"][f"{sh}shot"]["CCP-S"]["auroc"] - m[f"FRN_seed{s}"][f"{sh}shot"]["CCP-A"]["auroc"] for s in range(3)])))
    out += [t[h][f"{sh}shot"]["CCP-S"]["auroc"] - t[h][f"{sh}shot"]["CCP-A"]["auroc"] for h in ("frozen_FEAT_tiered", "TANE_tiered_pretrain")]
    return out


C(S5, "it outperforms CCP-A at 1 shot by 0.30--0.59 AUROC points, and at 5 shots the differences range from $-0.47$ to $+0.46$ points",
  S(EPIS, 1), F(lambda: min(ccps_minus_ccpa(1)), f"minimum CCP-S - CCP-A test AUROC of the five frozen hosts of rounds (i)-(ii) (`{MINI}`, `{TIERED}` › `rows`; FRN: mean of seeds)"),
  F(lambda: max(ccps_minus_ccpa(1)), "maximum of the same (1 shot)"), S(EPIS, 5), F(lambda: min(ccps_minus_ccpa(5)), "minimum (5 shots)"),
  F(lambda: max(ccps_minus_ccpa(5)), "maximum (5 shots)"))
C(S5, "reduces AUROC by 0.31 / 0.23 points on mini-ImageNet and 0.18 / 0.13 points on tieredImageNet",
  AD(AM, "CCP-A (registered)", "flip off", 1), AD(AM, "CCP-A (registered)", "flip off", 5), AD(AT, "CCP-A (registered)", "flip off", 1),
  AD(AT, "CCP-A (registered)", "flip off", 5))
SD = lambda sh, f: F(lambda: sens(PAIRED, sh, f), f"`{FIG_SENS}` › `b, {PAIRED}, {sh}-shot, factor {f}` (data of the repository figure sensitivity)", mag=True)  # noqa: E731
C(S5, "by factors between 1 and 2 changes validation AUROC by at most 0.05 points. Reducing the factor to 0.25 lowers AUROC by 0.47 / 0.58 points, "
      "and raising it to 4 lowers AUROC by 0.18 / 0.30 points",
  S("fusion-weight factor grid (docs/preregistration/variants.md)", 1), S("fusion-weight factor grid", 2),
  F(lambda: max(abs(sens(PAIRED, sh, "2.0")) for sh in SH), f"maximum |AUROC(factor 2) - AUROC(factor 1)| (`{FIG_SENS}`)", rule="le"),
  S("fusion-weight factor grid", 0.25), SD(1, "0.25"), SD(5, "0.25"), S("fusion-weight factor grid", 4), SD(1, "4.0"), SD(5, "4.0"))
CRIT = "ccp.variants (registered criteria; docs/preregistration/variants.md)"
C(S5, r"$\gamma=0.25$ selected on valA", F(lambda: float(at(VAR_B, "chosen / shared-cohort correction").split("gamma=")[1]),
                                         f"`{VAR_B}` › `chosen / shared-cohort correction`"))
C(S5, "(i) a mean valB AUROC gain of at least $+0.30$ points over CCP-A, or over $H_{\\mathrm{fix}}$ for trained hosts; (ii) no configuration below "
      "$-0.20$ points relative to its reference; (iii) an accuracy loss on frozen hosts of at most 0.20 points; and (iv) on valC, a mean gain of at "
      "least $+0.15$ points with no configuration below $-0.20$ points",
  S(CRIT, 0.3), S(CRIT, -0.2), S(CRIT, 0.2), S(CRIT, 0.15), S(CRIT, -0.2))
C(S5, "CCP-QW improves on its reference by +0.23 to +1.96 points in the six mini-ImageNet configurations but reduces AUROC by 0.50 / 0.68 points",
  F(lambda: min(variants_test("CCP-QW", VAR_MINI)), f"minimum CCP-QW test difference, mini-ImageNet (`{VAR_T}` › `cells_test / <cell> / CCP-QW / paired_auroc_vs_reference`)"),
  F(lambda: max(variants_test("CCP-QW", VAR_MINI)), "maximum of the same"),
  R(VAR_T, "cells_test / frozen FEAT tiered|1 / CCP-QW / paired_auroc_vs_reference / mean", mag=True),
  R(VAR_T, "cells_test / frozen FEAT tiered|5 / CCP-QW / paired_auroc_vs_reference / mean", mag=True))
C(S5, "building the 38,400-image mini-ImageNet cohort bank takes 7.7\\,s, and its 640-dimensional embeddings occupy 46.9\\,MB",
  R(COST, "cohort / n_base"), R(COST, "cohort / embed_seconds_both_views"), T("model property (ResNet-12 feature width)"),
  R(COST, "cohort / ccpA_bank_MB_fp16"))
C(S5, "the 150 queries and five prototypes to the bank and selecting the top $K$ for each takes 0.75 / 0.80\\,ms",
  S(EPIS, 150), R(COST, "per_episode / 1shot / CCP-A / ms_per_episode"), R(COST, "per_episode / 5shot / CCP-A / ms_per_episode"))
SN_CCP = "CCP-A on SnaTCHer-F features (extra over SnaTCHer-F: flip-view forward + CCP-A)"
DN_CCP = "CCP-A on DNPG features (extra over DNPG: flip-view forward + CCP-A)"
OVER = lambda: [at(COST_SN, f"per_episode / {sh}shot / {SN_CCP} / ms_per_episode") for sh in SH] + \
    [at(COST, f"per_episode / {sh}shot / {DN_CCP} / ms_per_episode") for sh in SH]  # noqa: E731
C(S5, "the total overhead is 31.5 to 35.6\\,ms", F(lambda: min(OVER()), f"minimum added time per episode (`{COST_SN}`, `{COST}` › `per_episode`)"),
  F(lambda: max(OVER()), "maximum of the same"))
C(S5, "adds about 11.0 million trainable parameters, and its training takes on average 4.47 GPU-hours",
  R(COST, "dnpg_params / new_modules_trainable", scale=1e-6),
  F(lambda: sum(at(COST, "training_hours / mini_per_seed_measured").values()) / 2, f"mean of `{COST}` › `training_hours / mini_per_seed_measured`"))
# discussion
C(S5, "CCP outperforms CL2N in 13 of the 16 ResNet configurations, with 95\\%",
  F(lambda: sum(1 for *_, d in table5() if d["CL2N"][0] - d["CL2N"][1] > 0), "Table 5 count of CL2N comparisons with CIs above zero"),
  F(lambda: len(table5()), "number of Table 5 configurations"), S("ccp.metrics.ci95", 95))
C(S5, "CL2N is the best training-free method in 16 of the 32", CL2N_BEST, N_CELLS)
C(S5, "$K=3000$ on validation, CCP-S at 1 shot on frozen hosts", S("ablation grid", 3000), S(EPIS, 1))
C(S5, "The fixed $K=1000$ also selects different fractions of the cohort banks, approximately 2.6\\% of each 38,400-image bank and 17\\% of the "
      "5,903-image CUB bank used for foundation models. The $K=154$",
  S(f"ccp.cohort.REGISTERED_K; {REG}", 1000), F(lambda: 100 * 1000 / 38400, "100 x 1000 / 38,400"), S("configs/hosts/ (bank)", 38400),
  F(lambda: 100 * 1000 / 5903, "100 x 1000 / 5,903"), S(DATA, 5903), S("configs/final_round_target_banks.yaml", 154))
# conclusion
C(S6, "AUROC falls in 16 of 24 DINOv2", DINO_LOSS, DINO_N)
C(S6, "Spearman rank correlation of 0.78 with foundation-model", H2_RHO)
C(S6, "CL2N is the best training-free method in 16 of the 32", CL2N_BEST, N_CELLS)


# ------------------------------------------------------------------------------------------------------------------- helpers used above
def hfix_vs_standardization():
    """Validation AUROC of episode-wise standardization minus H_fix, DNPG seeds 0-2 (valA / valB)."""
    out = []
    for k, v in at(DN_HFIX, "summaries").items():
        out.append(v["episode-wise standardization"]["auroc"] - v["H_fix"]["auroc"])
    return out


def resnet_comparisons() -> dict:
    """Wilcoxon p of every ResNet-12 host-dataset-shot comparison with and without CCP-A (Tables 2-4)."""
    out = {}
    for h in ("frozen FEAT mini", "TANE mini", "FRN mini", "frozen FEAT tiered", "TANE tiered"):
        for sh in SH:
            out[f"{h}|{sh}"] = der_resnet(h, sh, "auroc", "wilcoxon_p")
    for h in ("SnaTCHer-F mini", "DNPG mini"):
        for sh in SH:
            out[f"{h}|{sh}"] = frc(h, sh, "H_fix", "auroc", "wilcoxon_p")
    for sh in SH:
        out[f"RFS|{sh}"] = at(FR, f"tables / A / CIFAR-FS / {sh} / paired / host / CCP-A / auroc / wilcoxon_p")
        for h, row in (("frozen FEAT mini", "CCP-A"), ("SnaTCHer-F mini", "H_fix"), ("DNPG mini", "H_fix")):
            out[f"CUB {h}|{sh}"] = at(FR, f"tables / B / {h} / {sh} / paired / host / {row} / auroc / wilcoxon_p")
    return out


# ------------------------------------------------------------------------------------------------------------------- matching
@dataclass
class Row:
    token: X.Token
    spec: Spec | None
    value: float | None = None
    ok: bool | None = None
    note: str = ""
    claim: str = ""
    extra: list = field(default_factory=list)


def decimals(text: str) -> int:
    m = re.search(r"\d+(?:\.(\d+))?", text.replace(",", ""))
    return len(m.group(1)) if m and m.group(1) else 0


def compare(tok: X.Token, spec: Spec, v: float) -> bool:
    import math
    p = tok.value
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    if spec.mag:
        v, p = abs(v), abs(p)
    rule = spec.rule
    if rule == "pct":
        v, rule = 100 * v, "="
    if rule == "sci":
        if "times" in tok.text:
            mant = tok.text.split("\\times")[0].strip("$- ")
            k = decimals(mant)
            return f"{v:.{k}e}" == f"{p:.{k}e}"
        return f"{v:.0e}" == f"{p:.0e}"
    if rule == "pd":
        return abs(p - v) <= 0.01 + 1e-9
    if rule == "le":
        return v <= p + 1e-12
    if rule == "lt":
        return v < p
    if rule == "ge":
        return v >= p - 1e-12
    if rule == "gt":
        return v > p
    d = decimals(tok.text.split("\\times")[0])
    return f"{v:.{d}f}" == f"{p:.{d}f}" or (round(v, d) == 0 and round(p, d) == 0)


def match() -> tuple[list[Row], list[str]]:
    toks = X.all_tokens()
    by_file: dict[str, list[X.Token]] = {}
    for t in toks:
        by_file.setdefault(t.file, []).append(t)
    rows = {id(t): Row(t, None) for t in toks}
    problems = []
    claims = [(f, an, sp, False) for f, an, sp in CLAIMS]
    for options in ALTERNATIVES:            # numbers inside these anchors are wildcards, so a changed number is reported as a difference
        present = [o for o in options if len(fuzzy(o[1]).findall((X.PAPER / o[0]).read_text(encoding="utf-8"))) == 1]
        if len(present) != 1:
            problems.append(f"{options[0][0]}: {len(present)} of the alternative sentences found (exactly one expected): {options[0][1][:60]!r}")
            continue
        claims.append((*present[0], True))
    for file, anchor, specs, wild in claims:
        raw = (X.PAPER / file).read_text(encoding="utf-8")
        hits = list(fuzzy(anchor).finditer(raw)) if wild else [m for m in [None] * raw.count(anchor)]
        if len(hits) != 1:
            problems.append(f"{file}: anchor found {len(hits)} times: {anchor[:80]!r}")
            continue
        a, end = (hits[0].start(), hits[0].end()) if wild else (raw.index(anchor), raw.index(anchor) + len(anchor))
        anchor = raw[a:end]
        inside = [t for t in by_file.get(file, []) if a <= t.pos < a + len(anchor)]
        if len(inside) != len(specs):
            problems.append(f"{file}: anchor holds {len(inside)} numbers {[t.text for t in inside]} but {len(specs)} sources: {anchor[:80]!r}")
            continue
        for t, s in zip(inside, specs):
            r = rows[id(t)]
            if r.spec is not None:
                problems.append(f"{file}:{t.line}: {t.text} claimed twice")
            r.spec, r.claim = s, anchor
            if s.fn is not None:
                try:
                    r.value = float(s.fn())
                except Exception as e:  # noqa: BLE001
                    r.ok, r.note = False, f"source error: {type(e).__name__}: {e}"
                    continue
                r.ok = compare(t, s, r.value)
            else:
                r.ok = True
    out = [rows[id(t)] for t in toks]
    seen_known = set()
    for r in out:
        known = next((k for k in KNOWN if k[0] == r.token.file and k[1] == r.token.text and k[2] in r.claim), None)
        if r.spec is None:
            problems.append(f"{r.token.file}:{r.token.line}: {r.token.text} has no source ({r.token.context[:90]})")
        elif r.ok is False and known:
            r.note = f"known difference: {KNOWN[known]}"
            seen_known.add(known)
        elif r.ok is False:
            problems.append(f"{r.token.file}:{r.token.line}: printed {r.token.text}, source gives {r.value!r} (rule {r.spec.rule}"
                            f"{', mag' if r.spec.mag else ''}) {r.note} [{r.spec.source[:100]}]")
    problems += [f"{k[0]}: known difference {k[1]!r} ({KNOWN[k]}) no longer differs or was not found: update KNOWN and PAPER_EDITS.md"
                 for k in KNOWN if k not in seen_known]
    return out, problems


# Numbers that differ from their source and are listed in PAPER_EDITS.md as required edits (never changed to match): (file, printed, part
# of the claim anchor). Temporary: remove an entry as soon as the paper is fixed, so that the check is strict again.
KNOWN: dict[tuple[str, str, str], str] = {}


def esc(s: str) -> str:
    return s.replace("|", r"\|").replace("\n", " ")


def fmt_value(v):
    if v is None:
        return ""
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
        return f"{v:.4g}"
    return f"{v:.6g}"


def render(rows: list[Row], problems: list[str]) -> str:
    kinds = {}
    for r in rows:
        k = r.spec.kind if r.spec else "unmatched"
        kinds[k] = kinds.get(k, 0) + 1
    L = ["# Paper numbers", "",
         "Generated by `python scripts/paper_numbers.py`; do not edit by hand. Every numeric token of the paper's abstract, sections and figure "
         "captions (paper/, pinned in configs/paper.yaml) is listed with its source. The tables are not listed: `python scripts/make_tables.py "
         "--check` regenerates Tables 1-8 byte for byte from the recorded results. Rules and source kinds are explained in the docstring of "
         "`scripts/paper_numbers.py`; `pd` is the paired-difference rule (printed difference of two rounded means, within 0.01).", "",
         f"{len(rows)} numbers: " + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())) + ". " +
         ("All match their sources" if not problems else f"**{len(problems)} problems** (listed at the end)") +
         (f", except {n_known} known difference(s) listed in PAPER_EDITS.md and marked **differs** below." if (n_known := sum(
             1 for r in rows if r.ok is False and r.note.startswith("known difference"))) else "."), ""]
    sections = {f: [] for f in X.FILES}
    notes: list[str] = []
    for r in rows:
        sections[r.token.file].append(r)
    titles = {"manuscript.tex": "Abstract"}
    for f in X.FILES:
        if not sections[f]:
            continue
        L += [f"## {titles.get(f, f)}", "", "| Line | Printed | Context | Kind | Source | Value | Rule |", "|---|---|---|---|---|---|---|"]
        for r in sections[f]:
            t = r.token
            ctx = esc(re.sub(r"\s+", " ", t.context))[:110]
            kind = r.spec.kind if r.spec else "**unmatched**"
            src = esc(r.spec.source) if r.spec else ""
            if r.spec and r.spec.note:
                notes.append(f"{f}:{t.line}, printed `{esc(t.text)}`: {r.spec.note}")
                src += f" (note {len(notes)})"
            rule = (r.spec.rule + (", mag" if r.spec.mag else "")) if r.spec and r.spec.fn else ""
            ok = "" if r.ok in (True, None) else (f" **differs** ({r.note})" if r.note else " **differs**")
            L.append(f"| {t.line} | `{esc(t.text)}` | {ctx} | {kind} | {src} | {fmt_value(r.value)}{ok} | {rule} |")
        L.append("")
    if notes:
        L += ["## Notes", ""] + [f"{i}. {n}" for i, n in enumerate(notes, 1)] + [""]
    if problems:
        L += ["## Problems", ""] + [f"- {esc(p)}" for p in problems] + [""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not X.PAPER.is_dir():
        print("paper/ missing: run `python scripts/paper_pin.py extract` (skipped)")
        sys.exit(3)
    rows, problems = match()
    s = render(rows, problems)
    for p in problems:
        print("PROBLEM", p)
    if a.check:
        same = OUT.exists() and OUT.read_text(encoding="utf-8") == s
        print(f"{len(rows)} numbers, {len(problems)} problems; docs/paper_numbers.md {'up to date' if same else 'differs from a fresh render'}")
        sys.exit(0 if same and not problems else 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(s, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}: {len(rows)} numbers, {len(problems)} problems")


if __name__ == "__main__":
    main()
