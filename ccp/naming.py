"""The paper's names and the names used in the registration documents and the recorded result files (docs/naming.md is generated from
this module). The recorded files and the registration documents are never edited, so their keys keep the original names; code, CLI, logs,
tables, figures and documentation use the paper's names, and ccp.recorded translates recorded keys on reading.

Score keys have the form FAMILY[/param=value]... (for example `B2-KNN/k=1000/sign=-`). A key is translated family by family and parameter
by parameter, keeping every value exactly as written (so `B3-RMD/lambda=1.0` <-> `RMD/delta=1.0`). Fused baseline rows `H[<key>]`
become `fused[<paper key>]`.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------------------------------------------- score families
# recorded family -> paper family (training-free baselines in the registered order 1-9, then the method variants)
FAMILY = {
    "B1-QTERM": "query cohort term",
    "B2-KNN": "kNN",
    "B3-RMD": "RMD",
    "B4-CL2N": "CL2N",
    "B5-ZN": "ZN",
    "B6-NNN": "NNN",
    "B7a-QB-IS": "QB-Norm IS",
    "B7b-QB-DIS": "QB-Norm DIS",
    "B8-TEEN": "TEEN",
    "B9-CSLS-episode": "episode CSLS",
    "V1-QW": "CCP-QW",
    "V2-ASK": "full AS-norm",
    "V3-SHC": "shared-cohort correction",
    "V4-FW": "fusion-weight factor",
}
# per recorded family: recorded parameter name -> paper parameter name (others unchanged)
PARAM = {"B3-RMD": {"lambda": "delta"}, "B7a-QB-IS": {"beta": "temperature"}, "B7b-QB-DIS": {"beta": "temperature"},
         "V4-FW": {"mu": "factor"}}
# per recorded family: whole-segment values (V4-FW/alone)
SEGMENT = {"V4-FW": {"alone": "CCP-A alone"}}
# baseline family groups as named in the registration (the selection is per group; QB-Norm IS and DIS are separate groups)
BASELINE_GROUP = {"B1 QTERM": "query cohort term", "B2 KNN": "kNN", "B3 RMD": "RMD", "B4 CL2N": "CL2N", "B5 ZN": "ZN", "B6 NNN": "NNN",
                  "B7a QB-IS": "QB-Norm IS", "B7b QB-DIS": "QB-Norm DIS", "B8 TEEN": "TEEN",
                  "B9 CSLS-episode (TRANSDUCTIVE REFERENCE)": "episode CSLS (transductive reference)"}
VARIANT_GROUP = {"V1 QW": "CCP-QW", "V2 ASK": "full AS-norm", "V3 SHC": "shared-cohort correction", "V4 FW": "fusion-weight factor"}
# single row keys of the recorded files that are not family keys
ROW = {
    "V1": "CCP-QW",                              # final round: u_QW with CCP-A's label (Eq. 6)
    "V1fused": "CCP-QW fused",                   # final round: D + w_QW u_QW on trained hosts
    "CCP-A@target": "CCP-A, target bank",
    "CCP-A@target-K154": "CCP-A, target bank, K=154",
    "H_fix@target": "H_fix, target bank",
    "H_fix@target-K154": "H_fix, target bank, K=154",
    "REF": "reference",                          # variants round: CCP-A (frozen hosts) or H_fix (trained hosts)
}
# CCP-A ablation rows (registry keys of release 1.5.0 and row keys of results/recorded/validation/ccp_ablations.json)
ABLATION = {
    "CCP-A/K=100": "CCP-A/K=100", "CCP-A/K=300": "CCP-A/K=300", "CCP-A/K=3000": "CCP-A/K=3000", "CCP-A/K=all": "CCP-A/K=all",
    "CCP-A/query-side": "CCP-A/query only", "CCP-A/prototype-side": "CCP-A/prototype only",
    "CCP-A/class-means-topk": "CCP-A/class means, matched fraction", "CCP-A/class-means-all": "CCP-A/class means, all",
    "CCP-A/flip-off": "CCP-A/single view",
}
# field names of the recorded round records that carry a registered name
FIELD = {
    "v1_is_ccp_a": "qw_is_ccp_a", "v1_status": "qw_status", "gates_valB": "criteria_valB", "gate_G4_valC": "criterion_iv_valC",
    "paired_auroc_vs_REF": "paired_auroc_vs_reference", "pass_G1_G3": "pass_i_iii", "valA_mean_auroc_all_variants": "valA_mean_auroc_all_settings",
    "V1_weight": "qw_weight", "variant_weights": "variant_weights", "recorded_REF_test": "recorded_reference_test",
}
# fused-row rule of the trained hosts: recorded "H" / "K" are the transductive episode-wise standardization rows
EPISODE_STANDARDIZATION = {"H": "episode-wise standardization", "K": "episode-wise standardization, COSF control"}

# ---------------------------------------------------------------------------------------------------------------- other names
HYPOTHESIS = {"H1": "H1", "H2b": "H2", "H2": "H3", "H2b-s": "H2 within-group", "H2-s": "H3 within-group",
              "H3": "model-size trend (exploratory)"}
HYPOTHESIS_TEXT = {
    "H1": "CCP-A improves AUROC over COSF (one-sided Wilcoxon per configuration, Holm over the 32 configurations)",
    "H2": "validation (valA) SCG is positively rank-correlated with the test AUROC gain (one-sided permutation Spearman)",
    "H3": "test SCG is positively rank-correlated with the test AUROC gain (descriptive: same episodes)",
    "H2 within-group": "H2 within the 8 dataset x shot groups (mean within-group Spearman, permutation within groups)",
    "H3 within-group": "H3 within the 8 dataset x shot groups",
    "model-size trend (exploratory)": "within DINOv2 S -> B -> L, does the gain fall as COSF AUROC rises (description only; Section 5.5.3)",
}
SCG_KEY = {"d1": "scg", "d1_valA": "scg_valA", "d1_valB": "scg_valB", "d1_test": "scg_test", "d1_valA_predictions": "scg_valA_predictions"}
SPLIT = {"valA": "valA", "valB": "valB", "valC": "valC", "tvA": "valA", "tvB": "valB"}   # tieredImageNet: tvA / tvB are its valA / valB
CRITERION = {"G1": "(i)", "G2": "(ii)", "G3": "(iii)", "G4": "(iv)"}                     # variants-round acceptance criteria
# test rounds in the order they ran: (paper numeral, registered name, token, test date)
ROUNDS = [
    ("(i)", "MINI", "eaa3aab9d353da1e", "2026-09-25"),
    ("(ii)", "TIERED", "fd048685b138b1f9", "2026-09-27"),
    ("(iii)", "MINI-DNPG", "01fcac4666048f3e", "2026-09-28"),
    ("(iv)", "MINI-SNATCHER", "19da50a9bf457980", "2026-09-30"),
    ("(v)", "BASELINES", "45175a9bbd7ef27d", "2026-10-01"),
    ("(vi)", "VARIANTS", "eb11617bf34b95b9", "2026-10-02"),
    ("(vii)", "FINAL_ROUND", "97b238d3482dd690", "2026-10-05"),
    ("(viii)", "FOUNDATION_ROUND", "204f66d342b118be", "2026-10-06"),
]
HOST = {  # configs/hosts name -> paper host name (the recorded rounds (i)-(ii) also use TANE_DNPG_pretrain / TANE_tiered_pretrain)
    "frozen_feat_mini": "frozen FEAT", "frozen_feat_tiered": "frozen FEAT", "frn_mini_seed0": "FRN", "frn_mini_seed1": "FRN",
    "frn_mini_seed2": "FRN", "tane_pre_mini": "TANE pre-trained", "tane_pre_tiered": "TANE pre-trained", "cifarfs_rfs_seed0": "RFS",
    "snatcher_f_mini": "SnaTCHer-F", "snatcher_f_tiered": "SnaTCHer-F", "dnpg_mini_seed0": "DNPG", "dnpg_mini_seed1": "DNPG",
    "dnpg_mini_seed2": "DNPG", "dnpg_tiered_seed0": "DNPG", "TANE_DNPG_pretrain": "TANE pre-trained", "TANE_tiered_pretrain": "TANE pre-trained",
}
BACKBONE = {"clip_b16": "CLIP ViT-B/16", "dinov2_s14": "DINOv2 ViT-S/14", "dinov2_b14": "DINOv2 ViT-B/14", "dinov2_l14": "DINOv2 ViT-L/14"}
DATASET = {"mini": "mini-ImageNet", "tiered": "tieredImageNet", "cifarfs": "CIFAR-FS", "cub": "CUB"}

_PAPER_FAMILY = {v: k for k, v in FAMILY.items()}
_PAPER_ROW = {v: k for k, v in ROW.items()}
_PAPER_ABLATION = {v: k for k, v in ABLATION.items()}


def _split_family(key: str, families) -> tuple[str, list[str]] | None:
    """(family, [segments]) when `key` is FAMILY or FAMILY/..., the longest family match first."""
    for fam in sorted(families, key=len, reverse=True):
        if key == fam or key.startswith(fam + "/"):
            return fam, ([] if key == fam else key[len(fam) + 1:].split("/"))
    return None


def paper_name(key: str) -> str:
    """Recorded score or row key -> the paper's name. Unknown keys (COS, COSF, CCP-A, CCP-S, host, H_fix, K_fix, D, S, ...) are returned
    unchanged: they already are the paper's names."""
    m = re.fullmatch(r"H\[(.+)\]", key)
    if m:
        return f"fused[{paper_name(m.group(1))}]"
    if key in ROW:
        return ROW[key]
    if key in ABLATION:
        return ABLATION[key]
    hit = _split_family(key, FAMILY)
    if hit is None:
        return key
    fam, segs = hit
    out = []
    for s in segs:
        if s in SEGMENT.get(fam, {}):
            out.append(SEGMENT[fam][s])
            continue
        name, eq, value = s.partition("=")
        out.append(PARAM.get(fam, {}).get(name, name) + eq + value)
    return "/".join([FAMILY[fam]] + out)


def recorded_key(name: str) -> str:
    """The paper's name -> the recorded key (inverse of paper_name)."""
    m = re.fullmatch(r"fused\[(.+)\]", name)
    if m:
        return f"H[{recorded_key(m.group(1))}]"
    if name in _PAPER_ROW:
        return _PAPER_ROW[name]
    if name in _PAPER_ABLATION:
        return _PAPER_ABLATION[name]
    hit = _split_family(name, _PAPER_FAMILY)
    if hit is None:
        return name
    pfam, segs = hit
    fam = _PAPER_FAMILY[pfam]
    inv_param = {v: k for k, v in PARAM.get(fam, {}).items()}
    inv_seg = {v: k for k, v in SEGMENT.get(fam, {}).items()}
    out = []
    for s in segs:
        if s in inv_seg:
            out.append(inv_seg[s])
            continue
        name_, eq, value = s.partition("=")
        out.append(inv_param.get(name_, name_) + eq + value)
    return "/".join([fam] + out)


def is_baseline(name: str) -> bool:
    """True for a baseline score (paper name), fused or not."""
    m = re.fullmatch(r"fused\[(.+)\]", name)
    rec = recorded_key(m.group(1) if m else name)
    return bool(re.match(r"B\d", rec))
