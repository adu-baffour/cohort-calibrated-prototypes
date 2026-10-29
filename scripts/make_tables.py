r"""The paper's Tables 1-8 (LaTeX) from the recorded results.

    python scripts/make_tables.py [T1 ... T8 | all]          # -> paper_assets/tables/T*.tex
    python scripts/make_tables.py all --check                # compare with paper_assets/tables/T*.tex; write nothing

Every number is read from a recorded file through ccp.recorded (sha256-checked, keys in the paper's names). Each generated file starts with
one comment line (source files); the check requires every byte, that line included, to equal the committed table. Ranking marks
(Tables 2-7): bold = best, underline = second best, decided on the printed (rounded) values; ties share a mark and the next distinct value is
second. Bold is \textbf{\boldmath ...} and underline \underline{...}, applied to the value only, not to its interval.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ccp import naming, recorded  # noqa: E402

OUT = ROOT / "paper_assets" / "tables"
SH = (1, 5)
USED: list[str] = []


def load(rel: str):
    USED.append(rel)
    return recorded.load(rel)


# ---------------------------------------------------------------------------------------------------------------- formatting
def num(x, d=2) -> str:
    return f"$-${abs(x):.{d}f}" if x < 0 else f"{x:.{d}f}"


def sgn(x, d=2) -> str:
    return ("$-$" if x < 0 else "$+$") + f"{abs(x):.{d}f}"


def pm(e, small=r"\scriptsize", d=2) -> str:
    return f"{{{small}$\\pm${e:.{d}f}}}"


def published(v: str):
    """A recorded published value as printed ('69.10±0.29', '65.86 ± 0.85', '67.78') -> (value text, interval text)."""
    v = v.strip()
    m = re.match(r"^([0-9.]+)\s*±\s*([0-9.]+)$", v)
    if m:
        return m.group(1), f"{{\\scriptsize$\\pm${m.group(2)}}}"
    if re.match(r"^[0-9.]+$", v):
        return v, ""
    raise SystemExit(f"published value {v!r} has no printed form")


def value_of(text: str) -> float:
    """The numeric value of a printed cell ('$-$0.65^{\\downarrow}' -> -0.65)."""
    t = text.replace("$-$", "-").replace("$+$", "+")
    m = re.search(r"[-+]?\d+\.\d+", t)
    return float(m.group(0))


def ranks(printed: list[str | None], best: str = "max") -> list[str]:
    """'bold' / 'underline' / '' per entry, on the printed values (None = not ranked). Ties share a mark; the next distinct value is second."""
    vals = sorted({value_of(p) for p in printed if p is not None}, reverse=best == "max")
    out = []
    for p in printed:
        if p is None:
            out.append("")
            continue
        v = value_of(p)
        out.append("bold" if v == vals[0] else ("underline" if len(vals) > 1 and v == vals[1] else ""))
    return out


def marked(value: str, rank: str) -> str:
    return {"bold": f"\\textbf{{\\boldmath {value}}}", "underline": f"\\underline{{{value}}}", "": value}[rank]


def group(text, ncol) -> str:
    return f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{text}}}}} \\\\"


def float_env(env: str, place: str, size: str, colsep: str | None, caption: str, label: str, body: list[str], notes: list[str]) -> str:
    sep = [f"\\setlength{{\\tabcolsep}}{{{colsep}}}"] if colsep else []
    L = [f"\\begin{{{env}}}[{place}]", "\\centering", size] + sep + [f"\\caption{{{caption}}}\\label{{{label}}}"] + body
    if notes:
        L += ["\\par\\smallskip", "\\begin{minipage}{\\linewidth}\\scriptsize"] + [f"{n}\\par" for n in notes] + ["\\end{minipage}"]
    L.append(f"\\end{{{env}}}")
    return "\n".join(L) + "\n"


def tabular(cols: str, head: list[str], rows: list[str]) -> list[str]:
    return [f"\\begin{{tabular}}{{{cols}}}", "\\toprule"] + head + ["\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]


SHOT_HEAD = [r" & \multicolumn{2}{c}{1-shot} & \multicolumn{2}{c}{5-shot} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Method & Acc. & AUROC & Acc. & AUROC \\"]


def ci_get(x, m):
    return x[m], x[f"{m}_ci95"], ""


def seed_get(x, m):
    return x[m]["mean"], x[m]["std"], "s"


def val(x: dict, m: str):
    """(value, interval, mark): a single-model row stores m and m_ci95; a seeded row (DNPG, FRN) stores m = {mean, std, per_seed}."""
    if isinstance(x[m], dict):
        return x[m]["mean"], x[m]["std"], "s"
    return x[m], x[f"{m}_ci95"], ""


# ---------------------------------------------------------------------------------------------------------------- published rows
PUB_ROWS = {  # recorded method label -> (short name, bib keys, note letters)
    "PEELER (SnaTCHer's re-implementation, ResNet-12)": ("PEELER", "liu2020peeler", "a"),
    "SnaTCHer-F": ("SnaTCHer-F", "jeong2021snatcher", ""), "ATT-G": ("ATT-G", "huang2022tane", ""), "SEMAN-G": ("SEMAN-G", "huang2022tane", ""),
    "ProCAM": ("ProCAM", "song2022procam", ""), "GEL": ("GEL", "wang2023gel", ""), "MRMT-SnaTCHerF": ("MRMT", "che2023mrm", "b"),
    "DNPG": ("DNPG", "zhang2024dnpg", ""), "OAL-OFL": ("OAL-OFL", "kim2025oalofl", ""), "HDFA": ("HDFA", "yu2025hdfa", ""),
    "OPP+HDE": ("OPP+HDE", "sun2024opp,geng2025hde", "c"),
}
CIFAR_ROWS = [  # (prefix of the recorded method label, source table, short name, bib keys, note letter)
    ("SEMAN-G", "tane", "SEMAN-G", "huang2022tane", ""), ("ATT-G", "tane", "ATT-G", "huang2022tane", ""),
    ("Dynamic", "tane", "Dynamic", "gidaris2018dynamic", "d"), ("PEELER", "tane", "PEELER", "liu2020peeler", "d"),
    ("CounterFactural", "tane", "Counterfactual", "neal2018counterfactual", "d"), ("OpenMax", "tane", "OpenMax", "bendale2016openmax", "d"),
    ("GEL (Ours)", "gel", "GEL", "wang2023gel", ""), ("GEL, Ours w/o Pix", "gel", "GEL w/o Pix", "wang2023gel", "e"),
    ("FEAT*", "gel", "FEAT", "ye2020feat", "f"), ("SnaTCHer-F*", "gel", "SnaTCHer-F", "jeong2021snatcher", "f"),
    ("RFDNet*", "gel", "RFDNet", "deng2023rfdnet", "f"),
]
PUB_NOTES = {
    "a": r"\textsuperscript{a}PEELER: the ResNet-12 re-implementation reported by \cite{jeong2021snatcher} (the original uses ResNet-18).",
    "b": r"\textsuperscript{b}MRMT: the multi-relation margin loss on SnaTCHer-F (MRMT-SnaTCHerF in \cite{che2023mrm}).",
    "c": r"\textsuperscript{c}OPP+HDE: hyperopia distillation \cite{geng2025hde} applied to the overall positive prototype \cite{sun2024opp}.",
}


def method_cell(name: str, keys: str, notes: str) -> str:
    sup = f"\\textsuperscript{{{notes}}}" if notes else ""
    return f"{name}{sup}~\\cite{{{keys}}}"


def pub_rows(bench: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """[(method cell, [(value, interval) for acc 1, AUROC 1, acc 5, AUROC 5])] of the recorded published rows of a benchmark."""
    out = []
    for r in load("results/recorded/published/published_sota.csv"):
        if r["benchmark"] != bench:
            continue
        name, keys, nt = PUB_ROWS[r["method"]]
        out.append((method_cell(name, keys, nt), [published(r[c]) for c in ("acc_1shot", "auroc_1shot", "acc_5shot", "auroc_5shot")]))
    return out


def frozen_block(label_: str, R: dict, getter, small_d4=True) -> list[tuple[str, list[tuple[str, str]]]]:
    """COS / COSF / + CCP-A rows of one frozen host. COS and COSF that round to the same two decimals but differ are printed with four."""
    out = []
    for row, lab in (("COS", f"{label_}: COS"), ("COSF", r"\quad COSF"), ("CCP-A", r"\quad + CCP-A")):
        cells = []
        for sh in SH:
            for m in ("acc", "auroc"):
                v, e, _ = getter(R[sh][row], m)
                d = 2
                if row in ("COS", "COSF"):
                    a, b = getter(R[sh]["COS"], m)[0], getter(R[sh]["COSF"], m)[0]
                    if f"{a:.2f}" == f"{b:.2f}" and a != b and small_d4:
                        d = 4
                cells.append((f"{v:.{d}f}", pm(e) + (r"\textsuperscript{d}" if d == 4 and row == "COSF" else "")))
        out.append((lab, cells))
    return out


def ranked_rows(rows: list) -> list[str]:
    """rows: group / rule lines (str) or (label, [(value, suffix)]); bold / underline per column over every data row."""
    data = [r for r in rows if not isinstance(r, str)]
    ncol = len(data[0][1])
    marks = [ranks([r[1][j][0] for r in data]) for j in range(ncol)]
    out, i = [], 0
    for r in rows:
        if isinstance(r, str):
            out.append(r)
            continue
        out.append(" & ".join([r[0]] + [marked(v, marks[j][i]) + s for j, (v, s) in enumerate(r[1])]) + r" \\")
        i += 1
    return out


# ---------------------------------------------------------------------------------------------------------------- Table 2
def t2():
    mini = load("results/recorded/test/FINAL_TEST_FSOR_MINI_DONE.json")
    sn = load("results/recorded/test/FINAL_TEST_FSOR_MINI-SNATCHER_DONE.json")
    dn = load("results/recorded/test/FINAL_TEST_FSOR_MINI-DNPG_DONE.json")
    rows = [group("Published inductive methods, ResNet-12 (as reported)", 5)] + pub_rows("mini-ImageNet")
    rows += ["\\midrule", group("This work (one test pass, 600 episodes per shot)", 5)]
    for key, name in (("frozen_FEAT", "Frozen FEAT"), ("TANE_DNPG_pretrain", r"TANE pre-trained\textsuperscript{\dag}")):
        r = mini["rows"][key]
        rows += frozen_block(name, {sh: r[f"{sh}shot"] for sh in SH}, ci_get)
    fs = mini["frn_seeds"]
    rows += frozen_block(r"FRN\textsuperscript{s}", {sh: fs[f"{sh}shot"] for sh in SH}, seed_get)
    s = sn["rows"]
    for row, lab in (("S", "SnaTCHer-F: host score"), ("H_fix", r"\quad + CCP ($H_{\mathrm{fix}}$)")):
        rows.append((lab, [(f"{s[f'{sh}shot_T'][row][m]:.2f}", pm(s[f"{sh}shot_T"][row][f"{m}_ci95"])) for sh in SH for m in ("acc", "auroc")]))
    d = dn["seed_summary"]
    for row, lab in (("D", r"DNPG\textsuperscript{s}: host score"), ("H_fix", r"\quad + CCP ($H_{\mathrm{fix}}$)")):
        rows.append((lab, [(f"{d[f'{sh}shot_T'][row][m]['mean']:.2f}", pm(d[f"{sh}shot_T"][row][m]["std"])) for sh in SH for m in ("acc", "auroc")]))
    notes = [PUB_NOTES["a"], PUB_NOTES["b"], PUB_NOTES["c"],
             r"\textsuperscript{d}Frozen FEAT 5-shot: COS and COSF AUROC differ only in the third decimal and are printed with four.",
             "Published rows retain each paper's values and intervals (inductive, ResNet-12); these methods are not re-evaluated here.",
             r"Accuracy and AUROC are percentages. This work: mean and 95\% confidence interval (CI) over 600 episodes, except "
             r"\textsuperscript{s}mean and standard deviation over seeds 0--2. COS uses a single view; COSF uses mirror views. For trained hosts, "
             r"$H_{\mathrm{fix}}$ = host score + $w$ CCP-A, with $w$ fixed on validation and labels retained from the host.",
             r"\textsuperscript{\dag}Released TANE checkpoint whose stored epoch is consistent with selection on the test split; its comparison "
             "is paired on the same backbone."]
    body = tabular("lcccc", SHOT_HEAD, ranked_rows(rows))
    return float_env("table*", "tbp", "\\footnotesize", "4pt", "Test results on mini-ImageNet (5-way). Published methods are listed for context. "
                     "Bold: best; underlined: second-best results in each column.", "tab:mini", body, notes)


# ---------------------------------------------------------------------------------------------------------------- Table 3
def t3():
    ti = load("results/recorded/test/FINAL_TEST_FSOR_TIERED_DONE.json")
    fr = load("results/final_round/test.json")
    sota = load("results/recorded/published/sota.csv")
    panel_a = [group("(a) tieredImageNet. Published inductive methods, ResNet-12 (as reported)", 5)] + pub_rows("tieredImageNet")
    panel_a.append(group("This work (one test pass, 600 episodes per shot)", 5))
    for key, name in (("frozen_FEAT_tiered", "Frozen FEAT"), ("TANE_tiered_pretrain", r"TANE pre-trained\textsuperscript{\dag}")):
        r = ti["rows"][key]
        panel_a += frozen_block(name, {sh: r[f"{sh}shot"] for sh in SH}, ci_get)
    panel_b = [group("(b) CIFAR-FS. Published (as reported)", 5)]
    cols = ("acc_1shot_5way", "auroc_1shot", "acc_5shot_5way", "auroc_5shot")
    shown, skipped = [], []
    for r in sota:
        src = r["source_url_and_table"]
        tane = "2012.13073" in src and "Table 5" in src
        gel = "2304.11855" in src and "(Table 1" in src
        if not (r["benchmark"].startswith("CIFAR-FS") and (tane or gel)):
            continue
        vals = tuple(re.sub(r"\s+", "", r[c]) for c in cols)
        if gel and any(vals == v for _, v in shown):
            skipped.append(re.sub(r"[*\s].*$", "", r["method"]))
            continue
        hit = [x for x in CIFAR_ROWS if r["method"].startswith(x[0]) and x[1] == ("tane" if tane else "gel")]
        if len(hit) != 1:
            raise SystemExit(f"CIFAR-FS published row {r['method']!r}: no unique short name")
        _, _, name, keys, nt = hit[0]
        shown.append((r["method"], vals))
        panel_b.append((method_cell(name, keys, nt), [published(r[c]) for c in cols]))
    assert sorted(skipped) == ["ATT-G", "PEELER"], skipped
    panel_b.append(group("This work: RFS ResNet-12 (RFS recipe, seed 0), one test pass", 5))
    a = fr["tables"]["A"]["CIFAR-FS"]
    assert a["1"]["rows"]["host"] == a["1"]["rows"]["COSF"]
    for row, lab in (("COSF", r"RFS: COSF\textsuperscript{g}"), ("CCP-A", r"\quad + CCP-A")):
        panel_b.append((lab, [(f"{a[str(sh)]['rows'][row][m]:.2f}", pm(a[str(sh)]["rows"][row][f"{m}_ci95"])) for sh in SH for m in ("acc", "auroc")]))
    notes = [r"\textsuperscript{a--c}As in \cref{tab:mini}.",
             r"\textsuperscript{d}Implemented by the TANE authors on their pre-trained ResNet-12 (\cite{huang2022tane}, Table 5); intervals from its "
             "supplementary Table S2 (the main table prints none).",
             r"\textsuperscript{e}GEL without its pixel-wise branch (\cite{wang2023gel}, Table 1). DNPG \cite{zhang2024dnpg} and HDE \cite{geng2025hde} "
             "reprint these numbers under the label GEL.",
             r"\textsuperscript{f}Run or reprinted by the GEL authors (\cite{wang2023gel}, Table 1); its PEELER and ATT-G rows equal the TANE rows "
             "above and are not repeated.",
             r"\textsuperscript{g}The host's reference score; single-view COS was not evaluated for this host. RFS backbone trained by us "
             r"(\cref{sec:hosts}).",
             r"Accuracy and AUROC are percentages; published values and intervals are retained as reported. Protocols: TANE samples 600 tasks with "
             "15 queries per known class and 5 unknown classes of 15 queries; GEL uses the same query structure but states neither the number of "
             r"test episodes nor what its intervals are; this work uses 600 fixed episodes per shot and 95\% CIs. tieredImageNet: images are fed "
             r"in blue--green--red (BGR) channel order to both released checkpoints. \textsuperscript{\dag}Released TANE checkpoint (stored epoch "
             "consistent with selection on the test split)."]
    body = tabular("lcccc", SHOT_HEAD, ranked_rows(panel_a) + ["\\midrule"] + ranked_rows(panel_b))
    return float_env("table*", "tbp", "\\footnotesize", "4pt", "Test results on (a) tieredImageNet and (b) CIFAR-FS (5-way). Published methods "
                     "are listed for context. Bold: best; underlined: second-best results within each panel.", "tab:tiered-cifar", body, notes)


# ---------------------------------------------------------------------------------------------------------------- Table 4
CUB_HOSTS = (("frozen FEAT mini", "Frozen FEAT", "CCP-A", "CCP-QW", "CCP-A, target bank"),
             ("SnaTCHer-F mini", "SnaTCHer-F", "H_fix", "CCP-QW fused", "H_fix, target bank"),
             ("DNPG mini", r"DNPG\textsuperscript{s}", "H_fix", "CCP-QW fused", "H_fix, target bank"))


def t4():
    fr = load("results/final_round/test.json")["tables"]["B"]
    head = [r" & & & & & \multicolumn{2}{c}{Target bank\textsuperscript{a}} \\", r"\cmidrule(lr){6-7}",
            r"Host & Shot & Reference & CCP & CCP-QW & $K=1000$ & $K=154$ \\"]
    rows = []
    for host, lab, ccp, qw, tb in CUB_HOSTS:
        for sh in SH:
            r = fr[host][str(sh)]["rows"]
            cells = [val(r[k], "auroc") for k in ("host", ccp, qw, tb, tb + ", K=154")]
            printed = [f"{v:.2f}" for v, _, _ in cells]
            mk = ranks(printed[:3]) + ["", ""]
            rows.append(" & ".join([lab if sh == 1 else "", str(sh)] + [marked(p, m) + pm(e) for p, m, (_, e, _) in zip(printed, mk, cells)]) + r" \\")
    notes = [r"Mean AUROC (\%) and 95\% CI over 600 CUB test episodes per shot (\textsuperscript{s}DNPG: mean and standard deviation over seeds "
             "0--2); hosts trained on mini-ImageNet, cohort bank = the mini-ImageNet base images (registered). Reference: COSF for frozen FEAT and "
             r"the host score for SnaTCHer-F and DNPG. CCP: CCP-A for frozen FEAT and $H_{\mathrm{fix}}$ for SnaTCHer-F and DNPG; CCP-QW: the "
             r"query-weighted extension (\cref{eq:v1}), fused with the host score for the trained hosts. \Cref{tab:baselines} compares CCP with "
             "every training-free baseline on these configurations.",
             r"\textsuperscript{a}Separate, unregistered analysis: the same score with a cohort bank of CUB base images, at $K=1000$ and at $K=154$, "
             "which approximately matches the fraction that $K=1000$ selects from the mini-ImageNet bank."]
    body = tabular("lcccccc", head, rows)
    return float_env("table*", "tbp", "\\footnotesize", "4pt", r"Test AUROC under mini-ImageNet $\to$ CUB transfer, with hosts, cohort banks, "
                     "and constants fixed on mini-ImageNet. Bold: best; underlined: second best among the reference, CCP, and CCP-QW in each row. "
                     "The target-bank columns are a separate analysis.", "tab:cub", body, notes)


# ---------------------------------------------------------------------------------------------------------------- Table 1
ROUNDS = [  # (round, scope, validation decisions taken before the round, sections), from the registration records (docs/registration_log.md)
    ("(i)", "mini-ImageNet: frozen FEAT, FRN (three seeds), and TANE pre-trained; CCP-S as a secondary method", None, "sec:res-mini"),
    ("(ii)", "tieredImageNet: frozen FEAT and TANE pre-trained; CCP-S as a secondary method", None, "sec:res-tiered-cifar"),
    ("(iii)", "mini-ImageNet: DNPG (three seeds)", "CCP-A alone in place of the host score: not met; fixed-weight fusion: met", "sec:res-mini"),
    ("(iv)", "mini-ImageNet: SnaTCHer-F", "SnaTCHer-F fusion: met on mini-ImageNet, not met on tieredImageNet; DNPG fusion on tieredImageNet: not met",
     "sec:res-mini"),
    ("(v)", "training-free baselines for frozen FEAT, SnaTCHer-F, and DNPG on mini-ImageNet and frozen FEAT on tieredImageNet",
     "baseline settings selected on valA", "sec:res-baselines"),
    ("(vi)", "method variants", "CCP-QW selected; three variants not selected", "sec:ablations"),
    ("(vii)", "CIFAR-FS (RFS) and transfer from mini-ImageNet to CUB, with their training-free baselines; OSCR, FPR95, and SCG",
     "baseline settings selected on valA", "sec:res-mini,sec:res-tiered-cifar,sec:res-baselines"),
    ("(viii)", "foundation-model hosts on four datasets; hypotheses H1, H2, and H3",
     "baseline settings selected on valA; every registered configuration tested regardless of validation results", "sec:results-foundation,sec:when"),
]


def t1():
    rows = []
    for rd, scope, dec, sec in ROUNDS:
        rb = r"\raggedright " if "," in sec else ""
        rows.append(f"{rd} & \\raggedright {scope} & " + (f"\\raggedright {dec}" if dec else "--") + f" & {rb}\\cref{{{sec}}} \\tabularnewline")
    notes = ["Decisions apply the acceptance criteria registered for each host. Round (vii) also re-evaluated the configurations of rounds "
             "(i)--(iv) to compute OSCR and FPR95 and reproduced their recorded AUROC and accuracy exactly."]
    body = tabular("p{0.85cm}p{6.0cm}p{6.75cm}p{1.9cm}", [r"Round & Scope & Validation decisions taken before the round & Reported in \\"], rows)
    return float_env("table*", "!tbp", "\\footnotesize", "4pt", "The eight registered test rounds, in order. Each round was registered and run "
                     "on validation episodes before its single test pass.", "tab:rounds", body, notes)


# ---------------------------------------------------------------------------------------------------------------- Table 5
COLS5 = ["query cohort term", "kNN", "RMD", "CL2N", "ZN", "NNN", "QB-Norm", "TEEN", "episode CSLS"]
INDUCTIVE = [c for c in COLS5 if c != "episode CSLS"]
BASELINES = {v for k, v in naming.FAMILY.items() if k.startswith("B")}


def family(k: str) -> str:
    """'fused[NNN/alpha=1.0/k=1000]' -> 'NNN'; 'QB-Norm IS/temperature=10' -> 'QB-Norm IS'."""
    k = k[6:-1] if k.startswith("fused[") else k
    return k.split("/")[0]


def baseline_cells() -> list[tuple[str, str, int, dict]]:
    """Table 5's 16 configurations: (group, host label, shot, {baseline family: (CCP - baseline, 95% CI)}); QB-Norm IS / DIS kept apart."""
    bt = load("results/baselines/test.json")["paired_test"]
    fr = load("results/final_round/test.json")["tables"]["A"]["CIFAR-FS"]
    dv = load("results/derived/paired_derived.json")["cub"]
    cells = []   # (group, host label, shot, {family: (CCP - baseline, ci95)})
    for host, ds, lab in (("frozen FEAT mini", "mini-ImageNet", "Frozen FEAT"), ("SnaTCHer-F mini", "mini-ImageNet", "SnaTCHer-F"),
                          ("DNPG mini", "mini-ImageNet", "DNPG"), ("frozen FEAT tiered", "tieredImageNet", "Frozen FEAT")):
        for sh in SH:
            cells.append((ds, lab, sh, {family(k): (-v["minus_CCP"]["mean"], v["minus_CCP"]["ci95"]) for k, v in bt[host][str(sh)].items()}))
    for sh in SH:
        p = fr[str(sh)]["paired"]["CCP-A"]
        cells.append(("CIFAR-FS", "RFS", sh, {family(k): (-v["auroc"]["mean"], v["auroc"]["ci95"]) for k, v in p.items() if family(k) in BASELINES}))
    for host, lab in (("frozen FEAT mini", "Frozen FEAT"), ("SnaTCHer-F mini", "SnaTCHer-F"), ("DNPG mini", "DNPG")):
        for sh in SH:
            p = dv[f"{host}|{sh}"]["auroc"]
            cells.append(("CUB (mini-ImageNet hosts)", lab, sh, {family(k): (v["mean"], v["ci95"]) for k, v in p.items() if family(k) in BASELINES}))
    return cells


def t5():
    cells = baseline_cells()
    head = [r"Host & Shot & \begin{tabular}[b]{@{}c@{}}Query\\cohort term\end{tabular} & kNN & RMD & CL2N & ZN & NNN & QB-Norm & TEEN & "
            r"\begin{tabular}[b]{@{}c@{}}Episode\\CSLS\end{tabular} & $n_{+}$ / $n_{0}$ / $n_{-}$ \\"]
    rows, tot, last = [], [0, 0, 0], None
    for ds, lab, sh, d in cells:
        if ds != last:
            rows.append(group(ds, 12))
            last = ds
        assert sorted(d) == sorted(BASELINES), (lab, sorted(d))
        assert d["QB-Norm IS"] == d["QB-Norm DIS"], (lab, sh)    # identical recorded differences in every configuration: one column
        d = {**d, "QB-Norm": d["QB-Norm IS"]}
        printed, cnt = {}, [0, 0, 0]
        for b in COLS5:
            m, e = d[b]
            mark = ""
            if b != "episode CSLS":
                k = 0 if m - e > 0 else (2 if m + e < 0 else 1)
                cnt[k] += 1
                mark = {0: "", 1: r"^{\circ}", 2: r"^{\downarrow}"}[k]
            printed[b] = f"${'-' if m < 0 else ''}{abs(m):.2f}{mark}$"
        mk = dict(zip(INDUCTIVE, ranks([printed[b] for b in INDUCTIVE], best="min")))
        tot = [a + b for a, b in zip(tot, cnt)]
        rows.append(" & ".join([r"\quad " + lab if sh == 1 else "", str(sh)] + [marked(printed[b], mk.get(b, "")) for b in COLS5]
                               + [" / ".join(map(str, cnt))]) + r" \\")
    rows += ["\\midrule", f"\\multicolumn{{11}}{{l}}{{Total over {len(cells)} configurations $\\times$ {len(INDUCTIVE)} inductive baselines}} & "
             + " / ".join(map(str, tot)) + r" \\"]
    notes = [r"Entries: paired AUROC difference CCP $-$ baseline (points; positive: CCP higher), mean over the 600 test episodes. $n_{+}$ / "
             r"$n_{0}$ / $n_{-}$: number of the eight inductive baselines with the 95\% CI above zero (no mark), containing zero ($^{\circ}$), or "
             r"below zero ($^{\downarrow}$); QB-Norm IS and QB-Norm DIS are identical in every configuration and form one column.",
             r"Baselines as in \cref{sec:baselines}; query cohort term: $m_K(q)$. CCP = CCP-A on frozen hosts; for SnaTCHer-F and DNPG, CCP and "
             r"each baseline score are fused with the host score by fixed-weight fusion (\cref{eq:hfix}). Settings chosen on valA "
             r"(\cref{sec:baselines})."]
    body = tabular("lcrrrrrrrrrc", head, rows)
    return float_env("table*", "tbp", "\\scriptsize", "3pt", "Paired test AUROC differences, CCP minus each training-free baseline, on "
                     "ResNet-12 hosts (positive: CCP higher). Bold: strongest; underlined: second-strongest inductive baseline in each row. "
                     "Episode CSLS is a transductive reference and is not ranked.", "tab:baselines", body, notes)


# ---------------------------------------------------------------------------------------------------------------- Table 6
FM_BB = {"clip_b16": "CLIP ViT-B/16", "dinov2_s14": "DINOv2 ViT-S/14", "dinov2_b14": "DINOv2 ViT-B/14", "dinov2_l14": "DINOv2 ViT-L/14"}
FM_DS = {"mini": "mini-ImageNet", "tiered": "tieredImageNet", "cifarfs": "CIFAR-FS", "cub": "CUB"}


def t6():
    fm = load("results/foundation_round/test.json")["summaries"]
    H = load("results/foundation_round/test_hypotheses.json")
    dv = load("results/derived/paired_derived.json")["foundation"]
    sel = load("results/foundation_round/selection_valA.json")["scg_valA"]
    head = [r" & & \multicolumn{2}{c}{AUROC} & \multicolumn{2}{c}{$\Delta$CCP-A} & \multicolumn{3}{c}{$\Delta$AUROC, baselines} & \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}",
            r"Dataset & Shot & COSF & $\Delta$CCP-A & OSCR & FPR95 & CL2N & Best & Name & SCG (valA) \\"]
    rows = []
    for b, bl in FM_BB.items():
        if rows:
            rows.append("\\midrule")
        rows.append(group(bl, 10))
        for ds, dl in FM_DS.items():
            for sh in SH:
                c = f"{ds}|{b}|{sh}"
                pre = f"{ds}_{b}|T|{sh}|"
                R = {k[len(pre):]: v for k, v in fm.items() if k.startswith(pre)}
                cand = {k: v for k, v in R.items() if family(k) in BASELINES and family(k) != "episode CSLS"}
                bk = max(cand, key=lambda k: cand[k]["auroc"])
                x = H["H1"]["cells"][c]
                p = [sgn(x["mean"]), sgn(dv[c]["minus_COSF"]["CL2N"]["mean"]), sgn(dv[c]["minus_COSF"][bk]["mean"])]
                mk = ranks(p)
                # OSCR and FPR95 changes: differences of the means rounded to two decimals (as the paper prints them)
                oscr = round(R["CCP-A"]["oscr"], 2) - round(R["COSF"]["oscr"], 2)
                fpr = round(R["CCP-A"]["fpr95"], 2) - round(R["COSF"]["fpr95"], 2)
                rows.append(" & ".join([dl if sh == 1 else "", str(sh), f"{R['COSF']['auroc']:.2f}", marked(p[0], mk[0]) + pm(x["ci95"], r"\tiny"),
                                        sgn(oscr), sgn(fpr), marked(p[1], mk[1]), marked(p[2], mk[2]), family(bk),
                                        f"{sel[c]['std_gap']:.2f}"]) + r" \\")
    h1 = H["H1"]
    notes = [r"Frozen image encoders without text input; mean AUROC (\%) over 600 test episodes per shot. $\Delta$CCP-A (AUROC): paired "
             f"difference with its 95\\% CI, the registered H1 statistic ({h1['n_holm_significant']} of {h1['n_cells']} configurations "
             f"Holm-significant, {h1['n_bonferroni_ci_excludes_zero']} with a Bonferroni-simultaneous CI above zero).",
             "OSCR and FPR95: change from COSF to CCP-A in points, computed from means rounded to two decimals; lower FPR95 is better.",
             r"Best: the highest test AUROC among the eight inductive baselines (settings chosen on valA, \cref{sec:baselines}); the post-test "
             "maximum favors the baselines and is not a validation-selected winner.",
             "SCG (valA): the standardized gap on validation (valA), computed before the test pass. Spearman between it and the test gain (H2): "
             f"$\\rho$ = {H['H2']['rho']:.3f}; between test SCG and the test gain (H3): $\\rho$ = {H['H3']['rho']:.3f} (\\cref{{sec:SCG-results}}).",
             r"Possible pretraining overlap is discussed in \cref{sec:hosts}. Absolute AUROCs are not comparable with the ResNet-12 tables."]
    body = tabular("lcrrrrrrlr", head, rows)
    return float_env("table*", "tbp", "\\scriptsize", "4pt", "Frozen foundation-model results for 32 host--dataset--shot configurations; "
                     r"$\Delta$: change from COSF, in points. Bold: largest; underlined: second-largest $\Delta$AUROC among CCP-A, CL2N, and the "
                     "best baseline in each row.", "tab:foundation", body, notes)


# ---------------------------------------------------------------------------------------------------------------- Table 7
T7_GROUPS = [("Reference", [("COS", "COS"), ("COSF", "COSF"), ("CCP-A (registered)", "CCP-A (registered)")]),
             ("Cohort size", [("K=100", "$K$ = 100"), ("K=300", "$K$ = 300"), ("K=3000", "$K$ = 3000"), ("K=all", "$K$ = all")]),
             ("Cohort composition", [("class means (matched top-k)", "class means, matched fraction"), ("class means (all)", "class means, all")]),
             ("Correction", [("query-side only", "query only"), ("prototype-side only", "prototype only")]),
             ("Statistics", [("CCP-S (mean + std)", "CCP-S (mean, std)")]), ("Views", [("flip off", "single view")])]
VARIANTS = ["CCP-QW", "full AS-norm", "shared-cohort correction", "fusion-weight factor"]
SYM = {"lambda": r"\lambda", "beta": r"\beta", "gamma": r"\gamma"}


def setting(chosen) -> str:
    """'CCP-QW/lambda=1.0/beta=0.5' -> '$\\lambda=1$, $\\beta=\\tfrac12$'; no parameter -> '--'; one setting per host -> 'per host'."""
    if isinstance(chosen, dict):
        return "per host"
    out = []
    for p in chosen.split("/")[1:]:
        k, v = p.split("=")
        v = r"\tfrac12" if v == "0.5" else (v[:-2] if v.endswith(".0") else v)
        out.append(f"${SYM[k]}={v}$")
    return ", ".join(out) or "--"


def criteria(flags: list[bool]) -> str:
    met = [f"({r})" for r, f in zip(("i", "ii", "iii", "iv"), flags) if f]
    return "none" if not met else ("(i)--(iv)" if len(met) == 4 else ", ".join(met))


def t7():
    A = load("results/recorded/validation/ccp_ablations.json")["hosts"]
    fm, ft = A["frozen FEAT (mini)"]["rows"], A["frozen FEAT tiered (BGR)"]["rows"]
    vb = load("results/variants/validation_valB.json")
    vc = load("results/variants/validation_valC.json")
    vt = load("results/variants/test.json")["cells_test"]
    data, cis = [], {"mini": [], "tiered": []}     # (a) ablations on valB, ranked per column over every row
    for g, rows in T7_GROUPS:
        data.append(group(g, 9))
        for key, lab in rows:
            cells = []
            for d, dn in ((fm, "mini"), (ft, "tiered")):
                for metric in ("auroc", "acc"):
                    for sh in SH:
                        x = d[f"{key}|B{sh}"]
                        cells.append((f"{x[metric]:.2f}", ""))
                        if metric == "auroc":
                            cis[dn].append(x["auroc_ci95"])
            data.append((r"\quad " + lab, cells))
    head_a = [r" & \multicolumn{4}{c}{mini-ImageNet, valB} & \multicolumn{4}{c}{tieredImageNet, valB} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
              r" & \multicolumn{2}{c}{AUROC} & \multicolumn{2}{c}{Acc.} & \multicolumn{2}{c}{AUROC} & \multicolumn{2}{c}{Acc.} \\",
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
              r"Variant & 1-shot & 5-shot & 1-shot & 5-shot & 1-shot & 5-shot & 1-shot & 5-shot \\"]
    a = tabular("lrrrrrrrr", head_a, ranked_rows(data))
    a.insert(1, group("(a) Validation ablations on frozen FEAT", 9))
    rows = []                                      # (b) registered variants, ranked per gain column
    for v in VARIANTS:
        g, g4 = vb["criteria_valB"][v], vc["criterion_iv_valC"][v]
        diffs = [c[v]["paired_auroc_vs_reference"]["mean"] for c in vt.values() if v in c]
        rows.append((v[0].upper() + v[1:], setting(vb["chosen"][v]),
                     [sgn(g["mean_gain"]), sgn(g["min_cell_gain"]), sgn(g4["mean_gain"]), sgn(g4["min_cell_gain"])],
                     criteria([g["(i)"], g["(ii)"], g["(iii)"], g4["(iv)"]]), f"{sgn(min(diffs))} to {sgn(max(diffs))}"))
    mk = [ranks([r[2][j] for r in rows]) for j in range(4)]
    lines = [" & ".join([r[0], r[1]] + [marked(x, mk[j][i]) for j, x in enumerate(r[2])] + [r[3], r[4]]) + r" \\" for i, r in enumerate(rows)]
    head_b = [r" & & \multicolumn{2}{c}{valB gain} & \multicolumn{2}{c}{valC gain} & & \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
              r"Variant & Setting & Mean & Min. & Mean & Min. & Criteria met & Test difference \\"]
    b = tabular("llcccccl", head_b, lines)
    b.insert(1, group("(b) Registered method variants", 8))
    notes = [r"Validation only (1,000 episodes per shot); AUROC and accuracy are percentages. Frozen FEAT host; tieredImageNet images in BGR "
             r"channel order (\cref{tab:tiered-cifar}). The 95\% CI of an AUROC entry in (a) is "
             f"{min(cis['mini']):.2f}--{max(cis['mini']):.2f} on mini-ImageNet and {min(cis['tiered']):.2f}--{max(cis['tiered']):.2f} on "
             "tieredImageNet; between-row comparisons use paired differences on the same episodes.",
             "Registered CCP-A: $K$ = 1000, both cohort terms, mean, mirror views. $K$ = all: mean over the whole bank; class means: the bank "
             "replaced by base-class means, with the fraction of nearest means matched to $K$ or with all means; query only / prototype only: "
             "$c_k-m_K(q)$ / $c_k-m_K(p_k)$, each with coefficient 1; CCP-S: mean and standard deviation; single view: no mirror views for "
             "queries, prototypes, and bank.",
             r"(b) AUROC gains in points over the reference (CCP-A on frozen hosts, $H_{\mathrm{fix}}$ on trained hosts), averaged over the eight "
             "configurations of the variants round (the four trained-host configurations for the fusion-weight factor); min.: lowest "
             r"configuration gain. Criteria (i)--(iv) as in \cref{sec:ablations}. Test difference: range of the paired test differences from the "
             "reference over the same configurations, from a single test pass of all four variants."]
    return float_env("table*", "tbp", "\\footnotesize", "4pt", "(a) Validation ablations on frozen FEAT and (b) the four registered method "
                     "variants. Bold: best; underlined: second best, per column (test-difference ranges not ranked).", "tab:ablations",
                     a + ["", "\\medskip"] + b, notes)


# ---------------------------------------------------------------------------------------------------------------- Table 8
def t8():
    c = load("results/recorded/validation/fsor_cost.json")
    s = load("results/derived/cost_snatcher_valA.json")
    pe, sp = c["per_episode"], s["per_episode"]
    co, dp, th = c["cohort"], c["dnpg_params"], c["training_hours"]
    t = lambda d, k, p="": " / ".join(f"{p}{d[f'{sh}shot'][k]['ms_per_episode']:.2f}" for sh in SH)  # noqa: E731
    mb = lambda x: f"{x:.2f}" if x < 10 else f"{x:.0f}"  # noqa: E731
    m = lambda d, k, p="": " / ".join(f"{p}{mb(d[f'{sh}shot'][k]['peak_extra_MB'])}" for sh in SH)  # noqa: E731
    dn_own = "DNPG (own score, incl. its forward with 5x support augmentation)"
    dn_ccp = "CCP-A on DNPG features (extra over DNPG: flip-view forward + CCP-A)"
    sn_own = "SnaTCHer-F (own score S, incl. its fp32 forward)"
    sn_ccp = "CCP-A on SnaTCHer-F features (extra over SnaTCHer-F: flip-view forward + CCP-A)"
    th_mini = sum(th["mini_per_seed_measured"].values()) / len(th["mini_per_seed_measured"])
    assert c["gpu"] == s["gpu"] == "NVIDIA GeForce RTX 4090"
    body = [("Backbone forward", "--", "--", t(pe, "backbone"), m(pe, "backbone")),
            ("COSF", "none", "none", t(pe, "COSF"), m(pe, "COSF")),
            ("CCP-A", "none", f"bank of {co['n_base']:,} images: {co['embed_seconds_both_views']:.1f} s, {co['ccpA_bank_MB_fp16']:.0f} MB",
             t(pe, "CCP-A"), m(pe, "CCP-A")),
            ("CCP-S", "none", f"statistics, {co['ccpS_stats_MB_fp32']:.1f} MB", t(pe, "CCP-S"), m(pe, "CCP-S")),
            None,
            ("SnaTCHer-F", "released model", "--", t(sp, sn_own), m(sp, sn_own)),
            (r"\quad + CCP ($H_{\mathrm{fix}}$)", "one scalar $w$", "bank with its encoder", t(sp, sn_ccp, "+"), m(sp, sn_ccp, "+")),
            ("DNPG", f"{dp['new_modules_trainable'] / 1e6:.1f} M; {th_mini:.1f} GPU-hours per seed", "--", t(pe, dn_own), m(pe, dn_own)),
            (r"\quad + CCP ($H_{\mathrm{fix}}$)", "one scalar $w$", f"bank with its encoder: {co['dnpg_embed_seconds_both_views']:.1f} s",
             t(pe, dn_ccp, "+"), m(pe, dn_ccp, "+"))]
    rows = ["\\midrule" if r is None else " & ".join(f"\\raggedright {x}" for x in r[:3]) + f" & {r[3]} & {r[4]} \\tabularnewline" for r in body]
    notes = [f"{c['gpu']}; mean over {s['episodes']} mini-ImageNet validation episodes per shot (1 / 5-shot) after {s['warmup']} warm-up "
             "episodes; memory: peak allocation inside the timed block minus the allocation before it (1\\,MB~$=2^{20}$~bytes). Backbone "
             "forward: support set and 150 queries, both mirror views. Rows with + give the extra cost over the host (mirror-view forward and "
             "CCP-A).",
             f"DNPG: {dp['new_modules_trainable']:,} new trainable parameters plus {dp['phase2_centres']:,} values of the centers learned in "
             f"DNPG's second training phase; training time is the mean of seeds 1 and 2 on mini-ImageNet ({th['tiered_seed0_measured']:.1f} "
             "GPU-hours for tieredImageNet seed 0). CCP-A bank in half precision (fp16)."]
    tab = tabular("p{2.4cm}p{2.3cm}p{2.6cm}rr", [r"Method & Params; training & One-off cost & Time (ms) & Memory (MB) \\"], rows)
    return float_env("table*", "tbp", "\\footnotesize", "4pt", "Runtime, memory, and training costs on an RTX 4090 (1-shot / 5-shot episodes). "
                     "Entries with + are additions to the trained host.", "tab:cost", tab, notes)


# ---------------------------------------------------------------------------------------------------------------- main
TABLES = {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5, "T6": t6, "T7": t7, "T8": t8}
HEADER = "% Generated by scripts/make_tables.py from "


def render(tid: str) -> tuple[str, list[str]]:
    """(table text without the header line, the recorded files it read)."""
    USED.clear()
    body = TABLES[tid]()
    return body, sorted(set(USED))


def header(sources: list[str]) -> str:
    return HEADER + (", ".join(sources) if sources else "the registration records (docs/registration_log.md)") + "\n"


def check(tids) -> list[str]:
    """Table ids whose generated text differs from paper_assets/tables/ (whole file)."""
    bad = []
    for tid in tids:
        body, src = render(tid)
        asset = OUT / f"{tid}.tex"
        if not asset.exists() or asset.read_bytes().decode("utf-8") != header(src) + body:
            bad.append(tid)
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", nargs="*", default=["all"])
    ap.add_argument("--check", action="store_true", help="compare with paper_assets/tables/T*.tex byte for byte; write nothing")
    a = ap.parse_args()
    tids = list(TABLES) if a.tables in ([], ["all"]) else a.tables
    if a.check:
        bad = check(tids)
        for tid in tids:
            print(f"{tid}: " + ("DIFFERS from" if tid in bad else "identical to") + f" paper_assets/tables/{tid}.tex")
        raise SystemExit(1 if bad else 0)
    OUT.mkdir(parents=True, exist_ok=True)
    for tid in tids:
        body, src = render(tid)
        with open(OUT / f"{tid}.tex", "w", encoding="utf-8", newline="\n") as f:
            f.write(header(src) + body)
        print(f"{tid}: {OUT.relative_to(ROOT).as_posix()}/{tid}.tex")


if __name__ == "__main__":
    main()
