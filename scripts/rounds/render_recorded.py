"""Re-render every recorded report from the recorded files alone and compare it byte for byte with the recorded copy.

    python scripts/rounds/render_recorded.py [fsor baselines variants final_round foundation_round | all]

The reports were written by the registered code when each round ran; their text uses the names of the registration documents (for
example D1, V1, H2b, B1-B9), which differ from the paper's names (docs/naming.md). This module therefore reads the recorded files with
their original keys and keeps the original report text, so that the re-rendered files can be compared with the recorded ones byte for byte.
It is the one place outside docs/naming.md, ccp/naming.py and the recorded files where those names appear (scripts/release_scan.py exempts
it). The paper's tables and figures are made by scripts/make_tables.py and scripts/make_figures.py under the paper's names.

Re-rendered from the recorded JSON and the per-episode release assets (no test images, features or scores are opened):
  fsor              results/recorded/test/RESULTS_fsor.md (rounds (i)-(iv)), mini_{dnpg,snatcher}_test_paired.md,
                    results/recorded/validation/reports/{ablations,dnpg_hfix,snatcher_val,dnpg_tiered_seed0}.md
  baselines         results/baselines/{validation,test}.md                                   (round (v))
  variants          results/variants/{validation_valB,validation_valC,test}.md               (round (vi))
  final_round       results/final_round/{cifarfs_valB,cub_val_preview,recorded_valA,test}.md, d1_val.png, d1_test.png (round (vii))
  foundation_round  results/foundation_round/test.md, test_hypotheses.json, d1_val.png, d1_test.png (round (viii)); its
                    validation_valB.md is not re-rendered: its analysis rows print paired CIs that validation_valB.json does not store
Outputs: results/generated/rounds/<round>/recorded/<file>. Before a test report is written, every summary of the round's test record is
recomputed from its per-episode file (sha256-pinned) and must equal the recorded value.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rounds.common import GENERATED, bundle_block, record  # noqa: E402

from ccp import metrics, stats  # noqa: E402
from ccp.recorded import read_bytes  # noqa: E402
from ccp.rounds import final_round as FR  # noqa: E402
from ccp.utils.paths import assets_root, check_sha256, sha256_file  # noqa: E402

TEST_NPZ = {"baselines": ("baselines_test_episodes.npz", "bfe12a90bcc1ef0bca5084d75364b1e197a734a94a7f6a5bbf7544232be16a4a"),
            "variants": ("variants_test_episodes.npz", "86f9a2150a290dbfa27832434d955a61d5b2c0210a220549a557a75202bd5fcd"),
            "final_round": ("final_round_test_episodes.npz", "c4017d760ea5db8460241950aff3f4f9004d98057cc24f62e03316ff4cf0daf9"),
            "foundation_round": ("foundation_round_test_episodes.npz", "06a04e25e024705c20ba8b9e1d247d327366aa0640b2a3bd7a018ba91425fc49")}


def raw(rel: str):
    """A recorded JSON file with its original keys (checked against results/PROVENANCE.md)."""
    return record(rel, paper_names=False)


def out_path(round_name: str, name: str) -> Path:
    d = GENERATED / round_name / "recorded"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def write_text(round_name: str, name: str, lines: list[str]) -> Path:
    p = out_path(round_name, name)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return p


def test_npz(round_name: str):
    name, sha = TEST_NPZ[round_name]
    p = assets_root() / "recorded" / "test" / name
    check_sha256(p, sha, f"{round_name} per-episode test file")
    return np.load(p), p


# ==================================================================================================================== rounds (i)-(iv)
DNPG_ROWS = ("D", "H_fix", "K_fix", "H", "K", "CCP-A", "COSF")
ABL_GROUPS = [("reference", ["COS", "COSF"]), ("registered", ["CCP-A (registered)"]), ("a. cohort size", ["K=100", "K=300", "K=3000", "K=all"]),
              ("b. cohort composition", ["class means (matched top-k)", "class means (all)"]), ("c. side", ["query-side only", "prototype-side only"]),
              ("d. statistics", ["CCP-S (mean + std)"]), ("e. views", ["flip off"])]


def fsor_bundle(part):
    import yaml
    txt = read_bytes("docs/preregistration/FINAL_BUNDLE_fsor.md").decode("utf-8")
    block = re.search(rf"```yaml-fsor-{part}\n(.*?)\n```", txt, re.S).group(1)
    return yaml.safe_load(block), hashlib.sha256(block.encode("utf-8")).hexdigest()[:16]


def lines_mini_tiered(part, token, b, out):
    f = lambda v: f"{v['acc']:.2f} ± {v['acc_ci95']:.2f} | {v['auroc']:.2f} ± {v['auroc_ci95']:.2f}"  # noqa: E731
    L = ["", f"## FSOR FINAL TEST, part {part.upper()} (single evaluation of the registered bundle part, approved by the human)", "",
         f"Bundle token {token}; 600 test episodes per shot, 5-way, 5 known + 5 unknown classes, 15 queries per class; accuracy and AUROC (unknown = positive) with 95% CI over episodes.", "",
         "| Dataset | Backbone | Row | Shot | Accuracy ± 95% CI | AUROC ± 95% CI |", "|---|---|---|---|---|---|"]
    if part == "mini":
        for bk, r in out["rows"].items():
            for sh in (1, 5):
                for k, v in r[f"{sh}shot"].items():
                    L.append(f"| mini-ImageNet | {bk} | {k} | {sh} | {f(v)} |")
        fc = b["fusion"]["candidate"]
        for k, r in out["fusion"].items():
            for sh in (1, 5):
                L.append(f"| mini-ImageNet | FUSION {k} | {fc} | {sh} | {f(r[f'{sh}shot'])} |")
        L += ["", "| Summary over seeds (mean ± std) | Shot | Accuracy | AUROC |", "|---|---|---|---|"]
        for sh in (1, 5):
            p = out["headline"]["PRIMARY"][f"{sh}shot"]
            L.append(f"| **PRIMARY**: FUSION frozen FEAT + FRN seed s ({fc}), s = 0, 1, 2 | {sh} | {p['acc']['mean']:.2f} ± {p['acc']['std']:.2f} | {p['auroc']['mean']:.2f} ± {p['auroc']['std']:.2f} |")
        for sh in (1, 5):
            v = out["headline"]["SECONDARY"][f"{sh}shot"]
            L.append(f"| **SECONDARY**: frozen FEAT + {b['headline']['secondary']['candidate']} (single backbone, training-free; 95% CI) | {sh} | {v['acc']:.2f} ± {v['acc_ci95']:.2f} | {v['auroc']:.2f} ± {v['auroc_ci95']:.2f} |")
        for sh in (1, 5):
            for k, s in out["frn_seeds"][f"{sh}shot"].items():
                L.append(f"| FRN seeds 0-2, {k} (non-headline) | {sh} | {s['acc']['mean']:.2f} ± {s['acc']['std']:.2f} | {s['auroc']['mean']:.2f} ± {s['auroc']['std']:.2f} |")
    else:
        for bk, r in out["rows"].items():
            for sh in (1, 5):
                for k, v in r[f"{sh}shot"].items():
                    L.append(f"| tieredImageNet | {bk} ({b.get('channel_order', 'RGB')}) | {k} | {sh} | {f(v)} |")
        for k, r in out.get("fusion", {}).items():
            for sh in (1, 5):
                L.append(f"| tieredImageNet | FUSION {k} ({b.get('channel_order', 'RGB')}; non-headline) | {b['fusion']['candidate']} | {sh} | {f(r[f'{sh}shot'])} |")
        if "headline" in out:
            hl = b["headline"]
            L += ["", "| Headline | Shot | Accuracy ± 95% CI | AUROC ± 95% CI |", "|---|---|---|---|"]
            for sh in (1, 5):
                L.append(f"| **HEADLINE**: {hl['backbone']} + {hl['candidate']} ({b.get('channel_order', 'RGB')} input, single backbone, training-free) | {sh} | {f(out['headline']['HEADLINE'][f'{sh}shot'])} |")
    L.append("\nDecisions after this test: none (pre-registered).")
    return L


def lines_mini_dnpg(token, b, out):
    lab = {"H": "H (per-episode z, query statistics)", "H_fix": "**H_fix (HEADLINE)**", "D": "D (DNPG own score)"}
    L = ["", "## FSOR FINAL TEST, part MINI-DNPG (single evaluation of the registered bundle part, approved by the human)", "",
         f"Bundle token {token}; 600 test episodes per shot, 5-way, 5 known + 5 unknown classes, 15 queries per class; DNPG max_acc.pth seeds 0-2 "
         "(released TANE/DNPG backbone; selected on validation classes); H_fix = D + w * CCP-A, K_fix = D + w2 * COSF with w / w2 frozen on valA; "
         "accuracy = DNPG classifier for every row; AUROC (unknown = positive) with 95% CI over episodes. "
         f"Per-episode values: {out['per_episode_file']} (sha256 {out['per_episode_sha256'][:16]}).", ""]
    for key in sorted(out["seed_summary"]):
        sh, t = key.split("_")
        S = out["seed_summary"][key]
        L += [f"### {sh}, {'test' if t == 'T' else 'val' + t}", "",
              "| Row | seed 0 | seed 1 | seed 2 | mean ± std over seeds | accuracy mean ± std |", "|---|---|---|---|---|---|"]
        for r in DNPG_ROWS:
            per = [out["rows"][f"seed{e['seed']}_{sh}_{t}"][r] for e in b["seeds"]]
            L.append(f"| {lab.get(r, r)} | " + " | ".join(f"{x['auroc']:.2f} ± {x['auroc_ci95']:.2f}" for x in per)
                     + f" | {S[r]['auroc']['mean']:.2f} ± {S[r]['auroc']['std']:.2f} | {S[r]['acc']['mean']:.2f} ± {S[r]['acc']['std']:.2f} |")
        for d, name in (("Hfix_minus_D", "H_fix − D (paired)"), ("Hfix_minus_Kfix", "H_fix − K_fix (paired)")):
            per = [out["paired"][f"seed{e['seed']}_{sh}_{t}"][d] for e in b["seeds"]]
            L.append(f"| {name} | " + " | ".join(f"{x['mean']:+.2f} ± {x['ci95']:.2f}" for x in per) + f" | {S[d]['mean']:+.2f} ± {S[d]['std']:.2f} | |")
        L.append("")
    L.append("Decisions after this test: none (pre-registered).")
    return L


def lines_mini_snatcher(token, b, out):
    lab = {"S": "S (SnaTCHer-F score)", "H_fix": "**H_fix = S + w · CCP-A (HEADLINE)**", "K_fix": "K_fix = S + w2 · COSF",
           "CCP-A": "CCP-A alone", "COSF": "COSF alone", "Dist": "Dist (FEAT, −max logit)"}
    f = lambda x: f"{x['acc']:.2f} ± {x['acc_ci95']:.2f} | {x['auroc']:.2f} ± {x['auroc_ci95']:.2f}"  # noqa: E731
    L = ["", "## FSOR FINAL TEST, part MINI-SNATCHER (single evaluation of the registered bundle part, approved by the human)", "",
         f"Bundle token {token}; 600 test episodes per shot, 5-way, 5 known + 5 unknown classes, 15 queries per class; FEAT's released meta-trained "
         "ResNet-12 (one model per shot); SnaTCHer-F score S; H_fix = S + w * CCP-A, K_fix = S + w2 * COSF with w / w2 frozen on valA; accuracy = FEAT "
         "classifier for S / H_fix / K_fix / Dist, own argmax for CCP-A / COSF alone; AUROC (unknown = positive) with 95% CI over episodes. "
         f"Per-episode values: {out['per_episode_file']} (sha256 {out['per_episode_sha256'][:16]}).", ""]
    for key in sorted(out["rows"]):
        sh, t = key.split("_")
        L += [f"### {sh}, {'test' if t == 'T' else 'val' + t}", "", "| Row | Accuracy ± 95% CI | AUROC ± 95% CI |", "|---|---|---|"]
        for r in ("S", "H_fix", "K_fix", "CCP-A", "COSF", "Dist"):
            L.append(f"| {lab[r]} | {f(out['rows'][key][r])} |")
        for d, name in (("Hfix_minus_S", "H_fix − S (paired)"), ("Hfix_minus_Kfix", "H_fix − K_fix (paired)")):
            p = out["paired"][key][d]
            L.append(f"| {name} | | {p['mean']:+.2f} ± {p['ci95']:.2f} |")
        L.append("")
    L.append("Decisions after this test: none (pre-registered).")
    return L


def fsor_results_blocks():
    """(name, rendered lines, recorded lines) for the four test blocks of RESULTS_fsor.md."""
    txt = read_bytes("results/recorded/test/RESULTS_fsor.md").decode("utf-8")
    out = []
    for part, done, fn in (("mini", "FINAL_TEST_FSOR_MINI_DONE.json", None), ("tiered", "FINAL_TEST_FSOR_TIERED_DONE.json", None),
                           ("mini-dnpg", "FINAL_TEST_FSOR_MINI-DNPG_DONE.json", lines_mini_dnpg),
                           ("mini-snatcher", "FINAL_TEST_FSOR_MINI-SNATCHER_DONE.json", lines_mini_snatcher)):
        b, token = fsor_bundle(part)
        d = raw(f"results/recorded/test/{done}")
        if d["token"] != token:
            raise ValueError(f"{done}: token {d['token']} != bundle block token {token}")
        L = fn(token, b, d) if fn else lines_mini_tiered(part, token, b, d)
        rendered = "\n".join(L).split("\n")[1:]
        i = txt.index(rendered[0])
        end = "Decisions after this test: none (pre-registered)."
        j = txt.index(end, i) + len(end)
        out.append((f"RESULTS_fsor.md {part.upper()} block", rendered, txt[i:j].split("\n")))
    return out


def fsor_paired_blocks():
    out = []
    zd = np.load(assets_root() / "recorded" / "test" / "final_mini_dnpg_episodes.npz")
    zs = np.load(assets_root() / "recorded" / "test" / "final_mini_snatcher_episodes.npz")
    for name, rows in (("mini_dnpg_test_paired.md", [(sh, lab, np.mean([zd[f"seed{i}_{sh}shot_T_H_fix_auroc"] - zd[f"seed{i}_{sh}shot_T_{b}_auroc"] for i in range(3)], axis=0))
                                                    for sh in (1, 5) for lab, b in (("H_fix − D", "D"), ("H_fix − K_fix", "K_fix"))]),
                       ("mini_snatcher_test_paired.md", [(sh, lab, zs[f"{sh}shot_T_H_fix_auroc"] - zs[f"{sh}shot_T_{b}_auroc"])
                                                        for sh in (1, 5) for lab, b in (("H_fix − S", "S"), ("H_fix − K_fix", "K_fix"))])):
        rendered = []
        for sh, lab, d in rows:
            p = metrics.paired(d, np.zeros_like(d))
            rendered.append(f"| {sh} | {lab} | {p['mean']:+.3f} | ± {p['ci95']:.3f} | {p['wilcoxon_p']:.2e} | {p['frac_gain'] * 100:.1f}% | {p['frac_tie'] * 100:.1f}% |")
        rec = [x for x in read_bytes(f"results/recorded/test/{name}").decode("utf-8").split("\n") if re.match(r"^\| [15] \|", x)]
        out.append((name, rendered, rec))
    return out


def fsor_ablations_block():
    st = raw("results/recorded/validation/ccp_ablations.json")
    LABEL = "ablation, validation, not used for any selection"
    WMULT = (0.25, 0.5, 1.0, 2.0, 4.0)
    f = lambda x: f"{x['auroc']:.2f} ± {x['auroc_ci95']:.2f} / {x['acc']:.2f} ± {x['acc_ci95']:.2f}"  # noqa: E731
    L = ["# Ablations (validation only; not used for any selection)", "",
         f"Every row: **{LABEL}**. No registered choice is changed; the registered rows reproduce the recorded validation numbers "
         "(checked in the text below each table). Cells: AUROC ± 95% CI / accuracy ± 95% CI over 1,000 episodes (5 known + 5 unknown "
         "classes, 15 queries per class). Script scripts/ccp_ablations.py; raw numbers research/status/ccp_ablations.json.", "",
         "Variants (one change each from the registered CCP-A: symmetric, top-1000 base images, mean only, flip views):",
         "a. top-K in {100, 300, 1000, 3000, all}; b. bank = base-class means (top-k matched to the registered 2.6% fraction, or all);",
         "c. query-side or prototype-side calibration only; d. CCP-S (mean + std); e. flip views off. COS / COSF are the uncalibrated references.",
         "Query-side-only accuracy equals COSF accuracy (the query term does not change the argmax)."]
    for host, sets in (("frozen FEAT (mini)", ("A", "B")), ("frozen FEAT tiered (BGR)", ("A", "B"))):
        H = st["hosts"][host]
        tv = "tv" if "tiered" in host else "val"
        L += ["", f"## {host}" + (f" (class-mean matched top-k = {H['kcm']})"), "",
              f"| Group | Variant | {tv}A 1-shot | {tv}B 1-shot | {tv}A 5-shot | {tv}B 5-shot |", "|---|---|---|---|---|---|"]
        for g, vs in ABL_GROUPS:
            for v in vs:
                L.append(f"| {g} | {v} | " + " | ".join(f(H["rows"][f"{v}|{s}{sh}"]) for sh in (1, 5) for s in sets) + " |")
    Dn = st["hosts"]["DNPG seeds 0-2 (mini)"]["per_seed"]

    def ms(key, sh):
        v = [Dn[f"seed{s}_{sh}shot"][key] for s in (0, 1, 2)]
        a = np.array([x["auroc"] for x in v]); c = np.mean([x["auroc_ci95"] for x in v]); acc = np.mean([x["acc"] for x in v])
        return f"{a.mean():.2f} ± {a.std(ddof=1):.2f} (CI {c:.2f}) / {acc:.2f}"
    L += ["", "## DNPG seeds 0-2 (mini, max_acc.pth)", "",
          "Cells: AUROC mean ± std over seeds (mean 95% CI over episodes) / accuracy. Single-score rows: the variant alone on DNPG's features "
          "(its own argmax). Fused rows: H_fix(V) = D + w_V · V with w_V = pooled std(D) / pooled std(V) recomputed on valA for each variant and "
          "seed; accuracy = DNPG classifier. D = DNPG's own score (selected on validation classes).", "",
          "| Group | Variant | alone valA 1-shot | alone valB 1-shot | H_fix(V) valB 1-shot | alone valA 5-shot | alone valB 5-shot | H_fix(V) valB 5-shot |",
          "|---|---|---|---|---|---|---|---|",
          f"| reference | D (DNPG own) | {ms('D|A', 1)} | {ms('D|B', 1)} | | {ms('D|A', 5)} | {ms('D|B', 5)} | |"]
    for g, vs in ABL_GROUPS:
        for v in vs:
            L.append(f"| {g} | {v} | {ms(f'{v}|A', 1)} | {ms(f'{v}|B', 1)} | {ms(f'H_fix({v})|B', 1)} | {ms(f'{v}|A', 5)} | {ms(f'{v}|B', 5)} | {ms(f'H_fix({v})|B', 5)} |")
    L += ["", "### f. Fusion weight (DNPG, H_fix = D + (m × w) · CCP-A; w from valA)", "",
          "| m | valA 1-shot | valB 1-shot | valA 5-shot | valB 5-shot |", "|---|---|---|---|---|"]
    for mlt in WMULT:
        L.append(f"| {mlt}{' (registered)' if mlt == 1.0 else ''} | " + " | ".join(ms(f"w x {mlt}|{s}", sh) for sh in (1, 5) for s in ("A", "B")) + " |")
    L.append("| CCP-A alone | " + " | ".join(ms(f"CCP-A (registered)|{s}", sh) for sh in (1, 5) for s in ("A", "B")) + " |")
    L.append("| D alone | " + " | ".join(ms(f"D|{s}", sh) for sh in (1, 5) for s in ("A", "B")) + " |")
    rec = read_bytes("results/recorded/validation/reports/ablations.md").decode("utf-8").split("\n")
    return ("ablations.md", L, rec[:len(L)])


def fsor_dnpg_hfix_block():
    d = raw("results/recorded/validation/dnpg_hfix.json")
    W, E, ok, verdict, chk = d["weights"], d["rule"], d["ok"], d["verdict"], d["reproduction_max_abs_diff"]
    S = d["summaries"]
    L = ["# MINI-DNPG Part 1: inductive fusion H_fix on validation (DNPG seeds 0-2, max_acc.pth)", "",
         "Pre-registration: research/dnpg/DNPG_SETUP.md section 10 (commit 0c2a8db, before any valB computation for these rows). "
         "H_fix = D + w · CCP-A, K_fix = D + w2 · COSF; w = pooled std(D) / pooled std(CCP-A), w2 = pooled std(D) / pooled std(COSF), ddof 0 over "
         "all 1,000 x 150 valA query scores, one constant per seed and shot. H, K = per-episode z (query statistics). D = DNPG own score "
         "(selected on validation classes; optimistic vs training-free rows). Accuracy = DNPG classifier for every row. 1,000 episodes per file.", "",
         f"Reproduction of the H-COMB per-episode numbers (D, CCP-A, COSF, H, K AUROC and D accuracy, all seeds, shots, valA / valB): max |difference| {chk:.2e}.", "",
         "## Weights (valA)", "", "| Seed | Shot | std D | std CCP-A | std COSF | w | w2 |", "|---|---|---|---|---|---|---|"]
    for seed in (0, 1, 2):
        for shot in (1, 5):
            w = W[f"seed{seed}_{shot}shot"]
            L.append(f"| {seed} | {shot} | {w['std_D']:.4f} | {w['std_CCPA']:.4f} | {w['std_COSF']:.4f} | {w['w']:.4f} | {w['w2']:.4f} |")
    for s, name in (("B", "valB (rule set)"), ("A", "valA (information; w / w2 fitted here)")):
        for shot in (1, 5):
            L += ["", f"## {name}, {shot}-shot (AUROC ± 95% CI; accuracy = DNPG classifier)", "", "| Row | seed 0 | seed 1 | seed 2 | mean seeds 1-2 |", "|---|---|---|---|---|"]
            for r in DNPG_ROWS:
                v = [S[f"seed{n}_{shot}shot_{s}"][r] for n in (0, 1, 2)]
                lab = r + (" (per-episode z, query statistics)" if r in ("H", "K") else "")
                L.append(f"| {lab} | " + " | ".join(f"{x['auroc']:.2f} ± {x['auroc_ci95']:.2f}" for x in v) + f" | {np.mean([v[1]['auroc'], v[2]['auroc']]):.2f} |")
            acc = [S[f"seed{n}_{shot}shot_{s}"]["D"] for n in (0, 1, 2)]
            L.append("| accuracy (all rows) | " + " | ".join(f"{x['acc']:.2f} ± {x['acc_ci95']:.2f}" for x in acc) + " | |")
    e = lambda k, s: E[str(s)][k]  # noqa: E731
    L += ["", "## Rule (valB)", "", "| | 1-shot | 5-shot | Needed | Holds |", "|---|---|---|---|---|",
          f"| E1 mean seeds 0-2 (H − H_fix) | {e('E1_mean_H_minus_Hfix_valB_s012', 1):+.2f} ({', '.join(f'{x:+.2f}' for x in e('E1_per_seed', 1))}) | "
          f"{e('E1_mean_H_minus_Hfix_valB_s012', 5):+.2f} ({', '.join(f'{x:+.2f}' for x in e('E1_per_seed', 5))}) | ≤ 0.20 | {'yes' if ok['E1'] else 'no'} |",
          f"| E2 mean seeds 1-2 (H_fix − D) | {e('E2_mean_Hfix_minus_D_valB_s12', 1):+.2f} ({', '.join(f'{x:+.2f}' for x in e('E2_per_seed', 1))}) | "
          f"{e('E2_mean_Hfix_minus_D_valB_s12', 5):+.2f} ({', '.join(f'{x:+.2f}' for x in e('E2_per_seed', 5))}) | ≥ +0.50, each seed > 0 | {'yes' if ok['E2'] else 'no'} |",
          f"| E3 mean seeds 1-2 (H_fix − K_fix) | {e('E3_mean_Hfix_minus_Kfix_valB_s12', 1):+.2f} ({', '.join(f'{x:+.2f}' for x in e('E3_per_seed', 1))}) | "
          f"{e('E3_mean_Hfix_minus_Kfix_valB_s12', 5):+.2f} ({', '.join(f'{x:+.2f}' for x in e('E3_per_seed', 5))}) | ≥ +0.30 | {'yes' if ok['E3'] else 'no'} |",
          "", f"**{verdict}**" + (": H_fix becomes the MINI-DNPG headline; H stays as a row labelled 'per-episode z (query statistics)'." if verdict == "PASS" else ": stop after Part 1.")]
    rec = read_bytes("results/recorded/validation/reports/dnpg_hfix.md").decode("utf-8").split("\n")
    return ("dnpg_hfix.md", L, rec[:len(L)])


def fsor_snatcher_val_block():
    v = raw("results/recorded/validation/snatcher.json")["val"]
    res, rules, verdict, W = v["results"], v["rules"], v["verdict"], v["weights"]
    lab = {"S": "S (SnaTCHer)", "H_fix": "**H_fix = S + w · CCP-A**", "K_fix": "K_fix = S + w2 · COSF", "Dist": "Dist (FEAT, −max logit)"}
    f = lambda x: f"{x['auroc']:.2f} ± {x['auroc_ci95']:.2f} / {x['acc']:.2f} ± {x['acc_ci95']:.2f}"  # noqa: E731
    L = ["# SnaTCHer-F host on validation (mini valA / valB, tiered tvA / tvB)", "",
         "Pre-registration: research/snatcher/SNATCHER_SETUP.md section 2 (commit be8d61a, before any valB / tvB scoring). FEAT's released "
         "meta-trained ResNet-12 per shot (one model per shot, no seeds); tiered input BGR. 1,000 episodes per file (5 known + 5 unknown classes, "
         "15 queries each). Cells: AUROC ± 95% CI / accuracy ± 95% CI. Accuracy: FEAT's classifier for S, H_fix, K_fix, Dist; own argmax for "
         "CCP-A and COSF alone. w / w2 frozen from set A (research/status/snatcher_weights.json).", "",
         "## Weights (set A, frozen before set B)", "", "| Dataset | Shot | w | w2 | frozen at |", "|---|---|---|---|---|"]
    for ds in rules:
        for sh in (1, 5):
            w = W[f"{ds}_{sh}shot"]
            L.append(f"| {ds} | {sh} | {w['w']:.4f} | {w['w2']:.4f} | {w['frozen_at']} |")
    for ds in rules:
        a, b = ("valA", "valB") if ds == "mini" else ("tvA", "tvB")
        for sh in (1, 5):
            A, B = res[f"{ds}_{sh}shot_A"], res[f"{ds}_{sh}shot_B"]
            L += ["", f"## {ds}, {sh}-shot", "", f"| Row | {a} | {b} |", "|---|---|---|"]
            for r in ("S", "H_fix", "K_fix", "CCP-A", "COSF", "Dist"):
                L.append(f"| {lab.get(r, r)} | {f(A['rows'][r])} | {f(B['rows'][r])} |")
            for d, name in (("Hfix_minus_S", "H_fix − S (paired)"), ("Hfix_minus_Kfix", "H_fix − K_fix (paired)")):
                L.append(f"| {name} | {A['paired'][d]['mean']:+.2f} ± {A['paired'][d]['ci95']:.2f} | {B['paired'][d]['mean']:+.2f} ± {B['paired'][d]['ci95']:.2f} |")
    L += ["", "## Pre-registered rule (values 1-shot / 5-shot)", "",
          "| Dataset | S1 H_fix − S on B (≥ +0.50) | S2 H_fix − K_fix on B (≥ +0.30) | S3 H_fix − S on A (> 0) | Verdict |", "|---|---|---|---|---|"]
    for ds in rules:
        vv, ok = rules[ds]["values"], rules[ds]["ok"]
        cell = lambda k: f"{vv[k]['1']:+.2f} / {vv[k]['5']:+.2f} ({'holds' if ok[k] else 'FAILS'})"  # noqa: E731
        L.append(f"| {ds} | {cell('S1')} | {cell('S2')} | {cell('S3')} | **{verdict[ds]}** |")
    rec = read_bytes("results/recorded/validation/reports/snatcher_val.md").decode("utf-8").split("\n")
    return ("snatcher_val.md", L, rec[:len(L)])


def fsor_dnpg_tiered_block():
    d = raw("results/recorded/validation/dnpg_tiered_eval.json")
    seeds, W, rule, ok, verdict, res = d["seeds"], d["weights"], d["rule_seed0"], d["ok"], d["verdict"], d["summaries"]
    lab = {"D": "D (DNPG own; selected on validation classes)", "H_fix": "**H_fix**", "H": "H (per-episode z, query statistics)", "K": "K (per-episode z, query statistics)"}
    L = [f"# Tiered DNPG, seed{'s ' + ', '.join(map(str, seeds)) if len(seeds) > 1 else ' 0'} (validation only: tvA / tvB)", "",
         "Pre-registration: research/dnpg/DNPG_SETUP.md section 11 (commit 8cfd64b, unchanged). Released TANE tiered backbone (stored epoch 81, "
         "consistent with test-selected max_meta); DNPG Phase 2 + meta-learning, BGR npz, per-epoch selection on the 97 tiered val classes "
         "(selected on validation classes). 1,000 episodes per file (5 known + 5 unknown classes, 15 queries each). H_fix = D + w · CCP-A, "
         "K_fix = D + w2 · COSF, w / w2 frozen from tvA before tvB was scored (research/status/dnpg_tiered_weights.json). Accuracy = DNPG classifier.", "",
         "## Weights (tvA, frozen)", "", "| Seed | Checkpoint | Shot | w | w2 | frozen at |", "|---|---|---|---|---|---|"]
    for seed in seeds:
        for tag in ("max_acc", "last"):
            for sh in (1, 5):
                w = W[f"seed{seed}_{tag}_{sh}shot"]
                L.append(f"| {seed} | {tag} | {sh} | {w['w']:.4f} | {w['w2']:.4f} | {w['frozen_at']} |")
    for tag in ("max_acc", "last"):
        for sh in (1, 5):
            L += ["", f"## {tag}{' (rule checkpoint)' if tag == 'max_acc' else ' (last epoch, information)'}, {sh}-shot", "",
                  "| Seed (epoch) | Row | tvA AUROC | tvB AUROC | accuracy tvA / tvB |", "|---|---|---|---|---|"]
            for seed in seeds:
                A, B = res[f"seed{seed}_{tag}_{sh}shot_A"], res[f"seed{seed}_{tag}_{sh}shot_B"]
                for r in DNPG_ROWS:
                    a, b = A["summary"][r], B["summary"][r]
                    L.append(f"| {seed} ({A['epoch']}) | {lab.get(r, r)} | {a['auroc']:.2f} ± {a['auroc_ci95']:.2f} | {b['auroc']:.2f} ± {b['auroc_ci95']:.2f} | {a['acc']:.2f} / {b['acc']:.2f} |")
                for dd, name in (("Hfix_minus_D", "H_fix − D (paired)"), ("Hfix_minus_Kfix", "H_fix − K_fix (paired)")):
                    L.append(f"| {seed} | {name} | {A['paired'][dd]['mean']:+.2f} ± {A['paired'][dd]['ci95']:.2f} | {B['paired'][dd]['mean']:+.2f} ± {B['paired'][dd]['ci95']:.2f} | |")
    r1, r5 = rule["1"], rule["5"]
    L += ["", "## Pre-registered rule (seed 0, max_acc.pth)", "", "| | 1-shot | 5-shot | Needed | Holds |", "|---|---|---|---|---|",
          f"| T1 H_fix − D, tvB | {r1['T1_Hfix_minus_D_tvB']:+.2f} | {r5['T1_Hfix_minus_D_tvB']:+.2f} | ≥ +0.50 | {'yes' if ok['T1'] else 'no'} |",
          f"| T2 H_fix − K_fix, tvB | {r1['T2_Hfix_minus_Kfix_tvB']:+.2f} | {r5['T2_Hfix_minus_Kfix_tvB']:+.2f} | ≥ +0.30 | {'yes' if ok['T2'] else 'no'} |",
          f"| T3 H_fix − D, tvA | {r1['T3_Hfix_minus_D_tvA']:+.2f} | {r5['T3_Hfix_minus_D_tvA']:+.2f} | > 0 | {'yes' if ok['T3'] else 'no'} |",
          "", f"**{verdict}**" + (" → seeds 1-2, then a TIERED-DNPG bundle part." if verdict == "GO" else ": no tiered DNPG (seeds 1-2 not trained).")]
    rec = read_bytes("results/recorded/validation/reports/dnpg_tiered_seed0.md").decode("utf-8").split("\n")
    return ("dnpg_tiered_seed0.md", L, rec[:len(L)])


def render_fsor() -> list[tuple[str, list[str], list[str]]]:
    """Rounds (i)-(iv) and their validation reports: blocks of recorded markdown re-rendered line by line."""
    blocks = fsor_results_blocks() + fsor_paired_blocks() + [fsor_ablations_block(), fsor_dnpg_hfix_block(), fsor_snatcher_val_block(),
                                                              fsor_dnpg_tiered_block()]
    for name, rendered, _ in blocks:
        write_text("fsor", name.replace(" ", "_").replace(".md_", "_") + ("" if name.endswith(".md") else ".md"), rendered)
    return blocks


# ==================================================================================================================== round (v): baselines
BL_GROUPS = {"frozen FEAT mini": ["frozen_feat_mini"], "frozen FEAT tiered": ["frozen_feat_tiered"], "SnaTCHer-F mini": ["snatcher_f_mini"],
             "DNPG mini": ["dnpg_mini_seed0", "dnpg_mini_seed1", "dnpg_mini_seed2"]}


def bl_fmt(x):
    if "auroc" in x and isinstance(x["auroc"], dict):
        return f"{x['auroc']['mean']:.2f} ± {x['auroc']['std']:.2f} / {x['acc']['mean']:.2f}"
    return f"{x['auroc']:.2f} ± {x['auroc_ci95']:.2f} / {x['acc']:.2f}"


def render_baselines() -> list[Path]:
    sel, val = raw("results/baselines/selection_valA.json"), raw("results/baselines/validation.json")
    L = ["# Training-free baselines on validation B (tvB)", "",
         "Pre-registration: docs/preregistration/baselines.md (committed before any baseline was scored). Hyperparameters chosen on validation A "
         "(tvA) and frozen (results/baselines/selection_valA.json). Cells: AUROC ± 95% CI / accuracy (DNPG: AUROC mean ± std over seeds 0-2 / "
         "mean accuracy). Paired: per-episode AUROC difference, mean ± 95% CI over 1,000 episodes (DNPG: averaged over seeds per episode). "
         "'host' = COSF (frozen FEAT), S (SnaTCHer-F) or D (DNPG); 'CCP' = CCP-A (frozen FEAT) or H_fix (trained hosts). No go / no-go.", ""]
    for g in BL_GROUPS:
        L += [f"## {g}", "", "| Row | chosen on valA | 1-shot AUROC / acc | 5-shot AUROC / acc | − host (1 / 5) | − CCP (1 / 5) |", "|---|---|---|---|---|---|"]
        for r in ["host", "CCP"] + (["COS"] if "frozen" in g else []):
            L.append(f"| {r} | | {bl_fmt(val['valB'][g]['1'][r])} | {bl_fmt(val['valB'][g]['5'][r])} | | |")
        for fam, v in sel["chosen"][g].items():
            p1, p5 = val["paired_valB"][g]["1"][v], val["paired_valB"][g]["5"][v]
            pr = lambda a, b: f"{a['mean']:+.2f} ± {a['ci95']:.2f} / {b['mean']:+.2f} ± {b['ci95']:.2f}"  # noqa: E731
            L.append(f"| {fam} | `{v}` | {bl_fmt(val['valB'][g]['1'][v])} | {bl_fmt(val['valB'][g]['5'][v])} | {pr(p1['minus_host'], p5['minus_host'])} | "
                     f"{pr(p1['minus_CCP'], p5['minus_CCP'])} |")
        L.append("")
    outs = [write_text("baselines", "validation.md", L)]
    # test.md
    t = raw("results/baselines/test.json")
    b, token = bundle_block("baselines", paper_names=False)
    if t["token"] != token:
        raise SystemExit(f"test.json token {t['token']} != bundle token {token}")
    z, npz = test_npz("baselines")
    chosen = b["chosen"]
    worst, P = 0.0, {}
    for g, hosts in BL_GROUPS.items():
        P[g] = {}
        for sh in ("1", "5"):
            P[g][sh] = {}
            for v in chosen[g].values():
                P[g][sh][v] = {}
                for ref in ("host", "CCP"):
                    d = np.mean([z[f"{h}__{sh}shot__{v}"] - z[f"{h}__{sh}shot__{ref}"] for h in hosts], axis=0)
                    p = metrics.paired(d, np.zeros_like(d))
                    rec = t["paired_test"][g][sh][v][f"minus_{ref}"]
                    worst = max(worst, abs(p["mean"] - rec["mean"]), abs(p["ci95"] - rec["ci95"]), abs(p["frac_gain"] - rec["frac_gain"]))
                    P[g][sh][v][ref] = p
    if worst > 1e-9:
        raise SystemExit(f"paired statistics recomputed from the per-episode file differ from test.json by {worst:.2e}")
    pr = lambda a, b_: (f"{a['mean']:+.2f} ± {a['ci95']:.2f} ({a['frac_gain'] * 100:.0f}%) / "  # noqa: E731
                        f"{b_['mean']:+.2f} ± {b_['ci95']:.2f} ({b_['frac_gain'] * 100:.0f}%)")
    started = json.loads(read_bytes("bundles/FINAL_TEST_BASELINES_STARTED.json"))["started"]
    L = ["# Training-free baselines on TEST (single pass of bundle part yaml-fsor-baselines)", "",
         f"Bundle token {token} (docs/preregistration/FINAL_BUNDLE_baselines.md), approved in bundles/FINAL_APPROVED; one pass, started "
         f"{started}, done {t['when']} ({t['gpu_minutes']:.1f} GPU-minutes). "
         "600 test episodes per shot (5 known + 5 unknown classes, 15 queries per class); frozen hyperparameters and weights from validation A. "
         "Rendered from results/baselines/test.json and the per-episode file test_episodes.npz "
         f"(sha256 {sha256_file(npz)}); no test data was loaded to write this file.", "",
         "Cells: AUROC ± 95% CI / accuracy (DNPG: AUROC mean ± std over seeds 0-2 / mean accuracy). Paired: per-episode AUROC difference, mean ± 95% CI "
         "over 600 episodes (% of episodes improved) for 1-shot / 5-shot (DNPG: differences averaged over seeds per episode). 'host' = COSF (frozen FEAT), "
         "S (SnaTCHer-F) or D (DNPG); 'CCP' = CCP-A (frozen FEAT) or H_fix (trained hosts). B9 is a TRANSDUCTIVE REFERENCE. Decisions after this "
         "test: none.", ""]
    for g in BL_GROUPS:
        L += [f"## {g}", "", "| Row | chosen on valA | 1-shot AUROC / acc | 5-shot AUROC / acc | − host (1 / 5) | − CCP (1 / 5) |", "|---|---|---|---|---|---|"]
        for r in ["host", "CCP"] + (["COS"] if "frozen" in g else []):
            if r in t["test"][g]["1"]:
                L.append(f"| {r} | | {bl_fmt(t['test'][g]['1'][r])} | {bl_fmt(t['test'][g]['5'][r])} | | |")
        for fam, v in chosen[g].items():
            L.append(f"| {fam} | `{v}` | {bl_fmt(t['test'][g]['1'][v])} | {bl_fmt(t['test'][g]['5'][v])} | "
                     f"{pr(P[g]['1'][v]['host'], P[g]['5'][v]['host'])} | {pr(P[g]['1'][v]['CCP'], P[g]['5'][v]['CCP'])} |")
        L.append("")
    outs.append(write_text("baselines", "test.md", L))
    return outs


# ==================================================================================================================== round (vi): variants
V_FAMILIES = ["V1 QW", "V2 ASK", "V3 SHC", "V4 FW"]
V_GROUPS = BL_GROUPS
V_FROZEN = ["frozen FEAT mini", "frozen FEAT tiered"]
V_TRAINED = ["SnaTCHer-F mini", "DNPG mini"]
V_CELLS = [(g, sh) for g in V_GROUPS for sh in (1, 5)]
V4_CELLS = [(g, sh) for g in V_TRAINED for sh in (1, 5)]


def vkey(c):
    return f"{c[0]}|{c[1]}"


def variants_validation_md(out, s):
    cells = out[f"cells_val{s}"]
    title = {"B": "validation B (tvB): gates G1-G3", "C": "validation C (fresh episodes): gate G4 and promotion"}[s]
    L = [f"# Method variants on {title}", "",
         "Pre-registration: docs/prereg_variants.md. Settings chosen on validation A (results/variants/selection_valA.json) and frozen. "
         "REF = CCP-A (frozen FEAT) or H_fix = D + w CCP-A (trained hosts). Cells: paired per-episode AUROC change vs REF, mean ± 95% CI over "
         "1,000 episodes (DNPG: differences averaged over seeds 0-2 per episode); accuracy change in points (frozen FEAT; trained hosts keep "
         "their classifier).", "", f"Chosen on valA: {json.dumps(out['chosen'])}.", ""]
    if s == "B":
        L += [f"V1 status: {out['v1_status']}.", ""]
    L += ["| Cell | " + " | ".join(V_FAMILIES) + " |", "|---|" + "---|" * len(V_FAMILIES)]
    for c in V_CELLS:
        row = []
        for f in V_FAMILIES:
            x = cells[vkey(c)].get(f)
            if x is None:
                row.append("n/a")
            else:
                p = x["paired_auroc_vs_REF"]
                acc = f", acc {x['acc_change']:+.2f}" if c[0] in V_FROZEN else ""
                row.append(f"{p['mean']:+.2f} ± {p['ci95']:.2f}{acc}")
        L.append(f"| {c[0]} {c[1]}-shot | " + " | ".join(row) + " |")
    L.append("")
    if s == "B":
        L += ["| Family | mean gain (G1 >= +0.30) | min cell (G2 >= -0.20) | min frozen acc change (G3 >= -0.20) | G1-G3 |", "|---|---|---|---|---|"]
        for f, g in out["gates_valB"].items():
            a = "-" if g["min_frozen_acc_change"] is None else f"{g['min_frozen_acc_change']:+.2f}"
            L.append(f"| {f} | {g['mean_gain']:+.3f} {'PASS' if g['G1'] else 'FAIL'} | {g['min_cell_gain']:+.3f} {'PASS' if g['G2'] else 'FAIL'} | "
                     f"{a} {'PASS' if g['G3'] else 'FAIL'} | {'PASS' if g['pass_G1_G3'] else 'FAIL'} |")
        L += ["", f"Candidate (highest valB mean gain among G1-G3 passes; V1 excluded if NO-GO): {out['candidate']}."]
    else:
        L += ["| Family | mean gain (>= +0.15) | min cell (>= -0.20) | G4 |", "|---|---|---|---|"]
        for f, g in out["gate_G4_valC"].items():
            L.append(f"| {f} | {g['mean_gain']:+.3f} | {g['min_cell_gain']:+.3f} | {'PASS' if g['G4'] else 'FAIL'} |")
        L += ["", f"Candidate from valB: {out['candidate_from_valB']}. Decision: {out['decision']}."]
    return L


def render_variants() -> list[Path]:
    from ccp import variants as V
    outs = [write_text("variants", "validation_valB.md", variants_validation_md(raw("results/variants/validation_valB.json"), "B")),
            write_text("variants", "validation_valC.md", variants_validation_md(raw("results/variants/validation_valC.json"), "C"))]
    t = raw("results/variants/test.json")
    b, token = bundle_block("variants", paper_names=False)
    if t["token"] != token:
        raise SystemExit(f"test.json token {t['token']} != bundle token {token}")
    z, npz = test_npz("variants")
    worst = 0.0
    cells = t["cells_test"]
    for (g, sh) in V_CELLS:
        hosts, c = V_GROUPS[g], cells[vkey((g, sh))]
        ref = {h: (z[f"{h}__{sh}shot__REF__auroc"], z[f"{h}__{sh}shot__REF__acc"]) for h in hosts}
        for h in hosts:
            sm = metrics.summarise(*ref[h])
            worst = max(worst, abs(sm["auroc"] - c["REF"]["per_host"][h]["auroc"]), abs(sm["acc"] - c["REF"]["per_host"][h]["acc"]))
        for f, x in c.items():
            if f == "REF":
                continue
            v = {h: (z[f"{h}__{sh}shot__{x['variant']}__auroc"], z[f"{h}__{sh}shot__{x['variant']}__acc"]) for h in hosts}
            p = V.cell_paired([v[h][0] for h in hosts], [ref[h][0] for h in hosts])
            acc = float(np.mean([v[h][1].mean() - ref[h][1].mean() for h in hosts]))
            rec = x["paired_auroc_vs_REF"]
            worst = max(worst, abs(p["mean"] - rec["mean"]), abs(p["ci95"] - rec["ci95"]), abs(p["frac_gain"] - rec["frac_gain"]),
                        abs(acc - x["acc_change"]))
    if worst > 1e-9:
        raise SystemExit(f"statistics recomputed from the per-episode file differ from test.json by {worst:.2e}")
    started = json.loads(read_bytes("bundles/FINAL_TEST_VARIANTS_STARTED.json"))["started"]
    L = ["# Method variants on TEST (single pass of bundle part yaml-fsor-variants)", "",
         f"Bundle token {token} (docs/preregistration/FINAL_BUNDLE_variants.md), approved in bundles/FINAL_APPROVED; one pass, started {started}, "
         f"done {t['when']} ({t['gpu_minutes']:.1f} GPU-minutes), code at {t['git_head'][:7]}. Pre-registration: docs/prereg_variants.md. Settings "
         "chosen on validation A and frozen; V1 QW was promoted on validation (G1-G3 on valB, G4 on valC) before this pass. 600 test episodes per "
         "shot. Rendered from results/variants/test.json and the per-episode file "
         f"(sha256 {sha256_file(npz)}); no test data was loaded to write this file.", "",
         f"Reference check: REF reproduces the recorded single-pass test AUROC and accuracy of every host and shot: "
         f"{'yes' if t['reference_reproduced'] else 'NO'} (max |difference| {t['reference_max_abs_diff']:.1e}).", "",
         "REF = CCP-A (frozen FEAT) or H_fix = D + w CCP-A (trained hosts). Cells: paired per-episode AUROC change vs REF, mean ± 95% CI over "
         "600 episodes (% of episodes improved; DNPG: differences averaged over seeds 0-2 per episode); accuracy change in points (frozen FEAT; "
         "trained hosts keep their classifier). Decisions after this test: none.", "", f"Chosen on valA: {json.dumps(b['chosen'])}.", "",
         "| Cell | REF AUROC | " + " | ".join(V_FAMILIES) + " |", "|---|---|" + "---|" * len(V_FAMILIES)]
    for c in V_CELLS:
        x = cells[vkey(c)]
        ref_auroc = np.mean([s_["auroc"] for s_ in x["REF"]["per_host"].values()])
        row = []
        for f in V_FAMILIES:
            if f not in x:
                row.append("n/a")
                continue
            p = x[f]["paired_auroc_vs_REF"]
            acc = f", acc {x[f]['acc_change']:+.2f}" if c[0] in V_FROZEN else ""
            row.append(f"{p['mean']:+.2f} ± {p['ci95']:.2f} ({p['frac_gain'] * 100:.0f}%){acc}")
        L.append(f"| {c[0]} {c[1]}-shot | {ref_auroc:.2f} | " + " | ".join(row) + " |")
    L += ["", "| Family | applicable cells | mean change | min cell | max cell |", "|---|---|---|---|---|"]
    for f in V_FAMILIES:
        cl = V4_CELLS if f == "V4 FW" else V_CELLS
        g = [cells[vkey(c)][f]["paired_auroc_vs_REF"]["mean"] for c in cl]
        L.append(f"| {f}{' (promoted on validation)' if f == b['promotion']['promoted'] else ''} | {len(cl)} | {np.mean(g):+.3f} | {min(g):+.3f} | {max(g):+.3f} |")
    L += ["", "Descriptive only: the validation gates G1-G4 are not applied to test, and no decision is taken after this pass."]
    outs.append(write_text("variants", "test.md", L))
    return outs


# ==================================================================================================================== round (vii): final round
FR_SCG_CELLS = ("mini", "tiered", "CUB", "CIFAR-FS", "CUB@target", "CUB@target-K154")


def fr_fmt(x, k):
    v = x[k]
    if isinstance(v, dict):
        return f"{v['mean']:.2f} ± {v['std']:.2f}"
    return f"{v:.2f} ± {x[k + '_ci95']:.2f}"


def fr_md_table(title: str, table: dict, ref: str, note: str) -> list[str]:
    L = [f"## {title}", "", note, ""]
    for g, t in table.items():
        for sh in ("1", "5"):
            L += [f"### {g}, {sh}-shot", "", "| Row | AUROC | acc | OSCR | FPR95 | AUROC − ref | OSCR − ref | FPR95 − ref |", "|---|---|---|---|---|---|---|---|"]
            for r, x in t[sh]["rows"].items():
                p = t[sh]["paired"].get(ref, {}).get(r)
                pr = (lambda k: f"{p[k]['mean']:+.2f} ± {p[k]['ci95']:.2f} ({p[k]['frac_gain'] * 100:.0f}%)") if p else (lambda k: "")
                L.append(f"| {r} | {fr_fmt(x, 'auroc')} | {fr_fmt(x, 'acc')} | {fr_fmt(x, 'oscr')} | {fr_fmt(x, 'fpr95')} | {pr('auroc')} | {pr('oscr')} | {pr('fpr95')} |")
            L.append("")
    return L


def matplotlib_defaults(fn):
    """Draw a recorded PNG with matplotlib's default style (as when it was recorded), whatever style the process has set, and restore it."""
    import functools

    @functools.wraps(fn)
    def run(*a, **k):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib._mathtext as mathtext
        shrink = mathtext.SHRINK_FACTOR
        with matplotlib.rc_context():
            matplotlib.rcdefaults()
            mathtext.SHRINK_FACTOR = 0.7
            try:
                return fn(*a, **k)
            finally:
                mathtext.SHRINK_FACTOR = shrink
    return run


@matplotlib_defaults
def fr_scg_figure(scg: dict, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for name, mk in zip(FR_SCG_CELLS, "osD^vP"):
        for sh, col in zip((1, 5), ("tab:blue", "tab:orange")):
            x = scg[f"{name}|{sh}"]
            ax.scatter(x["std_gap"], x["ccpa_minus_cosf_auroc"], marker=mk, color=col, s=28)
            ax.annotate(f"{name} {sh}-shot", (x["std_gap"], x["ccpa_minus_cosf_auroc"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("D1 standardised gap (mean over bins)", fontsize=7)
    ax.set_ylabel("AUROC gain over COSF", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(title + ": gain of CCP-A (CUB@target points: CCP-A@target) over COSF; descriptive, no line fitted", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def render_final_round() -> list[Path]:
    outs = []
    a = raw("results/final_round/cifarfs_valB.json")
    L = ["# CIFAR-FS validation B (final round, (A))", "", "Baseline settings chosen on valA (cifarfs_selection_valA.json). Preview only; every "
         "registered row goes to test whatever validation shows.", ""]
    L += fr_md_table("CIFAR-FS valB", a["table"], "host", "host = COSF. Paired: per-episode difference vs host, mean ± 95% CI (% of episodes improved; "
                     "FPR95: % with a lower value).")
    outs.append(write_text("final_round", "cifarfs_valB.md", L))
    bp = raw("results/final_round/cub_val_preview.json")
    L = ["# mini-ImageNet -> CUB, validation preview (final round, (B))", "", "Every constant fixed on mini-ImageNet. CCP-A@target / H_fix@target "
         "use the CUB training-class bank: an ANALYSIS row in a different setting (unlabelled target-domain images), not the inductive "
         "cross-domain result. Preview only.", ""]
    L += fr_md_table("CUB val", bp["table"], "host", "host = COSF (frozen FEAT), S (SnaTCHer-F), D (DNPG; mean ± std over seeds).")
    outs.append(write_text("final_round", "cub_val_preview.md", L))
    c = raw("results/final_round/recorded_valA.json")
    rep = c["reproduction_of_recorded_dry_runs"]
    L = ["# Recorded rows with OSCR / FPR95 on validation A (final round, (C))", "",
         f"Reproduction of the recorded validation dry runs (AUROC and accuracy of {rep['compared_rows']} rows): "
         f"{'exact' if rep['reproduced'] else 'FAILED'} (max |difference| {rep['max_abs_diff']:.1e}).", ""]
    L += fr_md_table("valA / tvA", c["table"], "host", "host = COS (frozen backbones) or S / D (trained hosts). V1 / V1fused: the pre-registered extension.")
    outs.append(write_text("final_round", "recorded_valA.md", L))
    p = out_path("final_round", "d1_val.png")
    fr_scg_figure(raw("results/final_round/d1_val.json")["d1"], p, "Validation")
    outs.append(p)
    # test.md and d1_test.png
    t = raw("results/final_round/test.json")
    b, token = bundle_block("final_round", paper_names=False)
    if t["token"] != token:
        raise SystemExit(f"test.json token {t['token']} != bundle token {token}")
    z, _ = test_npz("final_round")
    worst = 0.0
    for k, s in t["summaries"].items():
        pt, h, sh, _, r = k.split("|")
        got = FR.summary({m: z[f"{pt}__{h}__{sh}shot__{r}__{m}"] for m in ("auroc", "acc", "oscr", "fpr95")})
        worst = max(worst, *(abs(got[m] - s[m]) for m in s if isinstance(s[m], float)))
    if worst > 1e-9:
        raise SystemExit(f"summaries recomputed from the per-episode file differ from test.json by {worst:.2e}")
    rep = t["reproduction_of_recorded_test"]
    L = ["# Final evaluation round on TEST (single pass of bundle part yaml-fsor-final-round)", "",
         f"Bundle token {token}; done {t['when']} ({t['gpu_minutes']:.1f} GPU-minutes), code at {t['git_head'][:7]}. Pre-registration: "
         "docs/preregistration/final_round.md. Decisions after this test: none.", "",
         f"Technical checks: part C reproduces the recorded test AUROC and accuracy of {rep['compared_rows']} rows: "
         f"{'yes, exactly' if rep['reproduced'] else 'NO: failed reproduction; the new metrics are not interpreted'} (max |difference| "
         f"{rep['max_abs_diff']:.1e}); NaN found: {t['nan_found']}.", ""]
    L += fr_md_table("(A) CIFAR-FS", t["tables"]["A"], "host", "host = COSF.")
    L += fr_md_table("(B) mini-ImageNet -> CUB", t["tables"]["B"], "host", "Constants from mini. CCP-A@target / H_fix@target: ANALYSIS rows "
                     "with a target-domain bank (unlabelled CUB training-class images), not the inductive cross-domain result.")
    L += fr_md_table("(C) Recorded rows with OSCR / FPR95", t["tables"]["C"], "host", "host = COS (frozen backbones) or S / D.")
    L += ["## (D) D1 (descriptive)", "", "| Dataset | shot | standardised gap | raw gap | CCP-A − COSF AUROC |", "|---|---|---|---|---|"]
    for k, x in t["d1"].items():
        if "raw_gap" in x:
            n, sh = k.split("|")
            L.append(f"| {n} | {sh} | {x['std_gap']:+.3f} | {x['raw_gap']:+.4f} | {x['ccpa_minus_cosf_auroc']:+.2f} |")
    outs.append(write_text("final_round", "test.md", L))
    p = out_path("final_round", "d1_test.png")
    fr_scg_figure(t["d1"], p, "Test")
    outs.append(p)
    return outs


# ==================================================================================================================== round (viii): foundation round
FM_BACKBONES = ("clip_b16", "dinov2_s14", "dinov2_b14", "dinov2_l14")
FM_DATASETS = ("mini", "tiered", "cifarfs", "cub")


def fm_cell(ds, b, sh):
    return f"{ds}|{b}|{sh}"


@matplotlib_defaults
def fm_scg_figure(points: dict, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    mk = dict(zip(FM_DATASETS, "osD^"))
    col = dict(zip(FM_BACKBONES, ("tab:red", "tab:blue", "tab:green", "tab:purple")))
    for k, (x, y) in points.items():
        ds, b, sh = k.split("|")
        ax.scatter(x, y, marker=mk[ds], color=col[b], s=18 if sh == "1" else 34, alpha=0.85)
    for ds in FM_DATASETS:
        ax.scatter([], [], marker=mk[ds], color="0.4", label=ds)
    for b in FM_BACKBONES:
        ax.scatter([], [], marker="s", color=col[b], label=b)
    ax.legend(fontsize=5, ncol=2)
    ax.set_xlabel("D1 standardised gap", fontsize=7)
    ax.set_ylabel("AUROC gain of CCP-A over COSF", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(f"{title}: 32 cells (small = 1-shot, large = 5-shot); descriptive", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def render_foundation_round() -> list[Path]:
    outs = []
    sel = raw("results/foundation_round/selection_valA.json")
    scg = sel["d1_valA"]
    p = out_path("foundation_round", "d1_val.png")
    fm_scg_figure({k: (x["std_gap"], x["gain_ccpa_minus_cosf"]) for k, x in scg.items()}, p, "Validation A")
    outs.append(p)
    t = raw("results/foundation_round/test.json")
    b, token = bundle_block("foundation_round", paper_names=False)
    if t["token"] != token:
        raise SystemExit(f"test.json token {t['token']} != bundle token {token}")
    z, _ = test_npz("foundation_round")
    worst = 0.0
    for k, s in t["summaries"].items():
        n, st, sh, r = k.split("|")
        got = FR.summary({m: z[f"{n}__{st}__{sh}shot__{r}__{m}"] for m in ("auroc", "acc", "oscr", "fpr95")})
        worst = max(worst, *(abs(got[m] - s[m]) for m in s if isinstance(s[m], float)))
    if worst > 1e-9:
        raise SystemExit(f"summaries recomputed from the per-episode file differ from test.json by {worst:.2e}")
    host = lambda ds, bb: f"{ds}_{bb}"  # noqa: E731
    diffs = {fm_cell(ds, bb, sh): z[f"{host(ds, bb)}__T__{sh}shot__CCP-A__auroc"] - z[f"{host(ds, bb)}__T__{sh}shot__COSF__auroc"]
             for bb in FM_BACKBONES for ds in FM_DATASETS for sh in (1, 5)}
    h1 = stats.h1_table(diffs)
    scg_t = t["d1_test"]
    keys = list(diffs)
    h2 = stats.spearman_perm([scg_t[k]["std_gap"] for k in keys], [scg_t[k]["gain_ccpa_minus_cosf"] for k in keys])
    h2b = stats.spearman_perm([b["d1_valA_predictions"][k] for k in keys], [scg_t[k]["gain_ccpa_minus_cosf"] for k in keys])
    strata = ["|".join((k.split("|")[0], k.split("|")[2])) for k in keys]
    h2s = stats.within_group_spearman_perm([scg_t[k]["std_gap"] for k in keys], [scg_t[k]["gain_ccpa_minus_cosf"] for k in keys], strata)
    h2bs = stats.within_group_spearman_perm([b["d1_valA_predictions"][k] for k in keys], [scg_t[k]["gain_ccpa_minus_cosf"] for k in keys], strata)
    L = ["# Foundation-model round on TEST (single pass of bundle part yaml-fsor-foundation-round)", "",
         f"Bundle token {token}; done {t['when']} ({t['gpu_minutes']:.1f} GPU-minutes), code at {t['git_head'][:7]}. NaN found: {t['nan_found']}. "
         "Decisions after this test: none. Pretraining overlap: docs/preregistration/foundation_round.md, Section 3 (absolute numbers are not "
         "comparable with ResNet-12 FSOR methods; the claim is the paired CCP-A vs COSF comparison on identical features).", "",
         "## H1: CCP-A − COSF AUROC > 0 (32 cells; Holm over one-sided Wilcoxon p)", "",
         f"Holm-significant cells (adjusted p < 0.05, positive mean): {h1['n_holm_significant']} of {h1['n_cells']}; Bonferroni-simultaneous 95% CI "
         f"excludes zero: {h1['n_bonferroni_ci_excludes_zero']} of {h1['n_cells']}.", "",
         "| Cell (dataset, backbone, shot) | gain ± 95% CI | Bonferroni CI | % improved | p (one-sided) | p (Holm) | headroom-normalised (A5) |",
         "|---|---|---|---|---|---|---|"]
    hr = {fm_cell(ds, bb, sh): stats.headroom_paired(z[f"{host(ds, bb)}__T__{sh}shot__CCP-A__auroc"], z[f"{host(ds, bb)}__T__{sh}shot__COSF__auroc"])
          for bb in FM_BACKBONES for ds in FM_DATASETS for sh in (1, 5)}
    for k, x in h1["cells"].items():
        L.append(f"| {k} | {x['mean']:+.2f} ± {x['ci95']:.2f} | [{x['bonferroni_lo']:+.2f}, {x['bonferroni_hi']:+.2f}] | {x['frac_gain'] * 100:.0f}% | "
                 f"{x['p_one_sided']:.1e} | {x['p_holm']:.1e} | {hr[k]['mean']:+.3f} ± {hr[k]['ci95']:.3f} |")
    L += ["", f"## H2: test D1 vs test gain: Spearman rho {h2['rho']:+.3f}, one-sided permutation p {h2['p_one_sided']:.4f} (n = {h2['n']})",
          f"## H2b: valA D1 (frozen in the bundle) vs test gain: Spearman rho {h2b['rho']:+.3f}, one-sided permutation p {h2b['p_one_sided']:.4f}",
          f"## H2-s (within dataset x shot, 8 strata of 4 backbones; A1): mean within-stratum rho {h2s['stat_mean_within_rho']:+.3f}, one-sided p "
          f"{h2s['p_one_sided']:.4f}",
          f"## H2b-s: valA D1 vs test gain within strata: mean rho {h2bs['stat_mean_within_rho']:+.3f}, one-sided p {h2bs['p_one_sided']:.4f}", "",
          "## H3 (exploratory, descriptive): DINOv2 S -> B -> L", "",
          "| dataset | shot | COSF AUROC S / B / L | raw gain S / B / L | headroom-normalised, cell level S / B / L | headroom-normalised, paired S / B / L |",
          "|---|---|---|---|---|---|"]
    for ds in FM_DATASETS:
        for sh in (1, 5):
            cs = [t["summaries"][f"{host(ds, bb)}|T|{sh}|COSF"]["auroc"] for bb in ("dinov2_s14", "dinov2_b14", "dinov2_l14")]
            gs = [scg_t[fm_cell(ds, bb, sh)]["gain_ccpa_minus_cosf"] for bb in ("dinov2_s14", "dinov2_b14", "dinov2_l14")]
            hc = [stats.headroom_cell(c_ + g, c_) for c_, g in zip(cs, gs)]
            hp = [hr[fm_cell(ds, bb, sh)]["mean"] for bb in ("dinov2_s14", "dinov2_b14", "dinov2_l14")]
            L.append(f"| {ds} | {sh} | {' / '.join(f'{c_:.2f}' for c_ in cs)} | {' / '.join(f'{g:+.2f}' for g in gs)} | "
                     f"{' / '.join(f'{x:+.3f}' for x in hc)} | {' / '.join(f'{x:+.3f}' for x in hp)} |")
    outs.append(write_text("foundation_round", "test.md", L))
    p = out_path("foundation_round", "d1_test.png")
    fm_scg_figure({k: (scg_t[k]["std_gap"], scg_t[k]["gain_ccpa_minus_cosf"]) for k in keys}, p, "Test")
    outs.append(p)
    p = out_path("foundation_round", "test_hypotheses.json")
    p.write_text(json.dumps({"H1": h1, "H2": h2, "H2b": h2b, "H2-s": h2s, "H2b-s": h2bs, "headroom_A5_descriptive": hr}, indent=1),
                 encoding="utf-8", newline="\n")
    outs.append(p)
    return outs


# ==================================================================================================================== compare
RECORDED = {  # generated file -> recorded file
    "baselines": {"validation.md": "results/baselines/validation.md", "test.md": "results/baselines/test.md"},
    "variants": {"validation_valB.md": "results/variants/validation_valB.md", "validation_valC.md": "results/variants/validation_valC.md",
                 "test.md": "results/variants/test.md"},
    "final_round": {"cifarfs_valB.md": "results/final_round/cifarfs_valB.md", "cub_val_preview.md": "results/final_round/cub_val_preview.md",
                    "recorded_valA.md": "results/final_round/recorded_valA.md", "d1_val.png": "results/final_round/d1_val.png",
                    "test.md": "results/final_round/test.md", "d1_test.png": "results/final_round/d1_test.png"},
    "foundation_round": {"d1_val.png": "results/foundation_round/d1_val.png", "test.md": "results/foundation_round/test.md",
                         "d1_test.png": "results/foundation_round/d1_test.png", "test_hypotheses.json": "results/foundation_round/test_hypotheses.json"},
}
RENDER = {"baselines": render_baselines, "variants": render_variants, "final_round": render_final_round, "foundation_round": render_foundation_round}
ROUNDS = ["fsor", "baselines", "variants", "final_round", "foundation_round"]


def check(round_name: str, verbose: bool = False) -> list[str]:
    """Re-render one round and compare; returns the list of mismatches (empty = every file / block identical)."""
    bad = []
    if round_name == "fsor":
        for name, rendered, rec in render_fsor():
            same = rendered == rec
            if verbose:
                print(f"{'OK  ' if same else 'DIFF'} fsor {name}: {len(rendered)} lines")
            if not same:
                bad.append(f"fsor {name}")
        return bad
    RENDER[round_name]()
    for gen, rec in RECORDED[round_name].items():
        a = (GENERATED / round_name / "recorded" / gen).read_bytes()
        b = read_bytes(rec)
        same = a == b
        if verbose:
            print(f"{'OK  ' if same else 'DIFF'} {round_name} {rec} ({len(b)} bytes)")
        if not same:
            bad.append(rec)
    return bad


def main():
    names = sys.argv[1:] or ["all"]
    names = ROUNDS if names == ["all"] else names
    bad = []
    for n in names:
        bad += check(n, verbose=True)
    print("re-render:", "all identical" if not bad else f"DIFFERENT: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
