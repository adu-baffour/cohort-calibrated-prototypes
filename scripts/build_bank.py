"""Build a host's cohort bank (Eq. 3 / Eq. 6 statistics) from the prepared base images with the host's own encoder.

    python scripts/build_bank.py --host frozen_feat_mini
    python scripts/build_bank.py --host dnpg_mini_seed0 --shot 1          # per-shot hosts (SnaTCHer-F, DNPG)
    python scripts/build_bank.py --all [--record]                          # every host / shot; --record writes sha256 into configs/hosts

Writes $CCP_ASSETS/<bank.file> (.npz). If the host config pins a sha256, the rebuilt file is compared with it (a mismatch is reported,
never overwritten into the config unless --record is given).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccp.hosts import build_host, load_host_config  # noqa: E402
from ccp.pipeline import bank_path, make_bank  # noqa: E402
from ccp.utils.logging import get_logger  # noqa: E402
from ccp.utils.paths import REPO  # noqa: E402

log = get_logger("build_bank")
PER_SHOT = ("snatcher_f", "dnpg")


def hosts_and_shots(names):
    for n in names:
        cfg = load_host_config(n)
        for sh in ((1, 5) if cfg["kind"] in PER_SHOT else (None,)):
            yield n, sh


def record_sha(host_name: str, shot, sha: str):
    p = REPO / "configs" / "hosts" / f"{host_name}.yaml"
    txt = p.read_text(encoding="utf-8")
    if shot is None:
        new = re.sub(r"(bank:\n  file: [^\n]+\n  sha256: )[^\n]+", rf"\g<1>{sha}", txt, count=1)
    else:
        new = re.sub(rf"(  {shot}shot:\n    file: [^\n]+\n    sha256: )[^\n]+", rf"\g<1>{sha}", txt, count=1)
    if new == txt and sha not in txt:
        raise SystemExit(f"could not record sha256 in {p}")
    p.write_text(new, encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shot", type=int, choices=(1, 5))
    ap.add_argument("--record", action="store_true", help="write the sha256 into configs/hosts/<host>.yaml")
    ap.add_argument("--force", action="store_true", help="rebuild even if the bank file exists")
    a = ap.parse_args()
    names = sorted(p.stem for p in (REPO / "configs" / "hosts").glob("*.yaml")) if a.all else a.host
    for name, sh in hosts_and_shots(names):
        if a.shot and sh not in (None, a.shot):
            continue
        cfg = load_host_config(name)
        entry = cfg["bank"][f"{sh}shot"] if sh else cfg["bank"]
        from ccp.utils.paths import assets_root
        out = assets_root() / entry["file"]
        if out.exists() and not a.force:
            log.info(f"{name} shot={sh}: {out} exists (use --force to rebuild)")
            continue
        host = build_host(cfg, shot=sh)
        b = make_bank(host, log.info)
        sha = b.save(bank_path(host))
        pinned = entry.get("sha256")
        if pinned and pinned != sha:
            log.warning(f"{name} shot={sh}: rebuilt sha256 {sha} != pinned {pinned}")
        log.info(f"{name} shot={sh}: {out} sha256 {sha}")
        if a.record:
            record_sha(name, sh, sha)
        del host
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
