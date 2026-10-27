"""Pack the release assets under this release's names, check them against the 1.5.0 archives, and write the manifests (run by the
authors before uploading; nothing is uploaded here).

    python scripts/release_manifest.py pack --old <1.5.0 upload folder>                  # -> release_assets/upload/ (git-ignored)
    python scripts/release_manifest.py check                                            # the packed archives match MANIFEST.txt

Archives (reproducible: sorted members, fixed mtime / owner / mode, USTAR):
  cohort-calibrated-prototypes-2.0.0-checkpoints-<host>.tar   our checkpoints, one archive per host (Zenodo, with the DOI)
  cohort-calibrated-prototypes-2.0.0-recorded.tar             per-episode test metrics and validation score caches (Zenodo and the GitHub release)
Member paths and bytes are those of the 1.5.0 archives of the authors' earlier release repository (private) (file names as published there): every
member is checked byte for byte (sha256) against its 1.5.0 archive before packing. The cohort banks are not uploaded: `python
scripts/build_bank.py` rebuilds them byte-identically (sha256 pinned in configs/hosts/*.yaml). Unpack every archive into one folder
(= CCP_ASSETS; its checkpoints/ folder goes into CCP_CKPT).
Writes release_assets/upload/{MANIFEST.txt, SHA256SUMS} and the "Release assets" section of results/PROVENANCE.md (old and new archive
checksums, per-member sha256).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ccp.utils.paths import assets_root  # noqa: E402

NAME = "cohort-calibrated-prototypes"
VERSION = "2.0.0"
OUT = REPO / "release_assets" / "upload"
PROVENANCE = REPO / "results" / "PROVENANCE.md"
SECTION = "\n## Release assets"
MAX_BYTES = int(1.9 * 2**30)
CKPT_HOSTS = ["frn_mini_seed0", "frn_mini_seed1", "frn_mini_seed2", "dnpg_mini_seed0", "dnpg_mini_seed1", "dnpg_mini_seed2", "dnpg_tiered_seed0"]
OLD = {**{f"checkpoints-{h}": f"zenodo/ccp-fsor-v1.0.0-checkpoints-{h}.tar" for h in CKPT_HOSTS}, "recorded": "github/ccp-fsor-v1.4.0-recorded.tar"}


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def members_of(tar: Path) -> dict[str, tuple[int, str]]:
    """{member path: (bytes, sha256)} of a tar archive."""
    out = {}
    with tarfile.open(tar) as t:
        for m in t:
            if m.isfile():
                f = t.extractfile(m)
                h = hashlib.sha256()
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
                out[m.name] = (m.size, h.hexdigest())
    return out


def new_name(key: str) -> str:
    return f"{NAME}-{VERSION}-{key}.tar"


def pack(old_dir: Path) -> list[dict]:
    root = assets_root()
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    for key, old_rel in OLD.items():
        old = old_dir / old_rel
        ref = members_of(old)
        files = sorted(ref)
        for m in files:                                   # the members on disk equal the 1.5.0 archive's members
            p = root / m
            if not p.exists() or (p.stat().st_size, sha_file(p)) != ref[m]:
                raise SystemExit(f"{m}: differs from {old_rel} (or missing under CCP_ASSETS)")
        dest = OUT / new_name(key)

        def norm(ti):
            ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname, ti.mode = 0, 0, 0, "", "", 0o644
            return ti
        with tarfile.open(dest, "w", format=tarfile.USTAR_FORMAT) as t:
            for m in files:
                t.add(root / m, arcname=m, recursive=False, filter=norm)
        if dest.stat().st_size > MAX_BYTES:
            raise SystemExit(f"{dest.name} is over 1.9 GiB")
        got = members_of(dest)
        if got != ref:
            raise SystemExit(f"{dest.name}: members differ from {old_rel}")
        out.append({"key": key, "old": old_rel, "old_sha": sha_file(old), "old_bytes": old.stat().st_size, "new": dest.name,
                    "new_sha": sha_file(dest), "new_bytes": dest.stat().st_size, "members": ref})
        print(f"{dest.name}: {len(ref)} members, identical to {old_rel}")
    return out


def manifest(arch: list[dict]) -> str:
    L = [f"# {NAME} {VERSION} release assets. Unpack every archive into one folder (= CCP_ASSETS; its checkpoints/ folder goes into CCP_CKPT).",
         "# Columns: path, bytes, sha256.", "", "## Archives", ""]
    L += [f"{a['new']}\t{a['new_bytes']}\t{a['new_sha']}" for a in arch]
    L += ["", "## Archive contents"]
    for a in arch:
        L += ["", f"# {a['new']}"] + [f"{m}\t{s}\t{h}" for m, (s, h) in sorted(a["members"].items())]
    L += ["", "## Cohort banks: not uploaded. `python scripts/build_bank.py` rebuilds them byte-identically; sha256 pinned in "
          "configs/hosts/*.yaml (docs/checkpoints.md).", ""]
    return "\n".join(L)


def provenance_section(arch: list[dict]) -> str:
    L = [SECTION.strip("\n"), "",
         f"Written by `python scripts/release_manifest.py pack`. The release assets of {VERSION} are the 1.5.0 archives repacked under this "
         "release's names: every member has the same path and the same bytes (sha256 below, checked against the 1.5.0 archive when packing). "
         + ("The packing is deterministic, so every archive is also byte-identical to its 1.5.0 archive; only the file names changed. "
            if all(a["new_sha"] == a["old_sha"] for a in arch) else "The archive bytes differ from 1.5.0 only in the tar headers. ") +
         "Not uploaded from this repository: the cohort banks (rebuilt byte-identically) and "
         "the RFS CIFAR-FS checkpoint (docs/checkpoints.md).", "",
         "| Archive (2.0.0) | bytes | sha256 | 1.5.0 archive | bytes | sha256 |", "|---|---|---|---|---|---|"]
    L += [f"| {a['new']} | {a['new_bytes']} | {a['new_sha']} | {Path(a['old']).name} | {a['old_bytes']} | {a['old_sha']} |" for a in arch]
    L += ["", "| Member | Archive (2.0.0) | bytes | sha256 |", "|---|---|---|---|"]
    for a in arch:
        L += [f"| {m} | {a['new']} | {s} | {h} |" for m, (s, h) in sorted(a["members"].items())]
    return "\n".join(L) + "\n"


def write_provenance(sec: str):
    txt = PROVENANCE.read_text(encoding="utf-8")
    txt = txt[:txt.index(SECTION)] if SECTION in txt else txt
    PROVENANCE.write_text(txt.rstrip("\n") + "\n" + "\n" + sec, encoding="utf-8", newline="\n")


def check() -> int:
    lines = (OUT / "MANIFEST.txt").read_text(encoding="utf-8").split("## Archive contents")[0].splitlines()
    bad = 0
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) == 3 and parts[0].endswith(".tar"):
            p = OUT / parts[0]
            ok = p.exists() and p.stat().st_size == int(parts[1]) and sha_file(p) == parts[2]
            bad += not ok
            print(f"{parts[0]}: {'OK' if ok else 'MISMATCH or missing'}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("pack", "check"))
    ap.add_argument("--old", type=Path, help="pack: the 1.5.0 upload folder (zenodo/, github/)")
    a = ap.parse_args()
    if a.cmd == "check":
        sys.exit(1 if check() else 0)
    if a.old is None:
        raise SystemExit("pack needs --old")
    arch = pack(a.old.resolve())
    (OUT / "MANIFEST.txt").write_text(manifest(arch), encoding="utf-8", newline="\n")
    (OUT / "SHA256SUMS").write_text("".join(f"{x['new_sha']}  {x['new']}\n" for x in arch) +
                                    f"{sha_bytes((OUT / 'MANIFEST.txt').read_bytes())}  MANIFEST.txt\n", encoding="utf-8", newline="\n")
    write_provenance(provenance_section(arch))
    print(f"wrote {OUT.relative_to(REPO).as_posix()}/MANIFEST.txt, SHA256SUMS and the release-asset section of results/PROVENANCE.md")


if __name__ == "__main__":
    main()
