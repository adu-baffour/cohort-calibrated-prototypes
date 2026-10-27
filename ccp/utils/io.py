"""Deterministic .npz writer: fixed member order and timestamps, so identical arrays give an identical file (and sha256).
np.savez stamps the current time into the zip, which would make rebuilt banks and caches differ byte-wise."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np


def save_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            z.writestr(info, buf.getvalue())
    tmp.replace(path)
