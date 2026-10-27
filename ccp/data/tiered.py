"""tieredImageNet (Ren et al. 2018; 351 / 97 / 160 classes) from the learn2learn copy (Zenodo record 7978538: PNG pickles).

The pickles hold 84 x 84 PNG bytes; they are decoded with PIL (RGB) and written as BGR uint8 arrays, because every released tiered
checkpoint used here (FEAT Res12-pre, FEAT meta-trained, TANE pre-trained) was trained on BGR tensors: on 200 base-class episodes the
FEAT tiered classifier scores 86.27 (RGB) vs 93.64 (BGR) at 1-shot (docs/protocol.md, "Channel order"). Normalisation: FEAT's
ResNet-12 constants, applied to the BGR tensor as in the original pipelines.
"""
from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
MD5 = {"train_images_png": "e1a48fcfa1c8b3a0b402a19ba892adfb", "val_images_png": "6162949244106b348e24264057f5b66e",
       "test_images_png": "4d9b27d03460a3b2c525c249655ae501"}
ZENODO = "https://zenodo.org/records/7978538"


def labels(raw: Path, split: str) -> np.ndarray:
    with open(Path(raw) / f"tiered-imagenet-{split}_labels.pkl", "rb") as f:
        return np.asarray(pickle.load(f, encoding="latin1")["label_specific"])


def images_pickle(raw: Path, split: str) -> list:
    with open(Path(raw) / f"tiered-imagenet-{split}_images_png.pkl", "rb") as f:
        return pickle.load(f, encoding="latin1")


def decode_rgb(png_bytes) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(bytes(png_bytes))).convert("RGB"), dtype=np.uint8)


def decode_bgr(png_bytes) -> np.ndarray:
    return np.ascontiguousarray(decode_rgb(png_bytes)[..., ::-1])
