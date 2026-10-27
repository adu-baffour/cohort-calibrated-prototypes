# CLIP loader fidelity (F1; base images only)

(a) clip_frozen host path vs (b) the official `clip.load(<pinned local ViT-B-16.pt>, device='cuda', jit=False)` from the pinned clone, then `.float()` and `encode_image`; both through the registered input rule and gap() convention, on the first 500 mini base images (eval84) (code at 4bc271b).

| View | bitwise identical | max abs difference | min cosine |
|---|---|---|---|
| plain | True | 0.000e+00 | 0.999999762 |
| flip | True | 0.000e+00 | 0.999999821 |

Packages: ftfy 6.3.1, regex 2026.9.29, CLIP @ d05afc4.
