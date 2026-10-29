# Third-party code, models, fonts and data

Licences were read from each repository's `LICENSE` file at the pinned commit. Only two kinds of third-party material are included in this
repository: the Liberation Sans fonts (`third_party/fonts/`) and our patch to DNPG (`third_party/patches/dnpg.patch`), which contains lines of
DNPG's code as diff context. Everything else is fetched at a pinned commit by `python third_party/clone.py` or downloaded from its source,
and is never copied here.

## Included in this repository

### Liberation Sans 2.1.5 (`third_party/fonts/liberation-sans-2.1.5/`)

| Item | Source | Licence |
|---|---|---|
| `LiberationSans-{Regular,Bold,Italic,BoldItalic}.ttf`, unmodified | github.com/liberationfonts/liberation-fonts, release 2.1.5 (`liberation-fonts-ttf-2.1.5.tar.gz`, sha256 `7191c669bf38899f73a2094ed00f7b800553364f90e2637010a69c0e268f25d0`) | SIL Open Font License 1.1: Copyright (c) 2010 Google Corporation (Reserved Font Arimo, Tinos and Cousine); Copyright (c) 2012 Red Hat, Inc. (Reserved Font Name Liberation) |

The licence text is included next to the fonts as `third_party/fonts/liberation-sans-2.1.5/LICENSE`, and the font authors as `AUTHORS`.
The OFL allows bundling the fonts with software; they are not sold on their own and are not renamed or modified. They are used to draw the
figures and are embedded as subsets in the figure PDFs. sha256: Regular `76d04c18ea243f426b7de1f3ad208e927008f961dc5945e5aad352d0dfde8ee8`,
Bold `788abee4c806d660e8aee46689dd8540cd4bb98da03dcc9d171ce3efd99a9173`, Italic `e5bae5c4cde31f22142753855f4f8fb86da6ff39955ed3c0a11248b0d16948b0`,
BoldItalic `698da70fc191cc5f33ad4d6d3fe830fe4624b898ea2e3169955928b7c491f1ee`, LICENSE `93fed46019c38bbe566b479d22148e2e8a1e85ada614accb0211c37b2c61c19b`,
AUTHORS `b627404917ef824675cb24c32e2facd0b57650627955f3fc3d8a8863c5d0adfc`.

### DNPG patch (`third_party/patches/dnpg.patch`)

Our changes to DNPG (github.com/gary23ai/DNPG at `38b8fd13656ae66c87488415223a8df24ad67c80`; 5 files, +36 / −12): output and checkpoint
paths taken from the command line instead of hard-coded paths, the number of base classes taken from the data instead of a hard-coded 60,
and optional gradient accumulation over episodes for the tieredImageNet 5-shot run (environment variable `DNPG_ACCUM_EPISODES=1`). The patch is applied
by `python third_party/clone.py DNPG`. Its context lines are DNPG's code, distributed under DNPG's licence:

```text
MIT License

Copyright (c) 2024 iCGY96

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The patch's own changes are under this repository's licence (LICENSE).

### Class names (`configs/datasets/mini_class_names.tsv`)

The public ImageNet-1k synset names of the 100 mini-ImageNet classes (text only), used for the labels of Figure 4.

## Fetched or downloaded, not included

| Project | Used for | Repository | Pinned commit | Licence |
|---|---|---|---|---|
| FEAT (Ye et al., CVPR 2020) | frozen FEAT backbone; FEAT meta-trained models (SnaTCHer-F); split CSVs | github.com/Sha-Lab/FEAT | `47bdc7c1672e00b027c67469d0291e7502918950` | MIT |
| SnaTCHer (Jeong et al., CVPR 2021) | SnaTCHer-F score and its attention module; fidelity check | github.com/MinkiJ/SnaTCHer | `335c42469f0a7ad72c5c3480c8effc8c293823e0` | MIT |
| FRN (Wertheimer et al., CVPR 2021) | FRN head and fine-tuning recipe | github.com/Tsingularity/FRN | not cloned: the head is re-implemented in `scripts/train/frn_head.py`, attributed there | MIT |
| TANE (Huang et al., CVPR 2022) | TANE pre-trained backbones (DNPG initialization); word vectors for DNPG training | github.com/shiyuanh/TANE | `fe089d477ba9387dd2d008d8872cdc84f8ad77ed` | MIT |
| DNPG (Zhang et al., ACM MM 2024) | DNPG host: model, training and evaluation code | github.com/gary23ai/DNPG (formerly github.com/iCGY96/DNPG, which redirects; docs/checkpoints.md) | `38b8fd13656ae66c87488415223a8df24ad67c80` | MIT |
| RFS (Tian et al., ECCV 2020) | ResNet-12 model and recipe of the CIFAR-FS host | github.com/WangYueFt/rfs | `f8c837ba93c62dd0ac68a2f4019c619aa86b8421` | **no licence file**: fetched and imported at run time only; neither its code nor our checkpoint trained with it is redistributed |
| CLIP (Radford et al., 2021) | CLIP ViT-B/16 image encoder | github.com/openai/CLIP | `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` | MIT |
| DINOv2 (Oquab et al., 2024) | DINOv2 ViT-S/14, B/14, L/14 | github.com/facebookresearch/dinov2 | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | Apache-2.0 (code and backbone weights) |
| learn2learn | tieredImageNet image pickles (Zenodo record 7978538); CIFAR-FS class lists | github.com/learnables/learn2learn | data only | MIT |

The full licence texts are in each clone (`third_party/<name>/LICENSE`) after `python third_party/clone.py <name>`.

## Weights and data

| Item | Source | Terms | Redistributed here? |
|---|---|---|---|
| FEAT and TANE checkpoints | their READMEs (Google Drive) | no separate licence for the weights; the repositories are MIT | no: linked (docs/checkpoints.md) |
| CLIP ViT-B/16 weights | OpenAI (the CLIP repository's download URL) | released with the MIT-licensed repository | no: downloaded, sha256-pinned |
| DINOv2 backbone weights | dl.fbaipublicfiles.com/dinov2 | Apache-2.0 | no: downloaded, sha256-pinned |
| Our FRN and DNPG checkpoints | this work | this repository's licence; derived from the MIT-licensed code and weights above | release assets (Zenodo) |
| Our RFS CIFAR-FS checkpoint | this work | trained with RFS's code, which has no licence | no (available from the authors on request) |
| Per-episode test metrics, validation score caches | this work: metrics and scores computed on the datasets | numbers only, no images; the datasets' terms apply to their use | release assets |
| mini-ImageNet, tieredImageNet (ImageNet), CIFAR-100, CUB-200-2011 images | their sources (docs/data.md) | each dataset's terms (ImageNet: non-commercial research and education) | **no** |

Figures 1 and 4 of the paper contain mini-ImageNet images and are therefore not part of this repository (README).

Python packages (requirements.txt) are installed from PyPI and keep their own licences; `ftfy` (Apache-2.0), `regex` (Apache-2.0 and
CNRI-Python) and `wcwidth` (MIT) are needed only to import the official `clip` package for the loader check.
