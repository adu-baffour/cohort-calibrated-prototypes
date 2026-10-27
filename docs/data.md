# Data

No dataset image is redistributed (README, "Figures with dataset images"). You obtain the four datasets; one script per dataset writes
fixed uint8 arrays under `$CCP_DATA` and checks counts and hashes. Every recorded number was computed from arrays with exactly the sha256
values below (pinned in `configs/datasets/*.yaml`). Episode manifests (indices into these arrays, no images) are in `manifests/`; test
manifests open only with `--allow-test` (REPRODUCE.md, tier 2).

## mini-ImageNet

- **Split**: Ravi and Larochelle, 64 / 16 / 20 classes x 600 images, exactly FEAT's split CSVs
  (github.com/Sha-Lab/FEAT/tree/47bdc7c/data/miniimagenet/split; downloaded automatically, or pass `--csv-dir`).
- **Images**: the original-resolution JPEGs in `train/`, `val/`, `test/` folders named by WordNet id, for example the `miniImageNet.zip`
  shared by FEAT and FRN (figshare 26340226, 9.71 GB, md5 `f93ae1b058db5d6e906150e55804f8f1`). Unpack to `$CCP_DATA/mini/raw/miniImageNet/`
  or pass `--image-root`.
- **Prepare** (CPU, about a minute with 12 workers; 2.4 GB): `python scripts/prepare_data/mini.py [--image-root DIR]`. The script refuses a
  folder layout that differs from the CSVs, wrong counts and all-zero images.
- **Views**: `eval84` = PIL `Resize(92)` on the short side (bilinear) + `CenterCrop(84)` (frozen FEAT, FRN, SnaTCHer-F); `whole84` = whole
  image, PIL LANCZOS to 84 x 84 (TANE pre-trained, DNPG).
- **Global index** (the index space of the manifests): train 0-38,399, val 38,400-47,999, test 48,000-59,999, in split order, then sorted
  WordNet id, then sorted file name. Labels 0-99 follow the same order (`mini/prepared/index.json`).

## tieredImageNet

- **Split**: Ren et al., 351 / 97 / 160 classes (448,695 / 124,261 / 206,209 images).
- **Images**: the learn2learn copy, Zenodo record 7978538: `tiered-imagenet-{train,val,test}_images_png.pkl` and `..._labels.pkl` in
  `$CCP_DATA/tiered/raw/` (or pass `--raw`). md5 of the image pickles: train `e1a48fcfa1c8b3a0b402a19ba892adfb`, val
  `6162949244106b348e24264057f5b66e`, test `4d9b27d03460a3b2c525c249655ae501` (checked by the script).
- **Prepare** (CPU, about 3 minutes with 16 workers; 16 GB): `python scripts/prepare_data/tiered.py [--raw DIR]`. Decodes the 84 x 84 PNGs
  with PIL and writes **BGR** arrays: both released tieredImageNet checkpoints expect BGR input (docs/protocol.md, "Channel order").
- **Global index**: train 0-448,694, val 448,695-572,955, test 572,956-779,164, label-pickle order within each split.
- **Bank subset** (CCP-A): `np.sort(np.random.RandomState(20260926).choice(448695, 38400, replace=False))`; CCP-S uses all 448,695.
- **Validation halves**: valA and valB use disjoint halves of the 97 validation classes (`RandomState(20260925)`); their manifest files
  are named `tvA_*` and `tvB_*`.

## CIFAR-FS

- **Split**: Bertinetto et al., 64 / 16 / 20 CIFAR-100 classes, 600 images each (500 CIFAR-100 train + 100 CIFAR-100 test images per class),
  class lists in `configs/datasets/cifarfs_split/` (the r2d2 lists as mirrored by learn2learn).
- **Images**: the CIFAR-100 python archive (`https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz`, md5
  `eb9058c3a382ffc7106e4002c42a8d85`) at `$CCP_DATA/raw/cifar-100-python.tar.gz`.
- **Prepare**: `python scripts/prepare_data/cifarfs.py`. View `eval32`: the original 32 x 32 RGB pixels.
- **Global index**: train 0-38,399, val 38,400-47,999, test 48,000-59,999 (split order, class-list order, CIFAR-100 train then test images
  in archive order).

## CUB-200-2011

- **Split**: FRN's rule on the 1-based class id i: i % 2 == 0 train (100 classes, 5,903 images), i % 4 == 1 val (50, 2,927), i % 4 == 3
  test (50, 2,958). Whole images; the bounding boxes are not used.
- **Images**: `CUB_200_2011.tgz` from data.caltech.edu (md5 `97eceeb196236b17998738112f37df78`) at `$CCP_DATA/raw/CUB_200_2011.tgz`.
- **Prepare**: `python scripts/prepare_data/cub.py` (views `eval84` and `whole84` as for mini-ImageNet, for the transfer from
  mini-ImageNet); `--views eval224` adds the 224 x 224 view of the input-resolution analysis (PIL `Resize(256)` bicubic on the short side +
  `CenterCrop(224)`). The foundation-model hosts upsample the stored 84 x 84 view to 224 x 224 (paper, Section 4.2).

## Expected prepared arrays

| Dataset | File (under `$CCP_DATA`) | Shape | Channels | sha256 |
|---|---|---|---|---|
| mini | `mini/prepared/eval84_train.npy` | (38,400, 84, 84, 3) uint8 | RGB | `a5896a9761d38ed1477f4a7cf6e499321f42075a07bd2f3dfd7239ad18e99f62` |
| mini | `mini/prepared/eval84_val.npy` | (9,600, 84, 84, 3) uint8 | RGB | `30b5528acecb6b00be15ecb042ee249f21bc2c52de0995d0cc625654e33410a8` |
| mini | `mini/prepared/eval84_test.npy` | (12,000, 84, 84, 3) uint8 | RGB | `b78c04affa10e404879ac139d8c57ddfb7a72e8b25de9aa2b07b9eaceddb4472` |
| mini | `mini/prepared/whole84_train.npy` | (38,400, 84, 84, 3) uint8 | RGB | `68909ca39c2de0d6ef69eba14835daa441d57315aec94feed49626cf6356f86b` |
| mini | `mini/prepared/whole84_val.npy` | (9,600, 84, 84, 3) uint8 | RGB | `dd3523d65dd96bac2f14df3e5011cc98bd41f7a3a5d5ff748c117a786e80926c` |
| mini | `mini/prepared/whole84_test.npy` | (12,000, 84, 84, 3) uint8 | RGB | `0d1effc3353caaf24bbb5ed8aea59cdaff5a003cdb8e9810d7c6612fe3e86e79` |
| mini | `mini/prepared/index.json` | wnids, labels, sizes | - | `cab74c2a1cf5c978ed5f84f07005edbc65f1784c266743beac5167ece0f0206b` |
| tiered | `tiered/prepared/train_bgr84.npy` | (448,695, 84, 84, 3) uint8 | BGR | `cdbfa0fc7845380852f51572b99de905dbb8d3b53ea6bb1b54c40c390caf9869` |
| tiered | `tiered/prepared/train_labels.npy` | (448,695,) int64 | - | `a918b1650bad4890095d78e72f95892cb78697cb8331645400baa2123436da81` |
| tiered | `tiered/prepared/val_bgr84.npy` | (124,261, 84, 84, 3) uint8 | BGR | `2b0d8522e4b87e3031ef0c4c6011cd7c4cfd3d61ae6f3676b5fe2d21ad9bb418` |
| tiered | `tiered/prepared/val_labels.npy` | (124,261,) int64 | - | `66f590814e4e2f7a37cf62f88b014d19e4e89125538f7fb24dc2281258f481bc` |
| tiered | `tiered/prepared/test_bgr84.npy` | (206,209, 84, 84, 3) uint8 | BGR | `9c4a134a3e5f62230077da9e4c63da9c646f852a6c402f0a371e8f25f97cbfe9` |
| tiered | `tiered/prepared/test_labels.npy` | (206,209,) int64 | - | `55ffb1308f9aa6066b99c84430be77099a927d95199379fb7c7c127d1c22551e` |
| CIFAR-FS | `cifarfs/prepared/eval32_train.npy` | (38,400, 32, 32, 3) uint8 | RGB | `9147af46a9b1bcbc0f4f9da5109c8f63b396f4a4b2742ea5235433af2761a339` |
| CIFAR-FS | `cifarfs/prepared/eval32_val.npy` | (9,600, 32, 32, 3) uint8 | RGB | `19603a98bf371918c9b9a78920b2bafe856056a9c283df203c37752c1761a293` |
| CIFAR-FS | `cifarfs/prepared/eval32_test.npy` | (12,000, 32, 32, 3) uint8 | RGB | `7515ad3f648865cc4901d7f5689f60f4f302c51463229feadc7688234cc0ff7c` |
| CIFAR-FS | `cifarfs/prepared/index.json` | classes, sizes | - | `309cb9031629108bdfd8f61b162956302b775d482abfde3afeb0f98416f982b1` |
| CUB | `cub/prepared/eval84_train.npy` | (5,903, 84, 84, 3) uint8 | RGB | `aafac6e39bd105a25d699c5ba1d5d750f152c3c0bf7d9d5303c37cdbd85b3cdd` |
| CUB | `cub/prepared/eval84_val.npy` | (2,927, 84, 84, 3) uint8 | RGB | `d772e6a21a2c438b246d8ec68e88fa4f5e3031fdf01cfe1ae49855b0d584594c` |
| CUB | `cub/prepared/eval84_test.npy` | (2,958, 84, 84, 3) uint8 | RGB | `a9105e2d4c7bfca4612ffa4e490de4cca349607607200409afa5ea00067df267` |
| CUB | `cub/prepared/whole84_train.npy` | (5,903, 84, 84, 3) uint8 | RGB | `69f24dd1817ef839a49f7073bc085d4e02b05dd32a8381316ad17f73f53182ad` |
| CUB | `cub/prepared/whole84_val.npy` | (2,927, 84, 84, 3) uint8 | RGB | `9ec17dc05abeea6378c0a141992d76254542e7b2674c981f097c7961e2dacf5b` |
| CUB | `cub/prepared/whole84_test.npy` | (2,958, 84, 84, 3) uint8 | RGB | `78e79fee8e29ec38a7c5f100ed2ae4e1fdcf54a2d6220f148392604e11e0ffc4` |
| CUB | `cub/prepared/eval224_train.npy` | (5,903, 224, 224, 3) uint8 | RGB | `bc144848c70b5284f7e9acfcd75564650980834afdf1793df9cd0f2719c78d81` |
| CUB | `cub/prepared/eval224_val.npy` | (2,927, 224, 224, 3) uint8 | RGB | `ce77089e267b01295a7707d26e51f98fd583590d0af4266a0f1699abfbb9a55e` |
| CUB | `cub/prepared/eval224_test.npy` | (2,958, 224, 224, 3) uint8 | RGB | `7338b7a79f7f74a2d23a84a1b40eff77fd121af1eb4b0ca4aa6f82f921baa080` |
| CUB | `cub/prepared/index.json` | classes, sizes | - | `a4568cba63ce7d63905c1d4e1ef73918a75658f8b3c84f0964d54251fa53884a` |
