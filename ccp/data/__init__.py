"""Dataset readers. Images are uint8 (n, 84, 84, 3) arrays prepared once by scripts/prepare_data/ (docs/data.md).

Channel order is explicit: every prepared array has a sidecar JSON with "channel_order" ("RGB" or "BGR"), and every host config
declares the order its checkpoint expects. ImageSet.as_order() converts (a pure channel flip, exact) and refuses unknown orders.
  mini-ImageNet   RGB (FEAT's mini checkpoints, TANE/DNPG mini backbone)
  tieredImageNet  BGR (the released tiered checkpoints were trained on cv2-decoded BGR arrays; research note TIERED_GAP)
"""
from ccp.data.images import ImageSet, open_split  # noqa: F401
