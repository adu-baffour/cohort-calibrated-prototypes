# Cost table (measured on this machine's NVIDIA GeForce RTX 4090)

Script scripts/fsor_cost.py; raw numbers research/status/fsor_cost.json. Per-episode times: mean over 200 mini valA_fsor episodes per shot after 20 warm-up episodes, torch.cuda.synchronize around each timed block, 5-way with 5 known + 5 unknown classes and 15 queries per class (150 queries). "Extra memory" = peak GPU memory allocated inside the timed block minus memory allocated before it. Backbone rows are frozen FEAT ResNet-12 (eval view); DNPG rows are DNPG seed 0 max_acc.pth (whole view). Training GPU-hours are the measured wall-clock of our runs (one job on the GPU at a time).

| Method | Trainable parameters added | Training GPU-hours | One-off cohort cost | Extra inference time per episode (1-shot / 5-shot) | Extra GPU memory (1-shot / 5-shot) |
|---|---|---|---|---|---|
| Shared backbone forward (support + 150 queries, both flip views; all feature-based rows) | — | — | — | 30.7 / 34.6 ms | 417 / 471 MB |
| Cosine (COSF) | 0 | 0 (uses the pre-trained backbone) | none | 0.09 / 0.08 ms | < 0.01 MB |
| CCP-A | 0 | 0 | embed 38,400 base images, both views: 7.7 s (4,970 images/s); store the unit flip-view bank: 46.9 MB fp16 (93.8 MB fp32). Tiered: the same 38,400-image bank | 0.75 / 0.80 ms | 24.2 / 24.2 MB (+ bank 47 MB resident) |
| CCP-S | 0 | 0 | embed all base images once (mini 38,400: 7.7 s; tiered 448,695: about 90 s, scaled from the measured rate); store mean + covariance only: 1.56 MB fp32 | 0.49 / 0.41 ms | 0.37 / 0.37 MB |
| DNPG (its own score) | 11,008,123 in new modules (prototype calibrator, negative-prototype generator, base / open weights) + 204,800 Phase 2 centres; the 12,424,320-parameter backbone is also fine-tuned (lr 1e-4) | mini: 4.47 h per seed (Phase 2 + meta 1-shot + meta 5-shot; seeds 1, 2 measured), 13.4 h for 3 seeds; tiered seed 0: 12.1 h (5-shot with gradient accumulation) | none (Phase 2 open weights are part of training) | 67.2 / 127.7 ms (its own forward with 5x support augmentation; replaces the shared backbone row) | 1067 / 2135 MB |
| DNPG + CCP (H_fix = D + w · CCP-A) | 0 beyond DNPG (w is one frozen scalar per shot) | 0 beyond DNPG | embed 38,400 base images with DNPG's backbone: 7.7 s; bank 46.9 MB fp16 | +31.5 / +35.4 ms over DNPG (a flip-view forward of the episode through DNPG's backbone ≈ the shared-backbone cost, plus CCP-A ≈ 0.8 ms) | +450 / +508 MB |

Notes:
- CCP-A / CCP-S add no trainable parameters and no training; their one-off cost is a single embedding pass over the base images, and CCP-S keeps only a 640-d mean and a 640 x 640 covariance.
- Per-episode scoring on top of the backbone forward is below 1 ms for COSF, CCP-A and CCP-S. For DNPG + CCP the extra time is dominated by the second (flip-view) forward pass: DNPG's own forward uses augmented supports and no flipped views, so its features are not reused here. Flip views off (ablation e) would drop that pass at a validation cost of about 0.2-0.4 AUROC.
- Training hours are wall-clock on one RTX 4090 under Windows (num_workers 0). They exclude the GPU time lost to the system-memory fallback (research/dnpg/DNPG_SETUP.md).
