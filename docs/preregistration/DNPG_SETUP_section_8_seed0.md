## 8. Pre-registration before any seed-0 evaluation (advisor, 2026-09-25)

Written while seed 0 is still training (Phase 2 at epoch ~25 of 50; no meta checkpoint exists yet) and before any seed-0
evaluation. Harness: `scripts/dnpg_eval.py` (one checkpoint per call; every row below comes from that same checkpoint).

1. **Information-only row** (not part of the decision rule): DNPG own score + CCP-A, per-episode z-scored sum over the episode's
   150 queries (`(x - mean)/std + (y - mean)/std`), same checkpoint. Closed-set accuracy for this row = DNPG's own classifier.
2. **Decision rule on seed 0**, `max_acc.pth` checkpoint, paired on the same checkpoint, DNPG's own-score rows labelled
   "selected on validation classes (optimistic vs training-free rows)":
   - **GO** if CCP-A AUROC − DNPG own-score AUROC ≥ +0.5 at 1-shot AND ≥ +0.5 at 5-shot on valA_fsor, with the same sign
     (CCP-A − DNPG own > 0) on valB_fsor at both shots.
   - Otherwise **NO-GO**: recorded as "CCP does not improve a trained FSOR head (DNPG)"; no further DNPG training.
3. **If GO, the paper's host chain is the clean chain:**
   (a) Re-run TANE's released pre-training code (`third_party/TANE/pretrain`, fe089d4) on the mini-ImageNet base classes with its own
       recipe (90 epochs, CE + rotation, SGD lr 0.05, decay x0.1 at epoch 60), seed 0, keeping `last.pth` at epoch 90. No checkpoint
       selection, and no test or val partition read during pre-training: its evaluation loader is pointed at nothing or at
       train-class episodes only (which one is recorded when it is set up).
   (b) DNPG Phase 2 and meta-learning from that backbone, seeds 0, 1, 2, same recipe and deviations as in this document, selection on
       our 16 val classes as ruled.
   (c) The released-backbone seed 0 stays as a reported comparison row, labelled "released pre-trained backbone (stored epoch 66,
       consistent with test-selected max_meta)".
   (d) The MINI-DNPG bundle is drafted on the clean chain: headline = DNPG `max_acc` + CCP-A vs DNPG own score, same checkpoint,
       mean ± std over seeds 0-2; last-epoch rows and the combined row are non-headline.

