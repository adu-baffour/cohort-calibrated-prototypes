## 2. Pre-registration (committed before any valB or tvB scoring)

Datasets and episodes: mini valA_fsor / valB_fsor (data/episodes_mini, 1,000 each; eval view Resize(92) + CenterCrop(84), RGB); tiered tvA_fsor / tvB_fsor (data/episodes_tiered, 1,000 each; 84 x 84, BGR). Checkpoint = FEAT's meta-trained model of that shot (table above).

Rows, per dataset and shot, on the FEAT meta-trained encoder:
- S = SnaTCHer score (as above).
- H_fix = S + w · CCP-A; K_fix = S + w2 · COSF.
- CCP-A alone, COSF alone, and FEAT's distance score Dist = −max logit.
- CCP-A exactly as registered: symmetric, mean of the top-1000 cosines to base images embedded with the same encoder, flip views; prototype = normalised mean of the flip-averaged support embeddings; encoder features (layer-4 GAP, registered ccp_premise._gap convention), no slf_attn. Bank: mini = all 38,400 base images (eval view, RGB); tiered = the RandomState(20260926) subset of 38,400 base images (as in T1 and the TIERED bundle), BGR.
- w = pooled std(S) / pooled std(CCP-A), w2 = pooled std(S) / pooled std(COSF), ddof 0 over all 1,000 x 150 valA (tvA) query scores, per dataset and shot, written to research/status/snatcher_weights.json before any valB (tvB) episode is scored.
- Accuracy = FEAT's classifier (argmax logits) for S, H_fix, K_fix and Dist; CCP-A and COSF alone also report their own argmax accuracy.

Rule, per dataset (AUROC, unknown = positive):
- S1: H_fix − S ≥ +0.50 on valB (tvB), both shots.
- S2: H_fix − K_fix ≥ +0.30 on valB (tvB), both shots.
- S3: H_fix − S > 0 on valA (tvA), both shots.
- GO (S1, S2, S3 all hold) → test path and bundle for that dataset. NO-GO → report and stop for that dataset.
