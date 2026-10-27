# Pre-registration: training-free baselines (written and committed 2026-09-30, before any baseline is scored on any split)

Purpose: compare CCP with training-free alternatives that use the same information (the host's own encoder features and the same base
bank). **There is no go / no-go.** Baselines are comparisons; their validation and test results will be reported whatever they are. CCP's
registered settings are not changed (CCP-A: symmetric, K = 1000, flip views; H_fix with the frozen weights of configs/hosts). No training.

## Hosts and data

| Host | Dataset | Validation (select / report) | Bank (registered, `scripts/build_bank.py`) |
|---|---|---|---|
| frozen FEAT (`frozen_feat_mini`) | mini-ImageNet | valA / valB, 1,000 episodes per shot | all 38,400 base images; statistics over the same |
| frozen FEAT (`frozen_feat_tiered`) | tieredImageNet | tvA / tvB | RandomState(20260926) subset of 38,400; C, b_bar, class means over all 448,695 |
| SnaTCHer-F (`snatcher_f_mini`) | mini-ImageNet | valA / valB | all 38,400, per shot (FEAT meta-trained encoder of that shot) |
| DNPG seeds 0, 1, 2 (`dnpg_mini_seed{0,1,2}`) | mini-ImageNet | valA / valB | all 38,400, per seed and shot (DNPG backbone) |

Notation (all from `ccp`): z(x) = unit flip-view embedding (Eq. 1); p_k = registered prototype (Eq. 2); c_k(q) = z(q) . p_k; B = the unit
flip-view bank; b_bar and C = mean and covariance (ddof 0) of the unit flip-view base embeddings (stored with the bank); mu_b = unit base-class
means (stored with the bank); g(x) = raw flip-summed GAP vector g(x) + g(flip x) (unnormalised); m_K(v) = mean of the K largest cosines of v
to B (Eq. 3). Episode layout as everywhere (5 known + 5 unknown classes, 15 queries per class; 600 test episodes per shot).

## Baselines (one file each in `ccp/scores/`)

Two kinds. **Matrix baselines** give a per-class score s_k(q); open-set score u(q) = −max_k s_k(q) (higher = more unknown), label =
argmax_k s_k(q). **Evidence baselines** give one number e(q) per query (higher = more unknown) and no label.

| Id | Name | Kind | Definition | Grid |
|---|---|---|---|---|
| B1 | QTERM | evidence | e(q) = m_K(q), K = 1000: the query cohort term of CCP-A alone (the key control: does the query term alone reproduce CCP's gain?) | none |
| B2 | KNN (Sun et al., ICML 2022) | evidence | s_(k)(q) = the k-th largest cosine of z(q) to B; sign +: e = −s_(k) (far from the base images = unknown, the direction of Sun et al.); sign −: e = +s_(k) | k in {10, 100, 1000} x sign in {+, −} |
| B3 | RMD (Ren et al., 2021) | matrix | Sigma_lambda = (1 − lambda) C + lambda (tr C / 640) I (shared covariance from the base bank, shrunk towards a scaled identity); MD(x, mu) = (x − mu)^T Sigma_lambda^-1 (x − mu); MD_k = MD(z(q), p_k), MD_0 = MD(z(q), b_bar); s_k = −(MD_k − MD_0), so u = min_k (MD_k − MD_0) and label = argmin_k MD_k | lambda in {0.1, 0.5, 1.0} |
| B4 | CL2N (SimpleShot, Wang et al., 2019) | matrix | f(v) = normalise(v − b_bar) applied to z; prototype p'_k = normalise(mean over the K supports of f(z(s))); s_k = cos(f(z(q)), p'_k) | none |
| B5 | ZN (Fei et al., ICCV 2021) | matrix | n(v) = (v − mean(v)) / std(v) over the 640 dimensions of g(x) (per-vector z-score); prototype p''_k = normalise(mean over the supports of n(g(s))); s_k = cos(n(g(q)), p''_k) | none |
| B6 | NNN (Chowdhury et al., EMNLP 2024) | matrix | prototype bias: s_k = c_k − alpha · m_k(p_k), m_k = mean of the k largest cosines of p_k to B (prototypes are the gallery; the bank is the reference set) | alpha in {0.25, 0.5, 0.75, 1} x k in {100, 1000} |
| B7a | QB-IS (Bogolin et al., CVPR 2022) | matrix | querybank = B, gallery = the 5 prototypes; Z_k = log sum_{b in B} exp(beta cos(b, p_k)); s_k = beta c_k − Z_k + log |B| (inverted softmax, in log space) | beta in {10, 20, 50} |
| B7b | QB-DIS (Bogolin et al., CVPR 2022) | matrix | activation (hub) set A = {k : k = argmax_j cos(b, p_j) for at least one b in B} (top-1, k = 1); for a query with k* = argmax_k c_k: if k* in A, s_k = beta c_k − Z_k + log |B| (IS), else s_k = beta c_k (unnormalised, the same units: IS with Z_k replaced by log |B|) | beta in {10, 20, 50} |
| B8 | TEEN (Wang et al., NeurIPS 2023) | matrix | p^_k = normalise(alpha p_k + (1 − alpha) sum_b w_kb mu_b), w_kb = softmax over base classes b of tau cos(p_k, mu_b); s_k = cos(z(q), p^_k); paper defaults tau = 16, alpha = 0.5 | none (no tuning) |
| B9 | CSLS-episode (Lample et al., 2018) — **TRANSDUCTIVE REFERENCE** | matrix | neighbourhoods inside the episode: r(q) = mean of the 10 largest cosines of z(q) to the other 149 queries and the 5 prototypes; r(p_k) = mean of the 10 largest cosines of p_k to the 150 queries and the other 4 prototypes; s_k = 2 c_k − r(q) − r(p_k). Uses the other queries of the episode: not inductive, reported as a reference only | none (K = 10) |

Adaptations (stated before any result): RMD uses the base-bank covariance because few-shot classes have no covariance estimate; QB-Norm and NNN
treat the prototypes as the gallery and the base bank as the querybank / reference set; QB-DIS writes the unnormalised branch in the same
log-units as IS so that scores are comparable across queries (needed for AUROC); CL2N and ZN keep the registered flip views.

## Reported rows

- **Backbone host (frozen FEAT)**: matrix baselines alone (AUROC of u, accuracy of their own label); evidence baselines fused with the host's
  cosine: u = COSF + w_b · e with w_b = pooled std(COSF score) / pooled std(e) (ddof 0) over all valA (tvA) query scores, one constant per host and
  shot (accuracy = COSF label).
- **Trained hosts (SnaTCHer-F, DNPG)**: H_b = own + w_b · u_b, own = S (SnaTCHer-F) or D (DNPG), u_b = the baseline's open-set score (matrix) or
  evidence e (evidence), w_b = pooled std(own) / pooled std(u_b) over all valA query scores, one constant per host, seed and shot (the H_fix rule).
  Accuracy = the host's classifier.
- **References on the same episodes**: host alone (frozen FEAT: COSF, flip views, the features every baseline uses; COS also listed; SnaTCHer-F: S;
  DNPG: D) and host with CCP (frozen FEAT: CCP-A; trained hosts: H_fix with the frozen weights of configs/hosts).

## Selection rule (validation A only)

For each host (frozen FEAT mini, frozen FEAT tiered, SnaTCHer-F mini, DNPG mini pooled over seeds 0-2) and each baseline with a grid, the
hyperparameters are chosen by the mean valA (tvA) AUROC of the reported row over both shots (DNPG: over seeds 0-2 and both shots), then frozen
(the same values for both shots). Ties (within 1e-9) go to the smaller value, in this order: KNN sign + before −, then smaller k; NNN smaller
alpha, then smaller k; RMD smaller lambda; QB smaller beta. The fusion weights w_b are fitted on valA per host, seed and shot for the chosen values.
The selection and the weights are written to `results/baselines/selection_valA.json` and committed before any valB (tvB) episode is scored.

## Metrics

Per host, seed and shot on valB (tvB): AUROC and closed-set accuracy, mean and 95% CI over the 1,000 episodes (DNPG also mean ± std over
seeds); paired per-episode AUROC differences with 95% CIs (1.96 std / sqrt(n)) against the host alone and against the host with CCP (DNPG:
per-episode differences averaged over seeds 0-2, as in T1). The valA numbers of the chosen values are reported too. Outputs:
`results/baselines/validation.json` and `validation.md`.

## Test

A test bundle part `yaml-fsor-baselines` (same test manifests as the paper: mini frozen FEAT, SnaTCHer-F, DNPG seeds 0-2; tiered frozen FEAT)
with the frozen hyperparameters and weights, per-episode saving and paired statistics, guarded exactly as before (a token that must be in
`bundles/FINAL_APPROVED`, STARTED / DONE markers, refusal to run twice). It is run only after the human writes its token. Its results will be
reported whatever they are. Decisions after that test: none.
