# MINI-DNPG test: paired differences pooled over seeds (from the saved per-episode file; no re-evaluation)

Source: research/fsor/final_mini_dnpg_episodes.npz (sha256 c42ea5f8a947c01eba06b6d1ec6fb7dda2af336f353876150fa89019aa6ccb0e), written by the single MINI-DNPG test pass (token 01fcac4666048f3e). 600 test episodes per shot. For each episode the per-episode AUROC difference is averaged over seeds 0, 1, 2; the table reports the mean over the 600 episodes, the 95% CI over episodes (1.96 x std / sqrt(600)), a two-sided Wilcoxon signed-rank p-value (scipy.stats.wilcoxon, zero-method 'wilcox') and the fraction of episodes with a positive difference. No decisions.

| Shot | Difference | Mean (AUROC points) | 95% CI | Wilcoxon p | Episodes > 0 | Episodes = 0 |
|---|---|---|---|---|---|---|
| 1 | H_fix − D | +0.583 | ± 0.135 | 1.08e-16 | 65.5% | 0.0% |
| 1 | H_fix − K_fix | +0.989 | ± 0.155 | 1.81e-29 | 69.3% | 0.0% |
| 5 | H_fix − D | +0.727 | ± 0.120 | 1.11e-26 | 67.3% | 0.0% |
| 5 | H_fix − K_fix | +1.942 | ± 0.156 | 8.88e-77 | 85.3% | 0.0% |
