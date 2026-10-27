# MINI-SNATCHER test: paired differences (from the saved per-episode file; no re-evaluation)

Source: research/fsor/final_mini_snatcher_episodes.npz (sha256 a5323f41e155a7d08e658a8733879a1be65859fad567b9e6ded859d7bba49514), written by the single MINI-SNATCHER test pass (token 19da50a9bf457980). 600 test episodes per shot; one released FEAT model per shot (no seeds), so each episode contributes one difference. Mean over the 600 episodes, 95% CI over episodes (1.96 x std / sqrt(600)), two-sided Wilcoxon signed-rank p-value (scipy.stats.wilcoxon, zero-method 'wilcox'), fraction of episodes with a positive difference. No decisions.

| Shot | Difference | Mean (AUROC points) | 95% CI | Wilcoxon p | Episodes > 0 | Episodes = 0 |
|---|---|---|---|---|---|---|
| 1 | H_fix − S | +1.671 | ± 0.207 | 3.98e-45 | 75.3% | 0.2% |
| 1 | H_fix − K_fix | +1.410 | ± 0.145 | 1.41e-57 | 79.8% | 0.7% |
| 5 | H_fix − S | +2.102 | ± 0.172 | 1.69e-75 | 84.2% | 0.2% |
| 5 | H_fix − K_fix | +1.894 | ± 0.117 | 1.37e-94 | 92.7% | 0.0% |
