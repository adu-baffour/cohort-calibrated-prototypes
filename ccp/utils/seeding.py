"""Seeds. Scoring is deterministic given the features; a seed only matters for DNPG's own test-time support augmentation (torch seed
1,000,000 * shot + episode index, as registered) and for training (scripts/train/)."""
from __future__ import annotations


def dnpg_episode_seed(shot: int, episode: int) -> int:
    """Registered per-episode torch seed for DNPG's stochastic support augmentation (docs/preregistration/DNPG_SETUP_sections_9-11.md)."""
    return 1_000_000 * shot + episode
