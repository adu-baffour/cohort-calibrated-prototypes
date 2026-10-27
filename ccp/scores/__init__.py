"""Score registry. Importing this package registers every score module in it (under the paper's names)."""
import importlib
import pkgutil

from ccp.scores.base import (  # noqa: F401
    EpisodeContext,
    TargetFeatures,
    get_evidence,
    get_score,
    register_evidence,
    registered_evidence,
    register_score,
    registered_scores,
    score_episode,
)

for _m in pkgutil.iter_modules(__path__):
    if _m.name != "base":
        importlib.import_module(f"{__name__}.{_m.name}")

REGISTERED = ["COS", "COSF", "CCP-A", "CCP-S"]
