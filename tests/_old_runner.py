"""Runs calls against the 1.5.0 package of the authors' earlier release repository (private) in a separate process (python -I, the old repository first on sys.path), so that no
module of this repository can be mixed in. Usage: python -I _old_runner.py <old repo> <in.pkl> <out.pkl>.

Each job is {"id", "func": "module.attr", "args": [...], "kwargs": {...}} or {"id", "battery": name, ...} for the episode batteries
below. Torch tensors in results are converted to numpy."""
import pickle
import sys

old, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, old)
import importlib  # noqa: E402

import torch  # noqa: E402

import ccp  # noqa: E402

assert ccp.__file__.replace("\\", "/").startswith(old.replace("\\", "/")), ccp.__file__


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: to_np(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_np(v) for v in x)
    return x


def call(name, args, kwargs):
    mod, attr = name.rsplit(".", 1)
    return getattr(importlib.import_module(mod), attr)(*args, **kwargs)


def episode_context(job):
    from ccp.cohort import Bank
    from ccp.scores import EpisodeContext, TargetFeatures
    b = Bank(**job["bank"])
    tf = TargetFeatures(torch.from_numpy(job["raw0"]), torch.from_numpy(job["rawf"]), b.to_device("cpu"))
    return [EpisodeContext(tf, r, job["shot"]) for r in job["rows"]]


def battery(job):
    import ccp.scores as S
    eps = episode_context(job)
    if job["battery"] == "scores":
        out = []
        for ep in eps:
            d = {n: S.get_score(n)(ep) for n in job["names"] if n in S.registered_scores()}
            d.update({n: S.get_evidence(n)(ep) for n in job["names"] if n in S.registered_evidence()})
            out.append(to_np(d))
        return out
    if job["battery"] == "variants":
        from ccp import variants as V
        return [to_np(V.episode_scores(ep)) for ep in eps]
    if job["battery"] == "final_round_rows":
        from ccp import final_round as FR
        return [to_np(FR.episode_rows(ep, job["names"], set(job["evidence"]))) for ep in eps]
    if job["battery"] == "score_episode":
        from ccp.scores import score_episode
        return [to_np({k: (v[0], v[1]) for k, v in score_episode(ep, job["names"]).items()}) for ep in eps]
    raise KeyError(job["battery"])


jobs = pickle.load(open(src, "rb"))
res = {}
for j in jobs:
    try:
        res[j["id"]] = ("ok", to_np(battery(j) if "battery" in j else call(j["func"], j.get("args", []), j.get("kwargs", {}))))
    except Exception as e:  # reported by the test that owns the job
        res[j["id"]] = ("error", f"{type(e).__name__}: {e}")
pickle.dump(res, open(dst, "wb"))
