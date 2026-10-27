"""Differential testing against the 1.5.0 package of the authors' earlier release repository (private) (read only; its path in $CCP_OLD_PKG).

Tests register jobs (OldJobs.add); the first `result` call runs every registered job in ONE subprocess (tests/_old_runner.py) and caches
the answers for the session. The new code runs in the test process; the comparison is bitwise unless a test documents a tolerance."""
from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


class OldJobs:
    def __init__(self, old_repo: Path | None):
        self.old = Path(old_repo) if old_repo is not None else None
        self.jobs: list[dict] = []
        self.results: dict | None = None

    @property
    def available(self) -> bool:
        return self.old is not None and (self.old / "ccp" / "cohort.py").exists()

    def add(self, job_id: str, func: str | None = None, *args, battery: str | None = None, **kwargs) -> str:
        if self.results is not None:
            raise RuntimeError("jobs must be registered before the first result is requested")
        job = {"id": job_id}
        if battery:
            job.update(battery=battery, **kwargs)
        else:
            job.update(func=func, args=list(args), kwargs=kwargs)
        self.jobs.append(job)
        return job_id

    def run(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "in.pkl", Path(d) / "out.pkl"
            pickle.dump(self.jobs, open(src, "wb"))
            r = subprocess.run([sys.executable, "-I", str(HERE / "_old_runner.py"), str(self.old), str(src), str(dst)],
                               capture_output=True, text=True, cwd=d)
            if r.returncode != 0:
                raise RuntimeError(f"old-package runner failed:\n{r.stderr[-3000:]}")
            self.results = pickle.load(open(dst, "rb"))

    def result(self, job_id: str):
        if self.results is None:
            self.run()
        status, value = self.results[job_id]
        if status != "ok":
            raise AssertionError(f"old package raised for {job_id}: {value}")
        return value
