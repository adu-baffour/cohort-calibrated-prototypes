"""Shim for tensorboardX.SummaryWriter (DNPG uses add_scalar only): appends rows to <logdir>/scalars.csv."""
import os
import time


class SummaryWriter:
    def __init__(self, logdir=None, *args, **kwargs):
        self.path = None
        if logdir:
            os.makedirs(logdir, exist_ok=True)
            self.path = os.path.join(logdir, "scalars.csv")

    def add_scalar(self, tag, value, step=None, *args, **kwargs):
        if self.path:
            with open(self.path, "a") as f:
                f.write("{:.0f},{},{},{}\n".format(time.time(), tag, step, float(value)))

    def flush(self):
        pass

    def close(self):
        pass
