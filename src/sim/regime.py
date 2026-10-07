import yaml
import numpy as np
from rich.console import Console
from rich.table import Table
from src.common import days_for, split
from src.sim.queue import load
from src.sim.core import sim_path, boot_ci

p = yaml.safe_load(open("configs/sim.yaml"))
GRID = [(sk, wd) for sk in (0, 1, 2, 4) for wd in (0, 1, 2)]
TICKS = {"BTCUSDT": 0.1, "ETHUSDT": 0.01}
HS = 0.0

def hourly_vol(bb, ba):
    lm = np.log((bb + ba) / 2)
    r = np.diff(lm, prepend=lm[0])
    nh = len(lm) // 3600
    return r[: nh * 3600].reshape(nh, 3600).std(axis=1) * 1e4

t = Table(title="Regime split (previous-hour realized vol vs train median), queue model, fee 0, half-spread 0bp, held-out")
for c in ("asset", "regime", "hours", "vanilla", "chosen", "thinned", "chosen-thinned", "95% CI (iid hours)"):
    t.add_column(c, justify="left" if c in ("asset", "regime") else "right")
for sym in ("BTCUSDT", "ETHUSDT"):
    days = days_for(sym); train, test = split(days)
    data = {d: load(d, sym) for d in days}
    tick = TICKS[sym]
    def path(d, g, keep=1.0, seed=0):
        return sim_path(*data[d], HS, float(g[1]), float(g[0]), float(p["inv_skew_bps"]), 0.0,
                        float(p["clip"]), int(p["qmax"]), tick, keep, seed)
    score = {g: sum(path(d, g)[0].sum() for d in train) for g in GRID}
    best = max(score, key=score.get)
    keep = min(1.0, sum(path(d, best)[1] for d in train) / sum(path(d, (0, 0))[1] for d in train))
    thr = np.median(np.concatenate([hourly_vol(data[d][0], data[d][1]) for d in train]))
    V, C, T, R = [], [], [], []
    for d in test:
        v = hourly_vol(data[d][0], data[d][1])
        van, cho, thn = path(d, (0, 0))[0], path(d, best)[0], path(d, (0, 0), keep, 7)[0]
        V.append(van[1:]); C.append(cho[1:]); T.append(thn[1:]); R.append(v[:-1] > thr)
    V, C, T, R = map(np.concatenate, (V, C, T, R))
    for name, m in (("low vol", ~R), ("high vol", R)):
        diff = (C - T)[m]
        lo, hi = boot_ci(diff, blk=1)
        t.add_row(sym, name, str(int(m.sum())), f"{V[m].sum():+.1f}", f"{C[m].sum():+.1f}", f"{T[m].sum():+.1f}",
                  f"{diff.sum():+.1f}", f"[{lo:+.1f}, {hi:+.1f}]")
Console().print(t)
