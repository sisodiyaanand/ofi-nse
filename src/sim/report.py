import yaml
import numpy as np
from numba import njit
from rich.console import Console
from rich.table import Table
from src.common import days_for, split
from src.sim.latency import load

p = yaml.safe_load(open("configs/sim.yaml"))
GRID = [(sk, wd) for sk in (0, 1, 2, 4) for wd in (0, 1, 2)]

@njit(cache=True)
def sim_path(mid, lo, hi, sig, hs_bps, widen, skew, inv_skew, fee, size, qmax, per_hour):
    n, q, cash, fills, last, j = len(mid), 0, 0.0, 0, 0.0, 0
    out = np.zeros(n // per_hour + 1)
    for i in range(1, n):
        m, s = mid[i - 1], sig[i - 1]
        r = m * (1 - inv_skew * 1e-4 * q / qmax + skew * 1e-4 * s)
        hs = m * (hs_bps + widen * abs(s)) * 1e-4
        bid, ask = min(r - hs, m), max(r + hs, m)
        if lo[i] < bid and q < qmax:
            cash -= bid * size * (1 + fee); q += 1; fills += 1
        if hi[i] > ask and q > -qmax:
            cash += ask * size * (1 - fee); q -= 1; fills += 1
        if (i + 1) % per_hour == 0:
            v = cash + q * size * mid[i]
            out[j] = v - last; last = v; j += 1
    return out[:j], fills

def path(data, d, sk, wd):
    mid, lo, hi, dimb = data[d]
    return sim_path(mid, lo, hi, dimb, 1.0, float(wd), float(sk), float(p["inv_skew_bps"]), 0.0,
                    float(p["clip"]), int(p["qmax"]), 3600)

def boot_ci(diff, B=5000, blk=3, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff); nb = int(np.ceil(n / blk)); tot = np.empty(B)
    for b in range(B):
        idx = (rng.integers(0, n, nb)[:, None] + np.arange(blk)).ravel()[:n] % n
        tot[b] = diff[idx].sum()
    return np.percentile(tot, [2.5, 97.5])

t = Table(title="Held-out: train-selected config vs vanilla, 1s bars, half-spread 1bp, fee 0, hourly block bootstrap")
for c in ("asset", "train/test days", "chosen (skew,widen)", "vanilla USDT", "chosen USDT", "diff USDT", "95% CI of diff", "hours diff>0"):
    t.add_column(c, justify="right" if c != "asset" else "left")
for sym in ("BTCUSDT", "ETHUSDT"):
    days = days_for(sym); train, test = split(days)
    data = {d: load(d, 1000, sym)[:4] for d in days}
    score = {g: sum(path(data, d, *g)[0].sum() for d in train) for g in GRID}
    best = max(score, key=score.get)
    van = np.concatenate([path(data, d, 0, 0)[0] for d in test])
    cho = np.concatenate([path(data, d, *best)[0] for d in test])
    diff = cho - van
    lo_, hi_ = boot_ci(diff)
    t.add_row(sym, f"{len(train)}/{len(test)}", str(best), f"{van.sum():+.1f}", f"{cho.sum():+.1f}",
              f"{diff.sum():+.1f}", f"[{lo_:+.1f}, {hi_:+.1f}]", f"{(diff > 0).mean() * 100:.0f}% of {len(diff)}")
Console().print(t)
