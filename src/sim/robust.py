import yaml
import numpy as np
from numba import njit
from rich.console import Console
from rich.table import Table
from src.common import days_for, split
from src.sim.queue import load

p = yaml.safe_load(open("configs/sim.yaml"))
GRID = [(sk, wd) for sk in (0, 1, 2, 4) for wd in (0, 1, 2)]
HS = (0.0, 0.25, 0.5, 1.0)
TICKS = {"BTCUSDT": 0.1, "ETHUSDT": 0.01}

@njit(cache=True)
def sim_path(bb, ba, bq, aq, sig, tsec, tsell, tp, ta, mode, hs_bps, widen, skew, inv_skew, fee, size, qmax, tick):
    n, nt, q, cash, fills, j = len(bb), len(tsec), 0, 0.0, 0, 0
    bid_px, ask_px, bid_ahead, ask_ahead = -1.0, -1.0, 0.0, 0.0
    bid_on, ask_on = False, False
    tol = 0.5 * tick
    out = np.zeros(n // 3600 + 1)
    last, h = 0.0, 0
    for i in range(n):
        k = i - 1
        if k >= 0:
            m, s = (bb[k] + ba[k]) / 2.0, sig[k]
            r = m * (1 - inv_skew * 1e-4 * q / qmax + skew * 1e-4 * s)
            hs = m * (hs_bps + widen * abs(s)) * 1e-4
            nb = np.floor(min(r - hs, bb[k]) / tick + 1e-9) * tick
            na = np.ceil(max(r + hs, ba[k]) / tick - 1e-9) * tick
            if (nb != bid_px) or (not bid_on):
                bid_ahead = bq[k] if abs(nb - bb[k]) < tol else 0.0
            if (na != ask_px) or (not ask_on):
                ask_ahead = aq[k] if abs(na - ba[k]) < tol else 0.0
            bid_px, ask_px = nb, na
            bid_on, ask_on = q < qmax, q > -qmax
        while j < nt and tsec[j] == i:
            pr, a = tp[j], ta[j]
            if tsell[j]:
                if bid_on:
                    hit = pr < bid_px - tol
                    if (not hit) and abs(pr - bid_px) < tol:
                        bid_ahead -= a
                        hit = bid_ahead < 0
                    if hit:
                        cash -= bid_px * size * (1 + fee); q += 1; fills += 1; bid_on = False
            else:
                if ask_on:
                    hit = pr > ask_px + tol
                    if (not hit) and abs(pr - ask_px) < tol:
                        ask_ahead -= a
                        hit = ask_ahead < 0
                    if hit:
                        cash += ask_px * size * (1 - fee); q -= 1; fills += 1; ask_on = False
            j += 1
        if (i + 1) % 3600 == 0:
            v = cash + q * size * (bb[i] + ba[i]) / 2.0
            out[h] = v - last; last = v; h += 1
    return out[:h], fills

def boot_ci(diff, B=5000, blk=3, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff); nb = int(np.ceil(n / blk)); tot = np.empty(B)
    for b in range(B):
        idx = (rng.integers(0, n, nb)[:, None] + np.arange(blk)).ravel()[:n] % n
        tot[b] = diff[idx].sum()
    return np.percentile(tot, [2.5, 97.5])

t = Table(title="Queue-position fill model, fee 0, 1s refresh: train-selected (skew,widen) vs vanilla, held-out, hourly block bootstrap")
for c in ("asset", "half-spread bps", "chosen", "vanilla USDT", "chosen USDT", "diff USDT", "95% CI of diff", "fills van/chosen"):
    t.add_column(c, justify="left" if c == "asset" else "right")
for sym in ("BTCUSDT", "ETHUSDT"):
    days = days_for(sym); train, test = split(days)
    data = {d: load(d, sym) for d in days}
    tick = TICKS[sym]
    def path(d, hs, g):
        return sim_path(*data[d], 2, hs, float(g[1]), float(g[0]), float(p["inv_skew_bps"]), 0.0,
                        float(p["clip"]), int(p["qmax"]), tick)
    for hs in HS:
        score = {g: sum(path(d, hs, g)[0].sum() for d in train) for g in GRID}
        best = max(score, key=score.get)
        rv = [path(d, hs, (0, 0)) for d in test]; rc = [path(d, hs, best) for d in test]
        van = np.concatenate([r[0] for r in rv]); cho = np.concatenate([r[0] for r in rc])
        diff = cho - van; lo, hi = boot_ci(diff)
        t.add_row(sym, f"{hs}", str(best), f"{van.sum():+.1f}", f"{cho.sum():+.1f}", f"{diff.sum():+.1f}",
                  f"[{lo:+.1f}, {hi:+.1f}]", f"{sum(r[1] for r in rv)}/{sum(r[1] for r in rc)}")
Console().print(t)
