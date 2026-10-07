import yaml
import numpy as np
from numba import njit
from rich.console import Console
from rich.table import Table
from src.common import days_for, split
from src.sim.queue import load

p = yaml.safe_load(open("configs/sim.yaml"))
GRID = [(sk, wd) for sk in (0, 1, 2, 4) for wd in (0, 1, 2)]
HS = (0.0, 0.5, 1.0)
TICKS = {"BTCUSDT": 0.1, "ETHUSDT": 0.01}

@njit(cache=True)
def sim_path(bb, ba, bq, aq, sig, tsec, tsell, tp, ta, hs_bps, widen, skew, inv_skew, fee, size, qmax, tick, keep, seed):
    np.random.seed(seed)
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
            act = np.random.random() < keep
            bid_on, ask_on = (q < qmax) and act, (q > -qmax) and act
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

t = Table(title="Volume-matched control (queue model, fee 0, held-out): is the gain more than trading less?")
for c in ("asset", "hs bps", "strategy", "fills", "USDT", "bps/fill", "vs thinned: diff USDT [95% CI]"):
    t.add_column(c, justify="left" if c in ("asset", "strategy") else "right")
for sym in ("BTCUSDT", "ETHUSDT"):
    days = days_for(sym); train, test = split(days)
    data = {d: load(d, sym) for d in days}
    tick = TICKS[sym]
    avg = np.mean([(data[d][0].mean() + data[d][1].mean()) / 2 for d in test])
    def path(d, hs, g, keep=1.0, seed=0):
        return sim_path(*data[d], hs, float(g[1]), float(g[0]), float(p["inv_skew_bps"]), 0.0,
                        float(p["clip"]), int(p["qmax"]), tick, keep, seed)
    for hs in HS:
        score = {g: sum(path(d, hs, g)[0].sum() for d in train) for g in GRID}
        best = max(score, key=score.get)
        fv = sum(path(d, hs, (0, 0))[1] for d in train); fc = sum(path(d, hs, best)[1] for d in train)
        keep = min(1.0, fc / fv)
        def run(g, kp, tag):
            r = [path(d, hs, g, kp, 7) for d in test]
            return np.concatenate([x[0] for x in r]), sum(x[1] for x in r)
        van, fvan = run((0, 0), 1.0, "v")
        cho, fcho = run(best, 1.0, "c")
        thin, fthin = run((0, 0), keep, "t")
        ci = boot_ci(cho - thin)
        for name, x, f in (("vanilla", van, fvan), (f"chosen {best}", cho, fcho), (f"vanilla thinned keep={keep:.2f}", thin, fthin)):
            t.add_row(sym, f"{hs}", name, str(f), f"{x.sum():+.1f}", f"{x.sum() / (f * float(p['clip']) * avg) * 1e4:+.2f}", "")
        t.add_row(sym, f"{hs}", "chosen minus thinned", "", f"{(cho - thin).sum():+.1f}", "", f"[{ci[0]:+.1f}, {ci[1]:+.1f}]")
Console().print(t)
