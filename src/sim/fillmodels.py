"""Descriptive analysis (not a hypothesis test): how much does the fill model matter?
Outputs fixed before running, no parameter is tuned (chosen config (skew 1, widen 2) is the v1 choice):
 1. Fill models A/B/C x {vanilla, chosen} x half-spread {0, 1} bp, fee 0, lag 0: total USDT, fills, bps/fill over all days, per asset.
    Gap A-C and B-C in USDT/day with 95% CI (bootstrap over days, 5000).
 2. Fee grid, model C (fee in bps, negative = rebate): total USDT, and linear breakeven maker fee.
 3. Latency, model C: total USDT for extra lag 0, 1, 2 s.
  python -m src.sim.fillmodels --selfcheck   # chosen/model C fills on v1 days must equal confirm-v1 (83000 BTC, 199876 ETH)
  python -m src.sim.fillmodels               # full run, writes results/fillmodels.txt
"""
import sys
import numpy as np, yaml
from rich.console import Console
from rich.table import Table
from src.common import days_for
from src.sim.queue import load, sim_q
from src.sim.v2 import V1_CONF

p = yaml.safe_load(open("configs/sim.yaml"))
TICKS = {"BTCUSDT": 0.1, "ETHUSDT": 0.01}
SYMS = ("BTCUSDT", "ETHUSDT")
CFGS = {"vanilla": (0.0, 0.0), "chosen": (1.0, 2.0)}   # (skew, widen)
MODES = {0: "A naive", 1: "B tick-aware", 2: "C tick+queue"}
HS = (0.0, 1.0)
FEES = (-1.0, -0.5, 0.5, 1.0, 2.0)   # bps besides 0; negative = rebate
LAGS = (1, 2)

def one(data, sym, mode, hs, cfg, fee=0.0, lag=0):
    sk, wd = CFGS[cfg]
    return sim_q(*data, int(mode), float(hs), float(wd), float(sk), float(p["inv_skew_bps"]),
                 float(fee) * 1e-4, float(p["clip"]), int(p["qmax"]), int(lag), TICKS[sym])

def keys():
    ks = []
    for mode in MODES:
        for hs in HS:
            for cfg in CFGS:
                ks.append((mode, hs, cfg, 0.0, 0))
    for hs in HS:
        for cfg in CFGS:
            for fee in FEES:
                ks.append((2, hs, cfg, fee, 0))
            for lag in LAGS:
                ks.append((2, hs, cfg, 0.0, lag))
    return ks

def collect(sym, days):
    out = {k: [] for k in keys()}; notl = []
    for d in days:
        data = load(d, sym)
        notl.append((data[0].mean() + data[1].mean()) / 2 * float(p["clip"]))
        for k in out:
            mode, hs, cfg, fee, lag = k
            out[k].append(one(data, sym, mode, hs, cfg, fee, lag))
        print(sym, d, flush=True)
    return out, np.array(notl)

def pf(out, k):
    a = np.array(out[k], dtype=float)
    return a[:, 0], a[:, 1]

def boot(x, B=5000, seed=0):
    rng = np.random.default_rng(seed); n = len(x)
    m = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    return np.percentile(m, [2.5, 97.5])

if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        want = {"BTCUSDT": 83000, "ETHUSDT": 199876}
        for sym in SYMS:
            f = sum(one(load(d, sym), sym, 2, 0.0, "chosen")[1] for d in V1_CONF[sym])
            print(sym, "chosen fills on v1 days:", f, "expected", want[sym])
        sys.exit(0)
    R = {}
    for sym in SYMS:
        days = days_for(sym)
        R[sym] = (days,) + collect(sym, days)
    con = Console(record=True)
    t1 = Table(title="1. Fill models: all days, fee 0, lag 0, 1s refresh")
    for c in ("asset", "days", "hs bp", "config", "fill model", "USDT", "fills", "bps/fill"):
        t1.add_column(c, justify="left" if c in ("asset", "config", "fill model") else "right")
    t1b = Table(title="1b. Realism gap per day, USDT/day [95% CI over days] (positive = naive model overstates)")
    for c in ("asset", "hs bp", "config", "A - C", "B - C"):
        t1b.add_column(c, justify="left" if c in ("asset", "config") else "right")
    t2 = Table(title="2. Model C fee grid: total USDT by maker fee (bps, negative = rebate); breakeven = fee where PnL = 0")
    for c in ("asset", "hs bp", "config") + tuple(f"{f:+.1f}" for f in sorted(FEES + (0.0,))) + ("breakeven bps",):
        t2.add_column(c, justify="left" if c in ("asset", "config") else "right")
    t3 = Table(title="3. Model C latency: total USDT by extra quote lag (s)")
    for c in ("asset", "hs bp", "config", "lag 0", "lag 1", "lag 2"):
        t3.add_column(c, justify="left" if c in ("asset", "config") else "right")
    for sym in SYMS:
        days, out, notl = R[sym]
        for hs in HS:
            for cfg in CFGS:
                pn = {}
                for mode, mname in MODES.items():
                    pnl, fl = pf(out, (mode, hs, cfg, 0.0, 0))
                    pn[mode] = pnl
                    t1.add_row(sym, str(len(days)), f"{hs:.0f}", cfg, mname, f"{pnl.sum():+.1f}", str(int(fl.sum())),
                               f"{pnl.sum() / (fl * notl).sum() * 1e4:+.3f}")
                ac, bc = boot(pn[0] - pn[2]), boot(pn[1] - pn[2])
                t1b.add_row(sym, f"{hs:.0f}", cfg, f"{(pn[0] - pn[2]).mean():+.2f} [{ac[0]:+.2f}, {ac[1]:+.2f}]",
                            f"{(pn[1] - pn[2]).mean():+.2f} [{bc[0]:+.2f}, {bc[1]:+.2f}]")
                row = {f: pf(out, (2, hs, cfg, f, 0))[0].sum() for f in FEES}
                row[0.0] = pf(out, (2, hs, cfg, 0.0, 0))[0].sum()
                be = -row[0.0] / (row[1.0] - row[0.0]) if row[1.0] != row[0.0] else float("nan")
                t2.add_row(sym, f"{hs:.0f}", cfg, *[f"{row[f]:+.1f}" for f in sorted(row)], f"{be:+.2f}")
                lg = [pf(out, (2, hs, cfg, 0.0, l))[0].sum() for l in (0,) + LAGS]
                t3.add_row(sym, f"{hs:.0f}", cfg, *[f"{x:+.1f}" for x in lg])
    for t in (t1, t1b, t2, t3):
        con.print(t)
    open("results/fillmodels.txt", "w").write(con.export_text())
