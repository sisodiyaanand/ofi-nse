"""Robustness checks for the v3 result, see ROBUSTNESS-v3.md.
  python -m src.sim.robust3 --selftest | python -m src.sim.robust3  (writes results/robust_v3.txt; ~30-60 min)
"""
import os, sys, tempfile, shutil
import yaml
import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from src.sim.queue import load
from src.sim.v2 import sim_h
from src.sim.v3 import detect_tick, NOTIONAL, CHOSEN, SYMS, CFG_SIM
import src.sim.v3 as v3

OUT = "results/robust_v3.txt"
SIX = ("BTCUSDT", "ETHUSDT") + SYMS
SHIFTS_H = (6, 9, 12)
NSEED_P = 10
B = 5000

def thinned_baseline(data, size, tick, p, keep, nseed):
    mm = [sim_h(*data, 0.0, 0.0, 0.0, float(p["inv_skew_bps"]), 0.0, size, int(p["qmax"]), tick, keep, s) for s in range(nseed)]
    return np.mean([x[0] for x in mm], axis=0), np.mean([x[1] for x in mm], axis=0)

def day_agg(sym, day, tick, p):
    data = load(day, sym)
    mid = (data[0] + data[1]) / 2
    size = NOTIONAL / float(mid.mean()); nt = size * float(mid.mean())
    def chosen(d):
        return sim_h(*d, 0.0, CHOSEN[1], CHOSEN[0], float(p["inv_skew_bps"]), 0.0, size, int(p["qmax"]), tick, 1.0, 0)
    pv, fv = sim_h(*data, 0.0, 0.0, 0.0, float(p["inv_skew_bps"]), 0.0, size, int(p["qmax"]), tick, 1.0, 0)
    def agg(d):
        pc, fc = chosen(d)
        keep = min(1.0, fc.sum() / fv.sum()) if fv.sum() > 0 else 1.0
        pm, fm = thinned_baseline(d, size, tick, p, keep, NSEED_P)
        return np.array([pc.sum(), fc.sum() * nt, pm.sum(), fm.sum() * nt])
    real = agg(data)
    pl_ = []
    for h in SHIFTS_H:
        d = list(data); d[4] = np.roll(data[4], h * 3600); pl_.append(agg(tuple(d)))
    spread_bps = float(np.median((data[1] - data[0]) / mid * 1e4))
    return real, np.mean(pl_, axis=0), float(mid.mean()), spread_bps

def ratio(a):   # a: (days, 4) summed -> diff bps/fill
    return a[0] / a[1] * 1e4 - a[2] / a[3] * 1e4

def boot_days(parts, B=B, seed=0):
    """parts: list over assets of (days,4) arrays with the same number of days (aligned); joint day resampling."""
    rng = np.random.default_rng(seed); n = parts[0].shape[0]
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        tot = sum(P[idx].sum(axis=0) for P in parts)
        out[b] = ratio(tot)
    return np.percentile(out, [2.5, 97.5])

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])

def report(R, new_syms):
    con = Console(record=True, width=170)
    t = Table(title="Robustness of the v3 effect: diff = chosen minus fill-matched thinned vanilla, bps per fill (model C, fee 0, half-spread 0)")
    for c in ("asset", "days", "tick bp", "spread bp", "REAL diff [day-bootstrap 95% CI]", "PLACEBO diff (signal shifted) [CI]"):
        t.add_column(c, justify="left" if c == "asset" else "right")
    rows = []
    for s, D in R.items():
        re_, pl_ = D["real"], D["placebo"]
        rd, pd = ratio(re_.sum(axis=0)), ratio(pl_.sum(axis=0))
        rlo, rhi = boot_days([re_], seed=1); plo, phi = boot_days([pl_], seed=2)
        t.add_row(s, str(len(re_)), f"{D['tick'] / D['mid'] * 1e4:.3f}", f"{D['spread']:.3f}",
                  f"{rd:+.3f} [{rlo:+.3f}, {rhi:+.3f}]", f"{pd:+.3f} [{plo:+.3f}, {phi:+.3f}]")
        rows.append((s, rd, D["tick"] / D["mid"] * 1e4, D["spread"]))
    con.print(t)
    # pooled over the four v3 assets: same calendar days resampled jointly
    re_p = [R[s]["real"] for s in new_syms]; pl_p = [R[s]["placebo"] for s in new_syms]
    n = min(len(x) for x in re_p)
    re_p = [x[:n] for x in re_p]; pl_p = [x[:n] for x in pl_p]
    d_r = ratio(sum(x.sum(axis=0) for x in re_p)); lo, hi = boot_days(re_p, seed=3)
    d_p = ratio(sum(x.sum(axis=0) for x in pl_p)); plo, phi = boot_days(pl_p, seed=4)
    con.print(f"POOLED, {len(new_syms)} v3 assets, same days resampled jointly across assets (keeps cross-asset dependence):")
    con.print(f"  REAL    diff {d_r:+.3f} bps/fill  95% CI [{lo:+.3f}, {hi:+.3f}]   (hour-block CI in the confirmatory run: [+0.099, +0.116])")
    con.print(f"  PLACEBO diff {d_p:+.3f} bps/fill  95% CI [{plo:+.3f}, {phi:+.3f}]")
    con.print(f"  Real effect survives placebo and cluster CI: {bool(lo > 0 and phi < d_r)}")
    sel = [r for r in rows]
    con.print(f"Cross-asset (n={len(sel)}, descriptive): Spearman(diff, tick bp) = {spearman([r[1] for r in sel], [r[2] for r in sel]):+.2f}; "
              f"Spearman(diff, spread bp) = {spearman([r[1] for r in sel], [r[3] for r in sel]):+.2f}")
    return con

def make_synth(td):
    rng = np.random.default_rng(1)
    os.makedirs(f"{td}/data/parquet"); os.makedirs(f"{td}/configs"); shutil.copy(CFG_SIM, f"{td}/configs/sim.yaml")
    n, tick = 3 * 3600 + 5, 0.01
    for day in ("2025-01-01", "2025-02-01", "2025-03-01"):
        mid = 150 + np.cumsum(rng.normal(0, 0.01, n))
        bb = np.floor(mid / tick) * tick; ba = bb + tick * (1 + (rng.random(n) < 0.2))
        cols = {"timestamp": (1_700_000_000 + np.arange(n)) * 1_000_000}
        for i in range(5):
            cols[f"bids[{i}].price"] = bb - i * tick; cols[f"asks[{i}].price"] = ba + i * tick
            cols[f"bids[{i}].amount"] = rng.uniform(1, 50, n); cols[f"asks[{i}].amount"] = rng.uniform(1, 50, n)
        pl.DataFrame(cols).write_parquet(f"{td}/data/parquet/TESTUSDT-book5-{day}.parquet")
        m = n * 3; tt = np.sort(rng.integers(0, n, m)); sell = rng.random(m) < 0.5
        pl.DataFrame({"timestamp": (1_700_000_000 + tt) * 1_000_000 + 500, "side": np.where(sell, "sell", "buy"),
                      "price": np.where(sell, bb[tt], ba[tt]), "amount": rng.uniform(0.1, 5, m)}).write_parquet(f"{td}/data/parquet/TESTUSDT-trades-{day}.parquet")

def collect(sym, days, p):
    tick = detect_tick(days[0], sym)
    real, plc, mids, spr = [], [], [], []
    for d in days:
        r, q, m, s = day_agg(sym, d, tick, p); real.append(r); plc.append(q); mids.append(m); spr.append(s); print(sym, d, flush=True)
    return {"real": np.array(real), "placebo": np.array(plc), "tick": tick, "mid": float(np.mean(mids)), "spread": float(np.median(spr))}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--selftest":
        with tempfile.TemporaryDirectory() as td:
            make_synth(td); cwd = os.getcwd(); os.chdir(td)
            try:
                p = yaml.safe_load(open("configs/sim.yaml"))
                R = {"TESTUSDT": collect("TESTUSDT", ["2025-01-01", "2025-02-01", "2025-03-01"], p)}
                print(report(R, ["TESTUSDT"]).export_text()); print("selftest OK")
            finally:
                os.chdir(cwd)
        sys.exit(0)
    if os.path.exists(OUT): sys.exit(f"already done: {OUT}")
    cfg = yaml.safe_load(open("configs/v3_days.yaml")); p = yaml.safe_load(open(CFG_SIM))
    from src.common import days_for
    R = {}
    for s in SIX:
        days = [str(d) for d in cfg[s]] if s in SYMS else days_for(s)
        R[s] = collect(s, days, p)
    con = report(R, list(SYMS))
    open(OUT, "w").write(con.export_text())
