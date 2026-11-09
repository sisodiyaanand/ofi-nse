"""Protocol v3 (see PROTOCOL-v3.md): out-of-asset confirmation on 4 new Binance perpetuals + fill-model gap table.
  python -m src.sim.v3 --check      validate configs/v3_days.yaml and data presence, no simulation
  python -m src.sim.v3 --selftest   synthetic-data smoke test (no real data needed)
  python -m src.sim.v3              the run (refuses if results/confirmatory_v3.txt exists or any asset has < n_min days)
Fixed a priori: chosen config (skew, widen) = (1, 2) from v1, quoted half-spread 0 bp, fee 0, 1 s refresh, lag 1 s,
fixed notional of about 100 USDT per fill, 20 thinning seeds, block bootstrap (block 3, 5000 resamples).
"""
import os, sys, tempfile
import yaml
import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from src.sim.queue import load, sim_q
from src.sim.v2 import sim_h

CFG_SIM = "configs/sim.yaml"
SYMS = ("SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT")
CHOSEN = (1.0, 2.0)          # (skew, widen), v1 choice, never retuned
NOTIONAL = 100.0             # USDT per fill (size = NOTIONAL / mid)
NSEED, BOOT, BLK = 20, 5000, 3
HS_GRID = (0.0, 1.0, 2.0)
MODES = {0: "A naive", 1: "B tick-aware", 2: "C tick+queue"}
OUT = "results/confirmatory_v3.txt"

def detect_tick(day, sym):
    b = pl.read_parquet(f"data/parquet/{sym}-book5-{day}.parquet", columns=["bids[0].price", "asks[0].price"])
    px = np.unique(np.round(np.concatenate([b["bids[0].price"].to_numpy(), b["asks[0].price"].to_numpy()]), 9))
    d = np.diff(px)
    d = d[d > 0]
    tick = float(f"{d.min():.10g}")
    frac = np.abs(px / tick - np.round(px / tick)).max()
    if frac > 1e-3:
        raise ValueError(f"{sym} {day}: prices not on a grid of {tick} (max frac {frac})")
    return tick

def thin_run(data, size, tick, p, keep, seed):
    return sim_h(*data, 0.0, CHOSEN[1], CHOSEN[0], float(p["inv_skew_bps"]), 0.0,
                 size, int(p["qmax"]), tick, keep, seed)

def van_run(data, size, tick, p, keep, seed):
    return sim_h(*data, 0.0, 0.0, 0.0, float(p["inv_skew_bps"]), 0.0, size, int(p["qmax"]), tick, keep, seed)

def per_day(sym, day, tick, p):
    data = load(day, sym)
    mid = (data[0] + data[1]) / 2
    size = NOTIONAL / float(mid.mean())
    nt = size * float(mid.mean())
    pc, fc = sim_h(*data, 0.0, CHOSEN[1], CHOSEN[0], float(p["inv_skew_bps"]), 0.0, size, int(p["qmax"]), tick, 1.0, 0)
    pv, fv = van_run(data, size, tick, p, 1.0, 0)
    mk = min(1.0, fc.sum() / fv.sum()) if fv.sum() > 0 else 1.0
    mm = [van_run(data, size, tick, p, mk, s) for s in range(NSEED)]
    pm = np.mean([x[0] for x in mm], axis=0); fm = np.mean([x[1] for x in mm], axis=0)
    gap = {}
    for hs in HS_GRID:
        for md in MODES:
            r = sim_q(*data, md, hs, 0.0, 0.0, float(p["inv_skew_bps"]), 0.0, size, int(p["qmax"]), 0, tick)
            gap[(hs, md)] = (r[0], r[1], r[1] * nt)       # pnl, fills, traded notional
    spread_bps = float(np.median((data[1] - data[0]) / mid * 1e4))
    return dict(pc=pc, fc=fc, pv=pv, fv=fv, pm=pm, fm=fm, nt=nt, gap=gap, spread_bps=spread_bps, mid=float(mid.mean()))

def boot_pooled(parts, key_b, B=BOOT, blk=BLK, seed=0):
    """parts: list per asset of dict arrays pc, fc, p<b>, f<b> (hourly) and nt. Returns 95% CI of pooled diff in bps/fill."""
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        nc = dc = nb = db = 0.0
        for P in parts:
            n = len(P["pc"]); k = int(np.ceil(n / blk))
            idx = (rng.integers(0, n, k)[:, None] + np.arange(blk)).ravel()[:n] % n
            w = P["ntv"][idx]
            nc += P["pc"][idx].sum(); dc += (P["fc"][idx] * w).sum()
            nb += P["p" + key_b][idx].sum(); db += (P["f" + key_b][idx] * w).sum()
        out[b] = nc / dc * 1e4 - nb / db * 1e4
    return np.percentile(out, [2.5, 97.5])

def assemble(sym, days, p):
    tick = detect_tick(days[0], sym)
    D = [per_day(sym, d, tick, p) for d in days]
    cat = lambda k: np.concatenate([x[k] for x in D])
    P = {k: cat(k) for k in ("pc", "fc", "pv", "fv", "pm", "fm")}
    P["ntv"] = np.concatenate([np.full(len(x["pc"]), x["nt"]) for x in D])
    P["tick"] = tick; P["days"] = len(days)
    P["mid"] = float(np.mean([x["mid"] for x in D])); P["spread_bps"] = float(np.median([x["spread_bps"] for x in D]))
    P["tick_bps"] = tick / P["mid"] * 1e4
    P["gap"] = {k: tuple(sum(x["gap"][k][i] for x in D) for i in range(3)) for k in D[0]["gap"]}
    return P

def bps(P, pk, fk):
    return P[pk].sum() / (P[fk] * P["ntv"]).sum() * 1e4

def build_tables(res, title):
    con = Console(record=True, width=170)
    t0 = Table(title="Assets (tick detected from data by rule)")
    for c in ("asset", "days", "tick", "tick bp", "median quoted spread bp", "mean mid"):
        t0.add_column(c, justify="left" if c == "asset" else "right")
    for s, P in res.items():
        t0.add_row(s, str(P["days"]), f"{P['tick']:.8g}", f"{P['tick_bps']:.3f}", f"{P['spread_bps']:.3f}", f"{P['mid']:.5g}")
    con.print(t0)
    t1 = Table(title=title)
    for c in ("asset", "baseline", "fills chosen", "fills baseline", "bps/fill chosen", "bps/fill baseline", "diff bps/fill [95% CI]"):
        t1.add_column(c, justify="left" if c in ("asset", "baseline") else "right")
    verdict, sup = {}, {}
    for key, label in (("m", "PRIMARY: thinned, fill-matched"), ("v", "secondary: vanilla")):
        for s, P in res.items():
            kb = "m" if key == "m" else "v"
            pb, fb = bps(P, "p" + kb, "f" + kb), None
            lo, hi = boot_pooled([P], kb, seed=1)
            d = bps(P, "pc", "fc") - pb
            t1.add_row(s, label, str(int(P["fc"].sum())), str(int(P["f" + kb].sum())), f"{bps(P, 'pc', 'fc'):+.3f}", f"{pb:+.3f}",
                       f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            if key == "m": sup[s] = bool(lo > 0)
        parts = list(res.values())
        lo, hi = boot_pooled(parts, "m" if key == "m" else "v", seed=2)
        nc = sum(P["pc"].sum() for P in parts); dc = sum((P["fc"] * P["ntv"]).sum() for P in parts)
        nb = sum(P["p" + kb].sum() for P in parts); db = sum((P["f" + kb] * P["ntv"]).sum() for P in parts)
        d = nc / dc * 1e4 - nb / db * 1e4
        t1.add_row("POOLED", label, str(int(sum(P["fc"].sum() for P in parts))), str(int(sum(P["f" + kb].sum() for P in parts))),
                   f"{nc / dc * 1e4:+.3f}", f"{nb / db * 1e4:+.3f}", f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        if key == "m": verdict["pooled_lo"] = float(lo)
    con.print(t1)
    t2 = Table(title="Fill-model gap (vanilla quoter, fee 0, lag 0): fills and bps/fill per fill model and quoted half-spread")
    for c in ("asset", "hs bp", "model", "fills", "bps/fill", "fills vs C"):
        t2.add_column(c, justify="left" if c in ("asset", "model") else "right")
    rows = []
    for s, P in res.items():
        for hs in HS_GRID:
            fC = P["gap"][(hs, 2)][1]
            for md, name in MODES.items():
                pn, fl, nt_ = P["gap"][(hs, md)]
                t2.add_row(s, f"{hs:g}", name, str(int(fl)), f"{pn / nt_ * 1e4:+.3f}", f"{fl / fC:.2f}x")
            rows.append((s, hs, P["tick_bps"], P["spread_bps"], P["gap"][(hs, 0)][1] / fC))
    con.print(t2)
    return con, sup, verdict, rows

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])

def check(cfg, verbose=True):
    ok = True
    for s in SYMS:
        days = [str(d) for d in cfg.get(s, [])]
        miss = [d for d in days if not all(os.path.exists(f"data/parquet/{s}-{k}-{d}.parquet") for k in ("book5", "trades"))]
        if len(set(days)) != len(days) or miss: ok = False
        if verbose: print(f"{s}: {len(days)} days (n_min {cfg['n_min']}); data missing: {miss or 'none'}")
    return ok

def selftest():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(f"{td}/data/parquet"); os.makedirs(f"{td}/configs")
        import shutil; shutil.copy(CFG_SIM, f"{td}/configs/sim.yaml")
        cwd = os.getcwd(); os.chdir(td)
        try:
            n, tick = 3 * 3600 + 5, 0.01
            for day in ("2025-01-01", "2025-02-01"):
                mid = 150 + np.cumsum(rng.normal(0, 0.01, n))
                bb = np.floor(mid / tick) * tick; ba = bb + tick * (1 + (rng.random(n) < 0.2))
                ts = (1_700_000_000 + np.arange(n)) * 1_000_000
                cols = {"timestamp": ts}
                for i in range(5):
                    cols[f"bids[{i}].price"] = bb - i * tick; cols[f"asks[{i}].price"] = ba + i * tick
                    cols[f"bids[{i}].amount"] = rng.uniform(1, 50, n); cols[f"asks[{i}].amount"] = rng.uniform(1, 50, n)
                pl.DataFrame(cols).write_parquet(f"data/parquet/TESTUSDT-book5-{day}.parquet")
                m = n * 3; tt = rng.integers(0, n, m); tt.sort()
                sell = rng.random(m) < 0.5
                px = np.where(sell, bb[tt], ba[tt])
                pl.DataFrame({"timestamp": (1_700_000_000 + tt) * 1_000_000 + 500, "side": np.where(sell, "sell", "buy"),
                              "price": px, "amount": rng.uniform(0.1, 5, m)}).write_parquet(f"data/parquet/TESTUSDT-trades-{day}.parquet")
            p = yaml.safe_load(open("configs/sim.yaml"))
            assert abs(detect_tick("2025-01-01", "TESTUSDT") - tick) < 1e-12, "tick detection"
            P = assemble("TESTUSDT", ["2025-01-01", "2025-02-01"], p)
            assert P["fc"].sum() > 0 and P["fv"].sum() > 0, "no fills"
            con, sup, verdict, rows = build_tables({"TESTUSDT": P}, "SELFTEST")
            print(con.export_text()[:1500])
            print("selftest OK; spearman demo:", spearman(np.arange(5.0), np.arange(5.0)[::-1]))
        finally:
            os.chdir(cwd)

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--selftest":
        selftest(); sys.exit(0)
    cfg = yaml.safe_load(open("configs/v3_days.yaml"))
    ok = check(cfg)
    if arg == "--check": sys.exit(0 if ok else 1)
    if not ok: sys.exit("v3 config invalid or data missing")
    if os.path.exists(OUT): sys.exit(f"v3 run already done: {OUT}. Not re-running.")
    if any(len(cfg[s]) < int(cfg["n_min"]) for s in SYMS): sys.exit("Fewer days than n_min: NOT evaluated.")
    p = yaml.safe_load(open(CFG_SIM))
    res = {s: assemble(s, [str(d) for d in cfg[s]], p) for s in SYMS}
    con, sup, verdict, rows = build_tables(res, "CONFIRMATORY v3: bps per fill, queue model, fee 0, half-spread 0bp. Positive diff = chosen loses less per fill")
    v = "SUPPORTED" if verdict["pooled_lo"] > 0 else "NOT CONFIRMED"
    con.print(f"H1 (pooled CI lower bound of chosen minus fill-matched thinned vanilla > 0): {v}; per-asset lower>0: {sup}")
    r0 = [r for r in rows if r[1] == 0.0]
    rho = spearman([r[2] for r in r0], [r[4] for r in r0])
    con.print(f"H2 (descriptive, n={len(r0)} assets, two-sided): Spearman(tick size in bp, naive/queue fill ratio at hs 0) = {rho:+.2f}")
    open(OUT, "w").write(con.export_text())
