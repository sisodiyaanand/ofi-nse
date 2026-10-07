import os, sys, yaml
import numpy as np
from rich.console import Console
from rich.table import Table
from src.sim.queue import load
from src.sim.core import sim_path, boot_ci

os.makedirs("results", exist_ok=True)
if os.path.exists("results/confirmatory.txt") and not os.environ.get("FORCE"):
    sys.exit("Confirmatory run already done: results/confirmatory.txt. Not re-running.")

p = yaml.safe_load(open("configs/sim.yaml"))
GRID = [(sk, wd) for sk in (0, 1, 2, 4) for wd in (0, 1, 2)]
TICKS = {"BTCUSDT": 0.1, "ETHUSDT": 0.01}
HS, NSEED = 0.0, 20
EXP = {"BTCUSDT": ["2025-10-01", "2025-11-01", "2025-12-01", "2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01",
                   "2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01", "2026-09-01", "2026-10-01"],
       "ETHUSDT": ["2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01", "2026-09-01", "2026-10-01"]}
CONF = {"BTCUSDT": ["2024-10-01", "2024-11-01", "2024-12-01", "2025-01-01", "2025-02-01", "2025-03-01",
                    "2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"],
        "ETHUSDT": ["2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01",
                    "2025-10-01", "2025-11-01", "2025-12-01", "2026-01-01", "2026-02-01", "2026-03-01"]}

def hv(bb, ba):
    lm = np.log((bb + ba) / 2)
    r = np.diff(lm, prepend=lm[0])
    nh = len(lm) // 3600
    return r[: nh * 3600].reshape(nh, 3600).std(axis=1) * 1e4

con = Console(record=True)
t = Table(title="CONFIRMATORY (frozen protocol-v1): queue model, fee 0, half-spread 0bp, 1s refresh")
for c in ("asset", "row", "fills", "USDT", "bps/fill", "chosen - thinned: USDT [95% CI, hourly blocks]"):
    t.add_column(c, justify="left" if c in ("asset", "row") else "right")
verdict = {}
for sym in ("BTCUSDT", "ETHUSDT"):
    tick = TICKS[sym]
    def path(data, g, keep=1.0, seed=0):
        return sim_path(*data, HS, float(g[1]), float(g[0]), float(p["inv_skew_bps"]), 0.0,
                        float(p["clip"]), int(p["qmax"]), tick, keep, seed)
    tr_days = EXP[sym][: round(len(EXP[sym]) * 0.6)]
    ld = {d: load(d, sym) for d in tr_days}
    score = {g: sum(path(ld[d], g)[0].sum() for d in tr_days) for g in GRID}
    best = max(score, key=score.get)
    keep = min(1.0, sum(path(ld[d], best)[1] for d in tr_days) / sum(path(ld[d], (0, 0))[1] for d in tr_days))
    thr = np.median(np.concatenate([hv(ld[d][0], ld[d][1]) for d in tr_days]))
    del ld
    V, C, T, R = [], [], [], []
    fv = fc = ft = nv = nc = nt = 0.0
    for d in CONF[sym]:
        data = load(d, sym)
        avg = (data[0].mean() + data[1].mean()) / 2 * float(p["clip"])
        van, a = path(data, (0, 0)); cho, b = path(data, best)
        ths = [path(data, (0, 0), keep, s) for s in range(NSEED)]
        thn = np.mean([x[0] for x in ths], axis=0); c_ = np.mean([x[1] for x in ths])
        v = hv(data[0], data[1])
        V.append(van[1:]); C.append(cho[1:]); T.append(thn[1:]); R.append(v[:-1] > thr)
        fv += a; fc += b; ft += c_; nv += a * avg; nc += b * avg; nt += c_ * avg
    V, C, T, R = map(np.concatenate, (V, C, T, R))
    diff = C - T
    lo, hi = boot_ci(diff)
    verdict[sym] = lo > 0
    t.add_row(sym, "vanilla", f"{int(fv)}", f"{V.sum():+.1f}", f"{V.sum() / nv * 1e4:+.2f}", "")
    t.add_row(sym, f"chosen {best} (selected on exploratory train)", f"{int(fc)}", f"{C.sum():+.1f}", f"{C.sum() / nc * 1e4:+.2f}", "")
    t.add_row(sym, f"vanilla thinned keep={keep:.2f}, mean of {NSEED} seeds", f"{int(ft)}", f"{T.sum():+.1f}", f"{T.sum() / nt * 1e4:+.2f}", "")
    t.add_row(sym, "PRIMARY: chosen - thinned", "", f"{diff.sum():+.1f}", "", f"[{lo:+.1f}, {hi:+.1f}]")
    for name, m in (("secondary low vol", ~R), ("secondary high vol", R)):
        l2, h2 = boot_ci(diff[m], blk=1)
        t.add_row(sym, f"{name} ({int(m.sum())}h)", "", f"{diff[m].sum():+.1f}", "", f"[{l2:+.1f}, {h2:+.1f}]")
con.print(t)
ok = all(verdict.values())
con.print("PRIMARY CLAIM (CI lower bound > 0 for both assets): " + ("SUPPORTED" if ok else "NOT CONFIRMED") + f"  {verdict}")
open("results/confirmatory.txt", "w").write(con.export_text())
