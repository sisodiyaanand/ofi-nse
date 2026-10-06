import glob, yaml
import numpy as np
from rich.console import Console
from rich.table import Table
from src.sim.as_sim import load, simulate

p0 = yaml.safe_load(open("configs/sim.yaml"))
days = sorted(f.split("BTCUSDT-")[1][:10] for f in glob.glob("data/parquet/bars-BTCUSDT-*.parquet"))
test = days[3:]
data = {d: load(d) for d in test}
avg = np.mean([data[d][0].mean() for d in test])
t = Table(title="Held-out days, half-spread 1bp, passive-only quotes: fee and latency stress")
for c in ("config", "fee_bps", "lag_s"):
    t.add_column(c)
for d in test:
    t.add_column(d[5:], justify="right")
for c in ("total USDT", "fills", "bps/fill"):
    t.add_column(c, justify="right")
for name, (sk, wd) in {"vanilla": (0, 0), "skew4/widen1": (4, 1), "skew4/widen2": (4, 2)}.items():
    for fee in (0.0, 0.5, 2.0):
        for lag in (0, 1, 2):
            p = dict(p0, sig_skew_bps=sk, widen_bps=wd, lag=lag)
            res = [simulate(*data[d][:3], data[d][3], 1.0, fee, p) for d in test]
            pnl = np.array([r[0] for r in res]); fl = sum(r[1] for r in res)
            t.add_row(name, str(fee), str(lag), *[f"{v:+.1f}" for v in pnl], f"{pnl.sum():+.1f}", str(fl),
                      f"{pnl.sum() / (fl * p['clip'] * avg) * 1e4:+.2f}" if fl else "n/a")
Console().print(t)
