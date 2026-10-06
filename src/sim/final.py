import glob, yaml
import numpy as np
from rich.console import Console
from rich.table import Table
from src.sim.as_sim import load, simulate

p0 = yaml.safe_load(open("configs/sim.yaml"))
days = sorted(f.split("BTCUSDT-")[1][:10] for f in glob.glob("data/parquet/bars-BTCUSDT-*.parquet"))
test = days[3:]
cfgs = {"vanilla (0,0)": (0, 0), "train-chosen (4,2)": (4, 2)}
t = Table(title="Held-out days, half-spread 1bp, fee 0: per-day net PnL (USDT) and fills")
t.add_column("config")
for d in test:
    t.add_column(d[5:], justify="right")
t.add_column("total", justify="right"); t.add_column("fills", justify="right"); t.add_column("bps/fill", justify="right")
data = {d: load(d) for d in test}
for name, (sk, wd) in cfgs.items():
    p = dict(p0, sig_skew_bps=sk, widen_bps=wd)
    res = [simulate(*data[d][:3], data[d][3], 1.0, 0.0, p) for d in test]
    pnl = np.array([r[0] for r in res]); fl = sum(r[1] for r in res)
    avg = np.mean([data[d][0].mean() for d in test])
    t.add_row(name, *[f"{v:+.2f}" for v in pnl], f"{pnl.sum():+.2f}", str(fl), f"{pnl.sum() / (fl * p['clip'] * avg) * 1e4:+.2f}")
Console().print(t)
