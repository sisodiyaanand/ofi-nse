import glob, yaml
import numpy as np
from rich.console import Console
from rich.table import Table
from src.sim.as_sim import load, simulate

p0 = yaml.safe_load(open("configs/sim.yaml"))
days = sorted(f.split("BTCUSDT-")[1][:10] for f in glob.glob("data/parquet/bars-BTCUSDT-*.parquet"))
data = {d: load(d) for d in days}
train, test = days[:3], days[3:]

def bps(ds, skew, widen, hs=1.0, fee=0.0):
    p = dict(p0, sig_skew_bps=skew, widen_bps=widen)
    pnl = fl = 0.0
    for d in ds:
        mid, lo, hi, dimb, z = data[d]
        r = simulate(mid, lo, hi, dimb, hs, fee, p)
        pnl += r[0]; fl += r[1]
    return pnl / (fl * p["clip"] * np.mean([data[d][0].mean() for d in ds])) * 1e4 if fl else 0.0

t = Table(title=f"dimb signal sweep, half-spread 1bp, fee 0. train={train} test={test}")
for c in ("skew_bps", "widen_bps", "train bps/fill", "test bps/fill"):
    t.add_column(c, justify="right")
for sk in (0, 1, 2, 4):
    for wd in (0, 1, 2):
        t.add_row(str(sk), str(wd), f"{bps(train, sk, wd):+.2f}", f"{bps(test, sk, wd):+.2f}")
Console().print(t)
