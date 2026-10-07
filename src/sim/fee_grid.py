import yaml
from rich.console import Console
from rich.table import Table
from src.common import SYM, days_for, split
from src.sim.latency import load, sim

p = yaml.safe_load(open("configs/sim.yaml"))
days = split(days_for())[1]
data = {d: load(d, 100, SYM) for d in days}
cfgs = {"vanilla": (0.0, 0.0), "skew4/widen0": (4.0, 0.0), "skew4/widen1": (4.0, 1.0)}
LAGS = (0, 100, 200, 500)

def total(sk, wd, fee_bps, lag_ms):
    return sum(sim(*data[d], 1.0, wd, sk, float(p["inv_skew_bps"]), fee_bps * 1e-4,
                   float(p["clip"]), int(p["qmax"]), lag_ms // 100)[0] for d in days)

def breakeven(sk, wd, lag_ms):
    if total(sk, wd, 0.0, lag_ms) <= 0:
        return None
    lo, hi = 0.0, 5.0
    for _ in range(25):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if total(sk, wd, mid, lag_ms) > 0 else (lo, mid)
    return lo

t = Table(title=f"{SYM}: total PnL (USDT), held-out, 100ms bars, half-spread 1bp, passive-only")
t.add_column("config"); t.add_column("fee_bps", justify="right")
for l in LAGS:
    t.add_column(f"lag {l}ms", justify="right")
for name, (sk, wd) in cfgs.items():
    for fee in (0.0, 0.1, 0.2, 0.3, 0.5):
        t.add_row(name, str(fee), *[f"{total(sk, wd, fee, l):+.0f}" for l in LAGS])
b = Table(title=f"{SYM}: breakeven maker fee (bps), total PnL hits zero")
b.add_column("config")
for l in LAGS:
    b.add_column(f"lag {l}ms", justify="right")
for name, (sk, wd) in cfgs.items():
    b.add_row(name, *[("none" if (v := breakeven(sk, wd, l)) is None else f"{v:.3f}") for l in LAGS])
Console().print(t); Console().print(b)
