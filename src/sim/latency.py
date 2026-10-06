import glob, yaml
import numpy as np, polars as pl
from numba import njit
from rich.console import Console
from rich.table import Table

L = 5

def load(day, dt_ms, sym="BTCUSDT"):
    u = dt_ms * 1000
    b = pl.read_parquet(f"data/parquet/{sym}-book5-{day}.parquet").sort("timestamp")
    bid = pl.sum_horizontal([pl.col(f"bids[{i}].amount") for i in range(L)])
    ask = pl.sum_horizontal([pl.col(f"asks[{i}].amount") for i in range(L)])
    b = b.select(k=pl.col("timestamp") // u,
                 mid=(pl.col("asks[0].price") + pl.col("bids[0].price")) / 2,
                 dimb=(bid - ask) / (bid + ask))
    bars = b.group_by("k", maintain_order=True).agg(pl.col("mid").last(), pl.col("dimb").last())
    tr = pl.read_parquet(f"data/parquet/{sym}-trades-{day}.parquet").group_by(
        (pl.col("timestamp") // u).alias("k")).agg(lo=pl.col("price").min(), hi=pl.col("price").max())
    grid = pl.DataFrame({"k": np.arange(bars["k"].min(), bars["k"].max() + 1)})
    d = (grid.join(bars, on="k", how="left").join(tr, on="k", how="left")
         .with_columns(pl.col("mid").forward_fill(),
                       pl.col("dimb").fill_nan(0.0).forward_fill().fill_null(0.0),
                       pl.col("lo").fill_null(float("inf")), pl.col("hi").fill_null(float("-inf"))))
    return tuple(d[c].to_numpy().astype(np.float64) for c in ("mid", "lo", "hi", "dimb"))

@njit(cache=True)
def sim(mid, lo, hi, sig, hs_bps, widen, skew, inv_skew, fee, size, qmax, lag):
    n, q, cash, fills = len(mid), 0, 0.0, 0
    for i in range(1 + lag, n):
        m, s = mid[i - 1 - lag], sig[i - 1 - lag]
        r = m * (1 - inv_skew * 1e-4 * q / qmax + skew * 1e-4 * s)
        hs = m * (hs_bps + widen * abs(s)) * 1e-4
        bid, ask = min(r - hs, m), max(r + hs, m)
        if lo[i] < bid and q < qmax:
            cash -= bid * size * (1 + fee); q += 1; fills += 1
        if hi[i] > ask and q > -qmax:
            cash += ask * size * (1 - fee); q -= 1; fills += 1
    return cash + q * size * mid[n - 1], fills

if __name__ == "__main__":
    p = yaml.safe_load(open("configs/sim.yaml"))
    days = sorted(f[-18:-8] for f in glob.glob("data/parquet/BTCUSDT-book5-*.parquet"))[3:]
    cfgs = {"vanilla": (0.0, 0.0), "skew4/widen1": (4.0, 1.0), "skew4/widen2": (4.0, 2.0)}
    t = Table(title="Latency breakeven, held-out days, half-spread 1bp, fee 0, passive-only")
    for c in ("bar_ms", "extra_lag_ms", "config"):
        t.add_column(c, justify="right" if c != "config" else "left")
    for d in days:
        t.add_column(d[5:], justify="right")
    for c in ("total USDT", "fills"):
        t.add_column(c, justify="right")
    for dt in (1000, 100):
        data = {d: load(d, dt) for d in days}
        for lag_ms in (0, 100, 200, 500, 1000, 2000):
            if lag_ms % dt:
                continue
            for name, (sk, wd) in cfgs.items():
                res = [sim(*data[d], 1.0, wd, sk, float(p["inv_skew_bps"]), 0.0, float(p["clip"]), int(p["qmax"]), lag_ms // dt) for d in days]
                pnl = [r[0] for r in res]
                t.add_row(str(dt), str(lag_ms), name, *[f"{v:+.1f}" for v in pnl], f"{sum(pnl):+.1f}", str(sum(r[1] for r in res)))
    Console().print(t)
