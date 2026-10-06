import glob, yaml
import numpy as np, polars as pl
from rich.console import Console
from rich.table import Table

def load(day, sym="BTCUSDT"):
    bars = pl.read_parquet(f"data/parquet/bars-{sym}-{day}.parquet").sort("sec")
    tr = pl.read_parquet(f"data/parquet/{sym}-trades-{day}.parquet").with_columns(sec=pl.col("timestamp") // 1_000_000)
    hl = tr.group_by("sec").agg(lo=pl.col("price").min(), hi=pl.col("price").max())
    z = (pl.col("ofi_ml") / pl.col("ofi_ml").rolling_std(600)).fill_nan(0.0).fill_null(0.0).tanh()
    d = bars.join(hl, on="sec", how="left").with_columns(
        pl.col("lo").fill_null(float("inf")), pl.col("hi").fill_null(float("-inf")),
        pl.col("dimb").fill_nan(0.0).fill_null(0.0), z.alias("z"))
    return d["mid"].to_numpy(), d["lo"].to_numpy(), d["hi"].to_numpy(), d["dimb"].to_numpy(), d["z"].to_numpy()

def simulate(mid, lo, hi, sig, hs_bps, fee_bps, p):
    n, q, cash, fills = len(mid), 0, 0.0, 0
    size, qmax, fee = p["clip"], p["qmax"], fee_bps * 1e-4
    pnl, maxq = np.zeros(n), 0
    for i in range(1 + p.get("lag", 0), n):
        m = mid[i - 1 - p.get("lag", 0)]  # quotes use information up to second i-1 only
        r = m * (1 - p["inv_skew_bps"] * 1e-4 * q / qmax + p["sig_skew_bps"] * 1e-4 * sig[i - 1 - p.get("lag", 0)])
        hs = m * (hs_bps + p.get("widen_bps", 0.0) * abs(sig[i - 1 - p.get("lag", 0)])) * 1e-4
        bid, ask = min(r - hs, m), max(r + hs, m)
        if lo[i] < bid and q < qmax:           # fill only if trades went THROUGH our quote
            cash -= bid * size * (1 + fee); q += 1; fills += 1
        if hi[i] > ask and q > -qmax:
            cash += ask * size * (1 - fee); q -= 1; fills += 1
        pnl[i] = cash + q * size * mid[i]
        maxq = max(maxq, abs(q))
    pm = np.diff(pnl[::60])
    sharpe = pm.mean() / pm.std() * np.sqrt(1440 * 365) if pm.std() > 0 else 0.0
    return pnl[-1], fills, maxq, sharpe, fills * size * mid.mean()

if __name__ == "__main__":
    p = yaml.safe_load(open("configs/sim.yaml"))
    days = sorted(f.split("BTCUSDT-")[1][:10] for f in glob.glob("data/parquet/bars-BTCUSDT-*.parquet"))
    data = {d: load(d) for d in days}
    t = Table(title=f"A-S style simulation, {len(days)} days, trade-through fills, no queue model")
    for c in ("spread_bps", "fee_bps", "variant", "fills", "net PnL USDT", "bps/fill", "max|inv|", "mean Sharpe", "days>0"):
        t.add_column(c, justify="right")
    for hs in p["half_spreads_bps"]:
        for fee in p["maker_fees_bps"]:
            for name in ("vanilla", "dimb", "dimb+ofi"):
                res = []
                for d in days:
                    mid, lo, hi, dimb, z = data[d]
                    sig = {"vanilla": np.zeros_like(dimb), "dimb": dimb, "dimb+ofi": 0.5 * dimb + 0.5 * z}[name]
                    res.append(simulate(mid, lo, hi, sig, hs, fee, p))
                pnl, fl, mq, sh, notl = map(np.array, zip(*res))
                bps = pnl.sum() / (fl.sum() * p["clip"] * np.mean([data[d][0].mean() for d in days])) * 1e4 if fl.sum() else 0
                t.add_row(f"{hs}", f"{fee}", name, f"{int(fl.sum())}", f"{pnl.sum():+.2f}", f"{bps:+.2f}",
                          f"{int(mq.max())}", f"{sh.mean():+.1f}", f"{int((pnl > 0).sum())}/{len(days)}")
    Console().print(t)
