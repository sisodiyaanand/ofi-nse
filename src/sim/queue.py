import yaml
import numpy as np, polars as pl
from numba import njit
from rich.console import Console
from rich.table import Table
from src.common import SYM, days_for, split

p = yaml.safe_load(open("configs/sim.yaml"))
TICK = {"BTCUSDT": 0.1, "ETHUSDT": 0.01}[SYM]
L = 5

def load(day, sym=SYM):
    u = 1_000_000
    b = pl.read_parquet(f"data/parquet/{sym}-book5-{day}.parquet").sort("timestamp")
    bd = pl.sum_horizontal([pl.col(f"bids[{i}].amount") for i in range(L)])
    ad = pl.sum_horizontal([pl.col(f"asks[{i}].amount") for i in range(L)])
    b = b.select(sec=pl.col("timestamp") // u, bb=pl.col("bids[0].price"), ba=pl.col("asks[0].price"),
                 bq=pl.col("bids[0].amount"), aq=pl.col("asks[0].amount"), dimb=(bd - ad) / (bd + ad))
    bars = b.group_by("sec", maintain_order=True).last()
    grid = pl.DataFrame({"sec": np.arange(bars["sec"].min(), bars["sec"].max() + 1)})
    d = grid.join(bars, on="sec", how="left").with_columns(pl.exclude("sec").forward_fill())
    d = d.with_columns(pl.col("dimb").fill_nan(0.0).fill_null(0.0))
    s0 = int(d["sec"][0])
    tr = pl.read_parquet(f"data/parquet/{sym}-trades-{day}.parquet").sort("timestamp").select(
        sec=pl.col("timestamp") // u, sell=(pl.col("side") == "sell"), price=pl.col("price"), amt=pl.col("amount"))
    tr = tr.filter((pl.col("sec") >= s0) & (pl.col("sec") < s0 + d.height))
    f = lambda c: d[c].to_numpy().astype(np.float64)
    return (f("bb"), f("ba"), f("bq"), f("aq"), f("dimb"), (tr["sec"] - s0).to_numpy().astype(np.int64),
            tr["sell"].to_numpy(), tr["price"].to_numpy().astype(np.float64), tr["amt"].to_numpy().astype(np.float64))

@njit(cache=True)
def sim_q(bb, ba, bq, aq, sig, tsec, tsell, tp, ta, mode, hs_bps, widen, skew, inv_skew, fee, size, qmax, lag, tick):
    n, nt, q, cash, fills, j = len(bb), len(tsec), 0, 0.0, 0, 0
    bid_px, ask_px, bid_ahead, ask_ahead = -1.0, -1.0, 0.0, 0.0
    bid_on, ask_on = False, False
    tol = 0.5 * tick if mode > 0 else 0.0
    for i in range(n):
        k = i - 1 - lag
        if k >= 0:
            m, s = (bb[k] + ba[k]) / 2.0, sig[k]
            r = m * (1 - inv_skew * 1e-4 * q / qmax + skew * 1e-4 * s)
            hs = m * (hs_bps + widen * abs(s)) * 1e-4
            if mode == 0:
                nb, na = min(r - hs, m), max(r + hs, m)
            else:
                nb = np.floor(min(r - hs, bb[k]) / tick + 1e-9) * tick
                na = np.ceil(max(r + hs, ba[k]) / tick - 1e-9) * tick
            if (nb != bid_px) or (not bid_on):
                bid_ahead = bq[k] if abs(nb - bb[k]) < 0.5 * tick else 0.0
            if (na != ask_px) or (not ask_on):
                ask_ahead = aq[k] if abs(na - ba[k]) < 0.5 * tick else 0.0
            bid_px, ask_px = nb, na
            bid_on, ask_on = q < qmax, q > -qmax
        while j < nt and tsec[j] == i:
            pr, a = tp[j], ta[j]
            if tsell[j]:
                if bid_on:
                    hit = pr < bid_px - tol
                    if (not hit) and mode == 2 and abs(pr - bid_px) < tol:
                        bid_ahead -= a
                        hit = bid_ahead < 0
                    if hit:
                        cash -= bid_px * size * (1 + fee); q += 1; fills += 1; bid_on = False
            else:
                if ask_on:
                    hit = pr > ask_px + tol
                    if (not hit) and mode == 2 and abs(pr - ask_px) < tol:
                        ask_ahead -= a
                        hit = ask_ahead < 0
                    if hit:
                        cash += ask_px * size * (1 - fee); q -= 1; fills += 1; ask_on = False
            j += 1
    return cash + q * size * (bb[n - 1] + ba[n - 1]) / 2.0, fills

if __name__ == "__main__":
    test = split(days_for())[1]
    data = {d: load(d) for d in test}
    cfgs = {"vanilla": (0.0, 0.0), "skew4/widen0": (4.0, 0.0), "skew4/widen1": (4.0, 1.0), "skew2/widen0": (2.0, 0.0)}
    modes = {0: "A naive (quote at mid)", 1: "B tick-aware (best bid/ask)", 2: "C + queue position"}
    t = Table(title=f"{SYM}: held-out {len(test)} days, 1s refresh, half-spread 1bp, fee 0. Does the edge survive realism?")
    for c in ("fill model", "config", "USDT lag 0s", "fills lag 0s", "USDT lag 1s"):
        t.add_column(c, justify="left" if c in ("fill model", "config") else "right")
    for md, mname in modes.items():
        for name, (sk, wd) in cfgs.items():
            def run(lag):
                r = [sim_q(*data[d], md, 1.0, wd, sk, float(p["inv_skew_bps"]), 0.0, float(p["clip"]), int(p["qmax"]), lag, TICK) for d in test]
                return sum(x[0] for x in r), sum(x[1] for x in r)
            (a, fa), (b, _) = run(0), run(1)
            t.add_row(mname, name, f"{a:+.1f}", str(fa), f"{b:+.1f}")
    Console().print(t)
