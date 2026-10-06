import sys
import polars as pl
from rich.console import Console
from rich.table import Table

L, HORIZONS = 5, (1, 5, 30)

def ofi_exprs():
    out = []
    for i in range(L):
        bp, bq = pl.col(f"bids[{i}].price"), pl.col(f"bids[{i}].amount")
        ap, aq = pl.col(f"asks[{i}].price"), pl.col(f"asks[{i}].amount")
        f = pl.Float64
        e = ((bp >= bp.shift(1)).cast(f) * bq - (bp <= bp.shift(1)).cast(f) * bq.shift(1)
             - (ap <= ap.shift(1)).cast(f) * aq + (ap >= ap.shift(1)).cast(f) * aq.shift(1))
        out.append(e.alias(f"ofi{i+1}"))
    return out

def build_bars(day, sym="BTCUSDT"):
    b = pl.read_parquet(f"data/parquet/{sym}-book5-{day}.parquet").sort("timestamp")
    b = b.with_columns(ofi_exprs()).with_columns(
        sec=pl.col("timestamp") // 1_000_000,
        mid=(pl.col("asks[0].price") + pl.col("bids[0].price")) / 2,
        spread=pl.col("asks[0].price") - pl.col("bids[0].price"),
        bid_depth=pl.sum_horizontal([pl.col(f"bids[{i}].amount") for i in range(L)]),
        ask_depth=pl.sum_horizontal([pl.col(f"asks[{i}].amount") for i in range(L)]),
    ).with_columns(dimb=(pl.col("bid_depth") - pl.col("ask_depth")) / (pl.col("bid_depth") + pl.col("ask_depth")))
    bars = b.group_by("sec", maintain_order=True).agg(
        [pl.col(f"ofi{i+1}").sum() for i in range(L)]
        + [pl.col("mid").last(), pl.col("spread").mean(), pl.col("dimb").last()]
    ).with_columns(ofi_ml=pl.sum_horizontal([pl.col(f"ofi{i+1}") for i in range(L)]))
    bars = bars.with_columns([(pl.col("mid").shift(-h) / pl.col("mid") - 1).alias(f"ret{h}s") for h in HORIZONS])
    bars.write_parquet(f"data/parquet/bars-{sym}-{day}.parquet")
    return bars.drop_nulls()

if __name__ == "__main__":
    day = sys.argv[1]
    bars = build_bars(day)
    c = Console()
    t = Table(title=f"Spearman IC vs forward mid return ({day}, n={bars.height} bars)")
    t.add_column("signal")
    for h in HORIZONS:
        t.add_column(f"{h}s", justify="right")
    for s in ("ofi1", "ofi2", "ofi3", "ofi4", "ofi5", "ofi_ml", "dimb"):
        row = [s] + [f"{bars.select(pl.corr(s, f'ret{h}s', method='spearman')).item():+.4f}" for h in HORIZONS]
        t.add_row(*row)
    c.print(t)
