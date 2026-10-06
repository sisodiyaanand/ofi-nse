import sys
import numpy as np, polars as pl
from scipy.stats import rankdata
from rich.console import Console
from rich.table import Table
from src.signals.vpin import vpin

H = (1, 5, 30)

def run(day, sym="BTCUSDT"):
    bars = pl.read_parquet(f"data/parquet/bars-{sym}-{day}.parquet").sort("sec")
    bars = bars.join_asof(vpin(day), on="sec")
    feats = bars.select("sec", "ofi_ml", "dimb", "vpin").with_columns(pl.col("sec") + 1)  # features of second s-1 -> trades in s
    tr = pl.read_parquet(f"data/parquet/{sym}-trades-{day}.parquet").with_columns(sec=pl.col("timestamp") // 1_000_000)
    tr = tr.join(feats, on="sec", how="left")
    for h in H:
        m = bars.select("sec", "mid").with_columns(pl.col("sec") - h).rename({"mid": f"mid{h}"})
        tr = tr.join(m, on="sec", how="left")
        # maker markout in bps: taker buy -> maker sold at price, taker sell -> maker bought
        tr = tr.with_columns((pl.when(pl.col("side") == "buy").then(pl.col("price") - pl.col(f"mid{h}"))
                              .otherwise(pl.col(f"mid{h}") - pl.col("price")) / pl.col("price") * 1e4).alias(f"mk{h}"))
    sgn = pl.when(pl.col("side") == "buy").then(1.0).otherwise(-1.0)
    tr = tr.with_columns((pl.col("ofi_ml") * sgn).alias("ofi_s"), (pl.col("dimb") * sgn).alias("dimb_s")).drop_nulls()
    t = Table(title=f"Maker markout (bps) by toxicity quintile, Q1=safe Q5=toxic ({day}, n={tr.height} trades)")
    for c in ("signal", "h", "Q1", "Q2", "Q3", "Q4", "Q5", "Q5-Q1"):
        t.add_column(c, justify="right" if c not in ("signal",) else "left")
    for s in ("ofi_s", "dimb_s", "vpin"):
        x = tr[s].to_numpy()
        q = np.minimum(((rankdata(x) - 1) / len(x) * 5).astype(int), 4)
        for h in H:
            mk = tr[f"mk{h}"].to_numpy()
            mean = [mk[q == k].mean() for k in range(5)]
            t.add_row(s, f"{h}s", *[f"{v:+.3f}" for v in mean], f"{mean[4] - mean[0]:+.3f}")
    Console().print(t)

if __name__ == "__main__":
    run(sys.argv[1])
