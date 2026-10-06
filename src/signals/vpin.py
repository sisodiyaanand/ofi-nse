import polars as pl

def vpin(day, sym="BTCUSDT", window=50, n_buckets=1000):
    tr = pl.read_parquet(f"data/parquet/{sym}-trades-{day}.parquet").sort("timestamp")
    V = tr["amount"].sum() / n_buckets
    tr = tr.with_columns(bk=(pl.col("amount").cum_sum() / V).floor(),
                         buy=pl.when(pl.col("side") == "buy").then(pl.col("amount")).otherwise(0.0),
                         sell=pl.when(pl.col("side") == "sell").then(pl.col("amount")).otherwise(0.0))
    b = tr.group_by("bk", maintain_order=True).agg(
        pl.col("buy").sum(), pl.col("sell").sum(), pl.col("timestamp").last())
    b = b.with_columns(((pl.col("buy") - pl.col("sell")).abs() / (pl.col("buy") + pl.col("sell")))
                       .rolling_mean(window).alias("vpin"), sec=pl.col("timestamp") // 1_000_000)
    return b.select("sec", "vpin").drop_nulls().sort("sec")
