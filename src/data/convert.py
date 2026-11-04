"""CSV.gz -> parquet, raw Tardis columns kept (timestamp in microseconds).
Dtypes are fixed explicitly (not inferred) so every day has an identical schema.
Usage: python -m src.data.convert BTCUSDT [day ...]   (no days = all downloaded)
"""
import sys, os, glob, gzip, shutil
import polars as pl

STR = {"exchange", "symbol", "side"}
INT = {"timestamp", "local_timestamp", "id"}

def conv(sym, short, day):
    src = f"data/raw/{sym}-{short}-{day}.csv.gz"
    dst = f"data/parquet/{sym}-{short}-{day}.parquet"
    if not os.path.exists(src) or os.path.exists(dst):
        return
    tmp = src[:-3]
    with gzip.open(src, "rb") as f, open(tmp, "wb") as g:
        shutil.copyfileobj(f, g)
    cols = pl.scan_csv(tmp).collect_schema().names()
    ov = {c: (pl.String if c in STR else pl.Int64 if c in INT else pl.Float64) for c in cols}
    pl.scan_csv(tmp, schema_overrides=ov).sink_parquet(dst)
    os.remove(tmp)
    print("ok", dst)

if __name__ == "__main__":
    os.makedirs("data/parquet", exist_ok=True)
    sym = sys.argv[1]
    days = sys.argv[2:] or sorted({f.split(f"{sym}-book5-")[1][:10] for f in glob.glob(f"data/raw/{sym}-book5-*.csv.gz")})
    for day in days:
        for short in ("book5", "trades"):
            conv(sym, short, day)
