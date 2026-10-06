import sys
import numpy as np, polars as pl
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.table import Table
from src.validation import purged_walk_forward

FEATS = ["ofi1", "ofi2", "ofi3", "ofi4", "ofi5", "dimb", "spread"]

def run(day, sym="BTCUSDT"):
    bars = pl.read_parquet(f"data/parquet/bars-{sym}-{day}.parquet").sort("sec").drop_nulls()
    X = bars.select(FEATS).to_numpy()
    t = Table(title=f"Out-of-sample IC, purged walk-forward, Ridge ({day})")
    t.add_column("horizon")
    for k in range(5):
        t.add_column(f"fold{k+1}", justify="right")
    t.add_column("mean", justify="right")
    for h in (1, 5, 30):
        y = bars[f"ret{h}s"].to_numpy()
        ics = []
        for tr, te in purged_walk_forward(len(bars), 5, embargo=30):
            sc = StandardScaler().fit(X[tr])
            m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
            ics.append(spearmanr(m.predict(sc.transform(X[te])), y[te])[0])
        t.add_row(f"{h}s", *[f"{v:+.4f}" for v in ics], f"{np.mean(ics):+.4f}")
    Console().print(t)

if __name__ == "__main__":
    run(sys.argv[1])
