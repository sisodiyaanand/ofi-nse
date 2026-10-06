import glob
import numpy as np, polars as pl
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.table import Table
from src.validation import purged_walk_forward

SETS = {"dimb+spread": ["dimb", "spread"],
        "+OFI1-5": ["dimb", "spread", "ofi1", "ofi2", "ofi3", "ofi4", "ofi5"]}

def oos_ic(X, y):
    ics = []
    for tr, te in purged_walk_forward(len(y), 5, embargo=30):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        ics.append(spearmanr(m.predict(sc.transform(X[te])), y[te])[0])
    return np.mean(ics)

t = Table(title="OOS IC (purged walk-forward, mean of 5 folds): does OFI add to depth imbalance?")
for c in ("day", "1s base", "1s +OFI", "5s base", "5s +OFI"):
    t.add_column(c, justify="right")
for f in sorted(glob.glob("data/parquet/bars-BTCUSDT-*.parquet")):
    b = pl.read_parquet(f).sort("sec").drop_nulls()
    row = [f.split("BTCUSDT-")[1][:10]]
    for h in (1, 5):
        y = b[f"ret{h}s"].to_numpy()
        for cols in SETS.values():
            row.append(f"{oos_ic(b.select(cols).to_numpy(), y):+.4f}")
    t.add_row(*row)
Console().print(t)
