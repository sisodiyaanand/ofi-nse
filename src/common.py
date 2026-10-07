import glob, os
SYM = os.environ.get("SYM", "BTCUSDT")
TRAIN_FRAC = 0.6   # chronological: first 60% of days train, rest held out

def days_for(sym=None):
    sym = sym or SYM
    return sorted(f.split(sym + "-")[1][:10] for f in glob.glob(f"data/parquet/bars-{sym}-*.parquet"))

def split(days):
    k = int(round(len(days) * TRAIN_FRAC))
    return days[:k], days[k:]
