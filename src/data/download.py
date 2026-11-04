"""Download Tardis free first-of-month days (no API key) for Binance USDT-M futures.
Usage: python -m src.data.download BTCUSDT 2026-10-01 [more days...]
       python -m src.data.download BTCUSDT          # default: 2024-10-01 .. 2026-10-01
"""
import sys, os, urllib.request, urllib.error

BASE = "https://datasets.tardis.dev/v1/binance-futures"
KINDS = {"book5": "book_snapshot_5", "trades": "trades"}

def default_days():
    out = []
    y, m = 2024, 10
    while (y, m) <= (2026, 10):
        out.append(f"{y}-{m:02d}-01")
        m += 1
        if m == 13: y, m = y + 1, 1
    return out

def fetch(sym, day, short, kind):
    y, m, d = day.split("-")
    url = f"{BASE}/{kind}/{y}/{m}/{d}/{sym}.csv.gz"
    dst = f"data/raw/{sym}-{short}-{day}.csv.gz"
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print("skip", dst); return
    print("get ", url)
    try:
        urllib.request.urlretrieve(url, dst + ".part")
        os.replace(dst + ".part", dst)
    except urllib.error.HTTPError as e:
        print("FAIL", e.code, url)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    sym = sys.argv[1]
    days = sys.argv[2:] or default_days()
    for day in days:
        for short, kind in KINDS.items():
            fetch(sym, day, short, kind)
