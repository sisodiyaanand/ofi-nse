"""Protocol v2 confirmatory run (see PROTOCOL-v2.md). Run once, after configs/v2_days.yaml is frozen.
  python -m src.sim.v2 --check     validate configs/v2_days.yaml only, no simulation
  python -m src.sim.v2 --selftest  re-run the v1 days with this code; must match results/posthoc.txt
  python -m src.sim.v2             the confirmatory run (refuses if results/confirmatory_v2.txt exists)
"""
import os, sys
import yaml
import numpy as np
from numba import njit
from rich.console import Console
from rich.table import Table
from src.sim.queue import load

p = yaml.safe_load(open("configs/sim.yaml"))
TICKS = {"BTCUSDT": 0.1, "ETHUSDT": 0.01}
CHOSEN = {"BTCUSDT": (1, 2), "ETHUSDT": (1, 2)}      # fixed in confirm-v1, not retuned
SYMS = ("BTCUSDT", "ETHUSDT")
NSEED = 20
OUT = "results/confirmatory_v2.txt"
V1_CONF = {"BTCUSDT": ["2024-10-01", "2024-11-01", "2024-12-01", "2025-01-01", "2025-02-01", "2025-03-01",
                       "2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"],
           "ETHUSDT": ["2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01",
                       "2025-10-01", "2025-11-01", "2025-12-01", "2026-01-01", "2026-02-01", "2026-03-01"]}

# sim_h, run, ratio_ci: copied verbatim from src/sim/posthoc.py (unchanged)
@njit(cache=True)
def sim_h(bb, ba, bq, aq, sig, tsec, tsell, tp, ta, hs_bps, widen, skew, inv_skew, fee, size, qmax, tick, keep, seed):
    np.random.seed(seed)
    n, nt, q, cash, fills, j = len(bb), len(tsec), 0, 0.0, 0, 0
    bid_px, ask_px, bid_ahead, ask_ahead = -1.0, -1.0, 0.0, 0.0
    bid_on, ask_on = False, False
    tol = 0.5 * tick
    pnl = np.zeros(n // 3600 + 1); fl = np.zeros(n // 3600 + 1)
    last, lastf, h = 0.0, 0, 0
    for i in range(n):
        k = i - 1
        if k >= 0:
            m, s = (bb[k] + ba[k]) / 2.0, sig[k]
            r = m * (1 - inv_skew * 1e-4 * q / qmax + skew * 1e-4 * s)
            hs = m * (hs_bps + widen * abs(s)) * 1e-4
            nb = np.floor(min(r - hs, bb[k]) / tick + 1e-9) * tick
            na = np.ceil(max(r + hs, ba[k]) / tick - 1e-9) * tick
            if (nb != bid_px) or (not bid_on):
                bid_ahead = bq[k] if abs(nb - bb[k]) < tol else 0.0
            if (na != ask_px) or (not ask_on):
                ask_ahead = aq[k] if abs(na - ba[k]) < tol else 0.0
            bid_px, ask_px = nb, na
            act = np.random.random() < keep
            bid_on, ask_on = (q < qmax) and act, (q > -qmax) and act
        while j < nt and tsec[j] == i:
            pr, a = tp[j], ta[j]
            if tsell[j]:
                if bid_on:
                    hit = pr < bid_px - tol
                    if (not hit) and abs(pr - bid_px) < tol:
                        bid_ahead -= a
                        hit = bid_ahead < 0
                    if hit:
                        cash -= bid_px * size * (1 + fee); q += 1; fills += 1; bid_on = False
            else:
                if ask_on:
                    hit = pr > ask_px + tol
                    if (not hit) and abs(pr - ask_px) < tol:
                        ask_ahead -= a
                        hit = ask_ahead < 0
                    if hit:
                        cash += ask_px * size * (1 - fee); q -= 1; fills += 1; ask_on = False
            j += 1
        if (i + 1) % 3600 == 0:
            v = cash + q * size * (bb[i] + ba[i]) / 2.0
            pnl[h] = v - last; fl[h] = fills - lastf; last = v; lastf = fills; h += 1
    return pnl[:h], fl[:h]

def run(data, g, keep=1.0, seed=0, sym="BTCUSDT"):
    return sim_h(*data, 0.0, float(g[1]), float(g[0]), float(p["inv_skew_bps"]), 0.0,
                 float(p["clip"]), int(p["qmax"]), TICKS[sym], keep, seed)

def ratio_ci(num_c, den_c, num_b, den_b, B=5000, blk=3, seed=0):
    rng = np.random.default_rng(seed)
    n = len(num_c); nb = int(np.ceil(n / blk)); out = np.empty(B)
    for b in range(B):
        idx = (rng.integers(0, n, nb)[:, None] + np.arange(blk)).ravel()[:n] % n
        out[b] = num_c[idx].sum() / den_c[idx].sum() * 1e4 - num_b[idx].sum() / den_b[idx].sum() * 1e4
    return np.percentile(out, [2.5, 97.5])

def analysed():
    """All first-of-month days up to 2026-10-01: every day used by v1 or exploratory work."""
    out, y, m = set(), 2024, 10
    while (y, m) <= (2026, 10):
        out.add(f"{y}-{m:02d}-01"); m += 1
        if m == 13: y, m = y + 1, 1
    return out

def evaluate(sym, days):
    H = {k: [] for k in ("pc", "fc", "pv", "fv", "pm", "fm", "nt")}
    for d in days:
        data = load(d, sym)
        avg = (data[0].mean() + data[1].mean()) / 2 * float(p["clip"])
        pc, fc = run(data, CHOSEN[sym], 1.0, 0, sym)
        pv, fv = run(data, (0, 0), 1.0, 0, sym)
        mk = min(1.0, fc.sum() / fv.sum())
        mm = [run(data, (0, 0), mk, s, sym) for s in range(NSEED)]
        for key, arr in (("pc", pc), ("fc", fc), ("pv", pv), ("fv", fv),
                         ("pm", np.mean([x[0] for x in mm], axis=0)), ("fm", np.mean([x[1] for x in mm], axis=0))):
            H[key].append(arr)
        H["nt"].append(np.full(len(pc), avg))
    H = {k: np.concatenate(v) for k, v in H.items()}
    nt = H["nt"]
    bps = lambda p_, f_: H[p_].sum() / (H[f_] * nt).sum() * 1e4
    res = {"fills_c": int(H["fc"].sum()), "bps_c": bps("pc", "fc")}
    for name, pk, fk in (("matched", "pm", "fm"), ("vanilla", "pv", "fv")):
        lo, hi = ratio_ci(H["pc"], H["fc"] * nt, H[pk], H[fk] * nt)
        res[name] = {"fills": int(H[fk].sum()), "bps": bps(pk, fk), "diff": res["bps_c"] - bps(pk, fk), "lo": lo, "hi": hi}
    return res

def table(results, title):
    t = Table(title=title)
    for c in ("asset", "days", "baseline", "fills chosen", "fills baseline", "bps/fill chosen", "bps/fill baseline", "diff bps/fill [95% CI]"):
        t.add_column(c, justify="left" if c in ("asset", "baseline") else "right")
    for sym, (nd, r) in results.items():
        for name, label in (("matched", "PRIMARY: thinned, fill-matched"), ("vanilla", "secondary: vanilla")):
            b = r[name]
            t.add_row(sym, str(nd), label, str(r["fills_c"]), str(b["fills"]), f"{r['bps_c']:+.3f}", f"{b['bps']:+.3f}",
                      f"{b['diff']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]")
    con = Console(record=True); con.print(t)
    return con

def check(cfg, verbose=True):
    n_min, seen, ok = int(cfg["n_min"]), analysed(), True
    for sym in SYMS:
        days = cfg.get(sym, [])
        bad = [d for d in days if d in seen]
        miss = [d for d in days if not all(os.path.exists(f"data/parquet/{sym}-{k}-{d}.parquet") for k in ("book5", "trades"))]
        dup = len(set(days)) != len(days)
        if bad or miss or dup: ok = False
        if verbose:
            print(f"{sym}: {len(days)} days (n_min {n_min}); already analysed: {bad or 'none'}; data missing: {miss or 'none'}; duplicates: {dup}")
    return ok

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--selftest":
        res = {s: (len(V1_CONF[s]), evaluate(s, V1_CONF[s])) for s in SYMS}
        table(res, "SELFTEST on v1 days (expect BTC chosen -0.962 / matched -1.019 / diff +0.058; ETH -0.947 / -1.004 / +0.057)")
        sys.exit(0)
    cfg = yaml.safe_load(open("configs/v2_days.yaml"))
    cfg.update({s: [str(d) for d in cfg[s]] for s in SYMS})
    ok = check(cfg)
    if arg == "--check":
        sys.exit(0 if ok else 1)
    if not ok:
        sys.exit("v2 config invalid, fix configs/v2_days.yaml (see lines above)")
    if os.path.exists(OUT):
        sys.exit(f"Confirmatory v2 run already done: {OUT}. Not re-running.")
    if any(len(cfg[s]) < int(cfg["n_min"]) for s in SYMS):
        sys.exit("Fewer days than n_min for at least one asset: NOT evaluated (no interim peeking, see PROTOCOL-v2.md).")
    res = {s: (len(cfg[s]), evaluate(s, cfg[s])) for s in SYMS}
    con = table(res, "CONFIRMATORY (frozen protocol-v2): bps per fill, queue model, fee 0, half-spread 0bp. Positive diff = chosen loses less per fill")
    sup = {s: bool(res[s][1]["matched"]["lo"] > 0) for s in SYMS}
    verdict = "SUPPORTED" if all(sup.values()) else "NOT CONFIRMED"
    con.print(f"H1 (CI lower bound > 0 for both assets): {verdict}  {sup}")
    open(OUT, "w").write(con.export_text())
