import yaml
import numpy as np
from numba import njit
from rich.console import Console
from rich.table import Table
from src.sim.queue import load

p = yaml.safe_load(open("configs/sim.yaml"))
TICKS = {"BTCUSDT": 0.1, "ETHUSDT": 0.01}
CHOSEN = {"BTCUSDT": (1, 2), "ETHUSDT": (1, 2)}      # as selected in confirm-v1
KEEP_TRAIN = {"BTCUSDT": 0.38, "ETHUSDT": 0.44}      # as printed in confirm-v1
CONF = {"BTCUSDT": ["2024-10-01", "2024-11-01", "2024-12-01", "2025-01-01", "2025-02-01", "2025-03-01",
                    "2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"],
        "ETHUSDT": ["2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01",
                    "2025-10-01", "2025-11-01", "2025-12-01", "2026-01-01", "2026-02-01", "2026-03-01"]}
NSEED = 20

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

con = Console(record=True)
t = Table(title="POST-HOC (not part of protocol-v1): bps per fill on confirmatory days, queue model, fee 0, half-spread 0bp. Positive diff = chosen loses less per fill")
for c in ("asset", "baseline", "fills chosen", "fills baseline", "bps/fill chosen", "bps/fill baseline", "diff bps/fill [95% CI]"):
    t.add_column(c, justify="left" if c in ("asset", "baseline") else "right")
for sym in ("BTCUSDT", "ETHUSDT"):
    H = {k: [] for k in ("pc", "fc", "pv", "fv", "pt", "ft", "pm", "fm", "nt")}
    sums = {"c": 0.0, "v": 0.0}
    cache = []
    for d in CONF[sym]:
        data = load(d, sym)
        avg = (data[0].mean() + data[1].mean()) / 2 * float(p["clip"])
        pc, fc = run(data, CHOSEN[sym], 1.0, 0, sym)
        pv, fv = run(data, (0, 0), 1.0, 0, sym)
        mk = min(1.0, fc.sum() / fv.sum())
        tt = [run(data, (0, 0), KEEP_TRAIN[sym], s, sym) for s in range(NSEED)]
        mm = [run(data, (0, 0), mk, s, sym) for s in range(NSEED)]
        for key, arr in (("pc", pc), ("fc", fc), ("pv", pv), ("fv", fv),
                         ("pt", np.mean([x[0] for x in tt], axis=0)), ("ft", np.mean([x[1] for x in tt], axis=0)),
                         ("pm", np.mean([x[0] for x in mm], axis=0)), ("fm", np.mean([x[1] for x in mm], axis=0))):
            H[key].append(arr)
        H["nt"].append(np.full(len(pc), avg))
    H = {k: np.concatenate(v) for k, v in H.items()}
    nt = H["nt"]
    bps = lambda p_, f_: H[p_].sum() / (H[f_] * nt).sum() * 1e4
    for name, pk, fk in (("vanilla", "pv", "fv"), ("thinned, train keep", "pt", "ft"), ("thinned, fill-matched on conf days", "pm", "fm")):
        lo, hi = ratio_ci(H["pc"], H["fc"] * nt, H[pk], H[fk] * nt)
        d_ = bps("pc", "fc") - bps(pk, fk)
        t.add_row(sym, name, f"{int(H['fc'].sum())}", f"{int(H[fk].sum())}", f"{bps('pc', 'fc'):+.3f}", f"{bps(pk, fk):+.3f}", f"{d_:+.3f} [{lo:+.3f}, {hi:+.3f}]")
con.print(t)
open("results/posthoc.txt", "w").write(con.export_text())
