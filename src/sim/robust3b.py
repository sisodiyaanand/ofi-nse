"""Follow-up to robust3 (added AFTER seeing results/robust_v3.txt, so exploratory): signal-specific effect = REAL minus PLACEBO.
robust_v3 showed that a time-shifted (placebo) signal reproduces most of the real diff. This script estimates what is left,
with a day-cluster bootstrap that resamples the same days jointly for real and placebo (and jointly across assets), using 6 shifts.
  python -m src.sim.robust3b --selftest | python -m src.sim.robust3b   (writes results/robust_v3b.txt, ~6-10 min)
"""
import os, sys, tempfile
import yaml
import numpy as np
from rich.console import Console
from rich.table import Table
import src.sim.robust3 as R3
from src.sim.robust3 import collect, ratio, SIX, SYMS, CFG_SIM, make_synth, spearman

OUT = "results/robust_v3b.txt"
R3.SHIFTS_H = (3, 6, 9, 12, 15, 18)
B = 5000

def spec(re_parts, pl_parts):
    return ratio(sum(x.sum(axis=0) for x in re_parts)), ratio(sum(x.sum(axis=0) for x in pl_parts)), \
        ratio(sum(x.sum(axis=0) for x in re_parts)) - ratio(sum(x.sum(axis=0) for x in pl_parts))

def boot(re_parts, pl_parts, seed):
    rng = np.random.default_rng(seed); n = re_parts[0].shape[0]; out = np.empty((B, 3))
    for b in range(B):
        idx = rng.integers(0, n, n)
        out[b] = spec([x[idx] for x in re_parts], [x[idx] for x in pl_parts])
    return np.percentile(out, [2.5, 97.5], axis=0)

def report(R, groups):
    con = Console(record=True, width=170)
    t = Table(title="Signal-specific effect = REAL minus PLACEBO (6 time shifts), bps per fill; day-cluster bootstrap 95% CI")
    for c in ("group", "days", "REAL", "PLACEBO", "REAL - PLACEBO [95% CI]", "share of REAL that is signal-specific"):
        t.add_column(c, justify="left" if c == "group" else "right")
    for name, syms in groups.items():
        n = min(len(R[s]["real"]) for s in syms)
        re_ = [R[s]["real"][:n] for s in syms]; pl_ = [R[s]["placebo"][:n] for s in syms]
        r, p, d = spec(re_, pl_); ci = boot(re_, pl_, seed=len(name))
        t.add_row(name, str(n), f"{r:+.3f}", f"{p:+.3f}", f"{d:+.3f} [{ci[0][2]:+.3f}, {ci[1][2]:+.3f}]", f"{d / r:.0%}" if r else "n/a")
    con.print(t)
    return con

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--selftest":
        with tempfile.TemporaryDirectory() as td:
            make_synth(td); cwd = os.getcwd(); os.chdir(td)
            try:
                p = yaml.safe_load(open("configs/sim.yaml"))
                Rr = {"TESTUSDT": collect("TESTUSDT", ["2025-01-01", "2025-02-01", "2025-03-01"], p)}
                print(report(Rr, {"TEST": ["TESTUSDT"]}).export_text()); print("selftest OK")
            finally:
                os.chdir(cwd)
        sys.exit(0)
    if os.path.exists(OUT): sys.exit(f"already done: {OUT}")
    cfg = yaml.safe_load(open("configs/v3_days.yaml")); p = yaml.safe_load(open(CFG_SIM))
    from src.common import days_for
    Rr = {}
    for s in SIX:
        days = [str(d) for d in cfg[s]] if s in SYMS else days_for(s)
        Rr[s] = collect(s, days, p)
    groups = {s: [s] for s in SIX}
    groups["4 v3 assets pooled"] = list(SYMS)
    groups["BTC+ETH pooled"] = ["BTCUSDT", "ETHUSDT"]
    con = report(Rr, groups)
    open(OUT, "w").write(con.export_text())
    np.savez("results/robust_v3b_days.npz", **{f"{s}_{k}": Rr[s][k] for s in SIX for k in ("real", "placebo")})
