# Toxicity-aware passive market making on BTCUSDT perps (research prototype)

## Question
Does a depth-imbalance / order-flow signal let a passive quoter avoid adverse selection, and does the edge survive fees and latency?

## Data
Tardis free sample days (1st of month): 2026-04, 05, 06, 07, 08, 09, 10-01. Binance USDT-M BTCUSDT, 5-level book snapshots + trades.
Train: first 3 days. Held-out: last 4 days.

## Pipeline
`src/signals/ofi.py` multi-level OFI (snapshot approximation), depth imbalance, 1s bars
`src/signals/vpin.py` VPIN | `src/validation.py` purged walk-forward CV (+ leakage test)
`src/markout.py` maker markout by signal quintile | `src/ablation.py` OFI vs depth imbalance
`src/sim/*` passive-quote simulator, skew/widen sweep, fee and latency stress

## Findings (all simulated, 7 days, single asset)
1. Depth imbalance (dimb) has strong OOS IC at 1s (about 0.2 to 0.4). Multi-level OFI adds almost nothing on top of dimb.
2. VPIN shows no consistent link to maker markout.
3. Maker markout is negative in every signal quintile; toxic quintile is worse by about 0.1 to 0.5 bps.
4. Held-out, half-spread 1bp, fee 0, lag 0: skew 4bps / widen 1bp gives +225 USDT (vanilla -173), positive on all 4 days.
5. The edge disappears with 1 second of latency or a 0.5 bps maker fee.

## Limitations
No queue-position model (trade-through fills only); 1s resolution; snapshot-based OFI; 7 non-consecutive days; one asset; fixed clip size; fee tiers not verified; parameters chosen on 3 train days only.
