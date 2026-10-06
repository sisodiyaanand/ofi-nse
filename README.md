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

## Reproduce
Data download and parquet conversion: see `src/data/`. Then `make all`.
Note: in the train sweep, widen 0/1/2 at skew 4 were tied (+0.24 to +0.26); widen 1 and 2 were evaluated on held-out days, not picked from test results.

## Latency breakeven (100ms bars, held-out days, fee 0, passive-only, no queue model)
Sanity: 1000ms bars reproduce the earlier 1s results (+139.3 vs +139.7 USDT).
skew4/widen1 total PnL by extra lag: 0ms +625, 100ms +405, 200ms +243, 500ms -141, 1000ms -587 USDT.
skew4/widen2: +412, +232, +98, -215, -578. Vanilla is negative at every lag.
Edge disappears between 200 and 500 ms. Rough breakeven maker fee at zero lag is about 0.3 bps (estimate, not simulated).
Fills at 100ms are 2-3x higher than at 1s; without a queue model these results are an upper bound.
