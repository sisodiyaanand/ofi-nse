# Frozen protocol v3: out-of-asset confirmation

Status: v1 and v2 (BTC, ETH) did not confirm a signal effect in both assets (results/confirmatory.txt, results/confirmatory_v2.txt).
v3 tests the same pre-specified hypothesis on four assets never analysed by the authors. Written and committed BEFORE any v3 data is downloaded.

## Assets and days
SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT (Binance USDT-M perpetuals, Tardis book_snapshot_5 and trades).
23 first-of-month days per asset, 2025-01-01 .. 2026-11-01 (configs/v3_days.yaml). n_min = 15 days per asset, otherwise no evaluation.
Tick size is detected from the data by rule (smallest positive gap between distinct best bid/ask prices on the first day; prices must lie on that grid).

## Hypothesis H1 (primary)
At fee 0, quoted half-spread 0 bp, tick-aware queue model, the chosen config (skew, widen) = (1, 2) has a higher per-fill PnL (bps)
than a fill-count-matched randomly thinned vanilla quoter, pooled over the four assets.
Supported only if the lower bound of the 95% CI of the pooled difference is above 0. Per-asset CIs are secondary.
Effect seen post hoc on BTC/ETH: about +0.05 bps/fill. Power: about 92 asset-days.

## Fixed rules, no retuning
- Config (1, 2) and all parameters in configs/sim.yaml unchanged. Fixed notional of 100 USDT per fill (size = 100 / mean mid of the day); inventory limit 10 clips.
- 1 s refresh, quote lag 1 s, fee 0. Baseline thinned to the same fill count as chosen on each day (20 seeds, mean).
- Metric: bps per fill, notional-weighted. Uncertainty: moving-block bootstrap over hours, block 3, 5000 resamples, resampled within each asset.
- Single run; results/confirmatory_v3.txt; the runner refuses to run twice. No interim peeking.

## H2 (descriptive, not a claim)
Fill-model gap table for the vanilla quoter: models A/B/C at half-spread 0, 1, 2 bp (fills, bps/fill, fills relative to C), per asset.
Two-sided Spearman correlation across the four assets between tick size (bp) and the A/C fill ratio at half-spread 0. With n = 4 it is reported as description only.

## Reporting
Failure is reported as plainly as success. BTC and ETH results from v1/v2 remain in the paper as the first two samples.
