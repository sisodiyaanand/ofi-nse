# Frozen confirmatory protocol v2 (DRAFT, not yet frozen)

Status: v1 primary claim (PnL chosen - thinned, CI lower bound > 0 for both assets) was NOT CONFIRMED.
v2 tests one pre-specified hypothesis, derived post hoc from v1 (results/posthoc.txt), on days never analysed before.

## Hypothesis (H1)
At fee 0, half-spread 0bp, queue fill model, the chosen config (skew, widen) = (1, 2) has a higher
per-fill PnL (bps) than a fill-count-matched randomly thinned vanilla baseline, for both BTCUSDT and ETHUSDT.
Effect size seen post hoc in v1: about +0.05 bps/fill. This is a hypothesis, not a result.

## Data
Unseen set U: only days that appear in no prior analysis (not in the 25 days used by v1 or exploratory work).
U is written to configs/v2_days.yaml and committed BEFORE any simulation is run on those days.
Known candidate already downloaded: 2026-11-01 (bars built, no simulation run).

## Fixed rules, no retuning
- Config chosen on exploratory data only: (skew, widen) = (1, 2) from v1. Parameters from configs/sim.yaml, unchanged.
- Fill model: tick-aware with queue position, fee 0, 1s refresh, quote lag 1s.
- Baseline: vanilla thinned to the SAME fill count as chosen on each test day (20 seeds, mean). Fill-matched on the test days themselves (not on train).
- Metric: bps per fill, chosen minus thinned baseline, per asset, over hourly windows that match for both sides.
- Uncertainty: moving-block bootstrap over hours, block 3, 5000 resamples, 95% CI.
- H1 is supported only if the CI lower bound is above 0 for BOTH assets. Otherwise: not confirmed.
- Minimum size: no claim is reported until |U| >= N_min days per asset. N_min is set in configs/v2_days.yaml at freeze time.
  (Power note: about 10 days if the true effect equals the post hoc estimate, about 40 days if it is half as large.)
- No interim peeking: results are computed once, after the last day of U is in.

## Secondary (reported, not used for claims)
Low/high vol split, bps per fill vs unthinned vanilla, maker fee 0.5 bps and real fee-tier sensitivity.

## Reporting
All outputs saved to results/confirmatory_v2.txt. Failure is reported as plainly as success.
