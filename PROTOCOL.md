# Frozen confirmatory protocol (written before looking at the confirmatory data)

Exploratory set (already analysed, not used for claims): BTCUSDT 13 days 2025-10-01..2026-10-01, ETHUSDT 7 days 2026-04-01..2026-10-01.
Confirmatory set (never analysed before this commit): BTCUSDT 2024-10-01..2025-09-01 (12 first-of-month days), ETHUSDT 2025-04-01..2026-03-01 (12 days).

Fixed rules, no retuning on confirmatory data:
- Fill model: tick-aware with queue position (src/sim/core.py), fee 0, 1s refresh, quote refresh lag 1s, clip and qmax from configs/sim.yaml.
- Config (skew, widen) is selected on the exploratory train days only (same GRID as src/sim/regime.py) at half-spread 0bp, then applied unchanged.
- Primary metric: PnL of chosen config minus PnL of vanilla thinned to the same fill count (mean over 20 random seeds), summed over confirmatory days, per asset.
- Uncertainty: block bootstrap over hours, block length 3, 5000 resamples, 95% CI.
- Secondary (reported, not used for claims): bps per fill, low/high vol split with threshold = exploratory-train median of hourly realized vol.
- Primary claim is supported only if the CI lower bound is above 0 for both assets. Otherwise report the result as not confirmed.
