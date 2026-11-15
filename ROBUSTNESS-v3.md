# Robustness plan for the v3 result (written before running; exploratory, not a new confirmatory claim)

Data: the same days already used (v3: 4 assets x 23 days; BTC/ETH: all 35 days). No parameter is changed: config (1, 2), half-spread 0, fee 0, model C, fixed 100 USDT notional.
1. Placebo: the signal is circularly shifted in time by 6, 9 and 12 hours (marginal distribution and autocorrelation kept, alignment with returns destroyed). Baseline is the vanilla quoter thinned to the placebo quoter's fill count (10 seeds). Prediction: diff near 0 or negative.
2. Day-cluster bootstrap (5000): resample whole days; for the pooled estimate the SAME calendar days are resampled jointly across assets, so cross-asset dependence is kept.
3. Cross-asset description for six assets (BTC, ETH, SOL, BNB, XRP, DOGE): diff, tick size in bp, spread in bp, Spearman correlation. Descriptive only (n = 6).
Output: results/robust_v3.txt. Report whatever comes out.
