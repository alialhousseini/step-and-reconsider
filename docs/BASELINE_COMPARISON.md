# Baseline beam-search comparison (feasibility%)

Full greedy -> bs8 -> bs16 feasibility sweep for every model row (N=256 test instances). Baselines (SL, GD-SIL) were greedy-only before; this completes the beam-search columns. Ours' bs8 is taken from the Phase-3 sweep.

| Arch | Method | greedy | bs 8 | bs 16 | Ours @ TaSaR k=8,s=4 / k=16,s=4 |
|---|---|---|---|---|---|
| BQ | SL | 27.7 | 69.5 | 79.3 |  |
| BQ | GD-SIL | 62.9 | 90.6 | 94.5 |  |
| BQ | Ours | 53.1 | 87.5 | **93.8** | 99.2 / 100.0 |
| LEHD | SL | 58.2 | 89.1 | 93.4 |  |
| LEHD | GD-SIL (diverged) | 18.4 | 52.3 | 66.4 |  |
| LEHD | Ours | 55.5 | 91.0 | **96.9** | 98.4 / 99.6 |

**Reading it.**
- All models scale strongly with beam search (e.g. BQ SL 27.7 -> 79.3), so beam search alone is powerful here.
- At **plain beam search**, GD-SIL and Ours are comparable (BQ bs16: GD-SIL 94.5 vs Ours 93.8). **Ours' distinct advantage appears at the TaSaR budgets** (BQ 99.2/100.0, LEHD 98.4/99.6) -- the decoder Ours was trained for.
- **LEHD GD-SIL diverged** (18.4 -> 66.4), staying below both LEHD SL and LEHD Ours at every budget -- reinforcing that un-truncated (p=1.0) SIL is unstable for LEHD while TaSaR's Top-p is stable.

**Caveat.** Baselines were evaluated at greedy/bs8/bs16 only (deterministic beam search), not at TaSaR budgets; a fully budget-matched Ours-vs-GD-SIL comparison at TaSaR decoding is future work.