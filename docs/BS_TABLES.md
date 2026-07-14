# Beam-search comparison tables (bs8 / bs16)

All three methods (**SL, GD-SIL, Ours**) evaluated under **deterministic beam search** at matched
budgets (beam width 8 and 16), per architecture group. N = 256 test instances. Source JSONs:
`eval_out/bq/{eval_sl,eval_gdsil,eval_tasar}.json` and `eval_out/lehd/{...}.json`
(produced by `scripts/vne_eval_table.py` with `EVAL_PLAN="bs:8,bs:16"`).

Gap% is averaged over the group's **global F\*** — the instances feasible under *every* row at
*both* budgets — so gaps are comparable across rows. Feasibility% is over the full N.

### BQ — beam-search comparison  (N=256, |F*|=150)
| Method | bs8 Feas% | bs8 Gap%(F*) | bs16 Feas% | bs16 Gap%(F*) |
|--------|-----------|--------------|------------|---------------|
| SL     | 69.5 | 3.38 | 79.3 | 2.95 |
| GD-SIL | 90.6 | 2.01 | 94.5 | 1.62 |
| Ours   | 87.5 | 2.67 | 93.8 | 1.97 |

### LEHD — beam-search comparison  (N=256, |F*|=119)
| Method | bs8 Feas% | bs8 Gap%(F*) | bs16 Feas% | bs16 Gap%(F*) |
|--------|-----------|--------------|------------|---------------|
| SL                 | 89.1 | 2.97 | 93.4 | 2.32 |
| GD-SIL *(diverged)*| 52.3 | 1.41 | 66.4 | 1.21 |
| Ours               | 91.0 | 1.77 | 96.9 | 1.43 |

## How to read these

- **Matched-budget, deterministic beam search.** Unlike the main Phase-2 table (which evaluates
  `Ours` at its TaSaR budgets), here *every* row uses plain beam search at the same k, so it is a
  clean like-for-like comparison of the trained policies.
- **BQ:** SIL (both GD-SIL and Ours) hugely beats the supervised seed; at plain beam search
  **GD-SIL and Ours are on par** (bs16: 94.5 vs 93.8 feas, 1.62 vs 1.97 gap). Ours' distinct
  advantage is at the **TaSaR budgets** (99.2 / 100.0% feas — see the main Phase-2 table), the
  decoder it was trained for.
- **LEHD:** **Ours is best** (bs16: 96.9% feas, 1.43 gap), above SL; **GD-SIL diverged** in
  training (bs16 only 66.4%), so it also shrinks the LEHD F\* denominator.
- **Gaps here are more robust** than the Phase-2 headline (|F\*| = 150 / 119 vs 39 / 26), because
  the bs8/bs16 rows are all fairly feasible, giving a larger shared denominator.

## Reproduce

```bash
mkdir -p eval_out/bq eval_out/lehd
# per model (example): EVAL_PLAN="bs:8,bs:16" EVAL_OUT=eval_out/bq/eval_sl.json \
#   python scripts/vne_eval_table.py <ckpt> bq 256
python scripts/vne_assemble_bs.py eval_out/bq  BQ
python scripts/vne_assemble_bs.py eval_out/lehd LEHD
```
