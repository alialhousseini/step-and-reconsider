# VNE Phase 2 — Results (TaSaR self-improved learning)

Paper-aligned reporting for VNE, mirroring TaSaR (Pirnay & Grimm, ECAI-2024,
arXiv:2407.17206) Table 1. Methodology verified by a 4-role paper-alignment review.

> **Status:** numbers pending the genuine-TaSaR (Top-p = 0.9) retrain + baseline evals.
> Skeleton + methodology are final; cells fill from `scripts/vne_eval_table.py` output.

## Method rows (per architecture group, paper's SL → GD SIL → Ours structure)

| Row | Source |
|---|---|
| `BQ SL, greedy` | Phase 1 supervised checkpoint `phase1_bq128_42476` |
| `GD SIL (BQ)` | SIL run with **Top-p = 1.0** (un-truncated WOR-SBS = Gumbeldore baseline), `2026-06-24--09-58-33` |
| `Ours (BQ, TaSaR)` | SIL run with **Top-p = 0.9** (genuine TaSaR), `2026-06-24--15-04-20` |
| (LEHD group) | same three rows, deferred |

Each trained model is then evaluated at inference budgets: greedy, BS k∈{8,16}, TaSaR (k,s)∈{(8,2),(8,4),(16,4)}.

## Results table (skeleton)

```
ckpt=<...>  N=<...>  |F*|=<...>  top_p=0.9

| Method            | Feas%* | Gap%(F*)† | ⟨g(k,s)⟩‡ | Time set(s)§ | ms/inst |
|-------------------|--------|-----------|-----------|--------------|---------|
| greedy            |        |           |           |              |         |
| bs k=8            |        |           |           |              |         |
| bs k=16           |        |           |           |              |         |
| TaSaR k=8 s=2     |        |           |           |              |         |
| TaSaR k=8 s=4     |        |           |           |              |         |
| TaSaR k=16 s=4    |        |           |           |              |         |
```

## Caption / methodology (the documented deviations from the paper)

- `*` **Feas%** = fraction of the full test set fully embedded. No analogue in TaSaR Table 1
  (routing/JSSP are feasible by construction); it is the headline VNE metric. **Bold best.**
- `†` **Gap%(F\*)** = mean relative gap to the HiGHS ILP optimum, averaged over **F\*** =
  instances feasible under **every** listed method (fixed denominator ⇒ orderable). The paper
  averages over the entire (all-feasible) set, so the two gap numbers are not directly
  comparable. Instances with `ilp_obj == 0` are excluded (counted separately). **Bold best.**
- `‡` **⟨g(k,s)⟩** = mean node transitions per instance (paper Eq. 3):
  `g_BS = k·l`, `g_TaSaR = k·(t·l − (s·t² − s·t)/2)`, `t = ⌈l/s⌉`,
  with `l` = number of virtual-link assignment decisions per instance.
  **Smaller s ⇒ more replan rounds ⇒ larger g** (opposite of a naive k·s).
- `§` Times are single-worker on one **RTX 5060 (8 GB)**; **not** comparable to the paper's
  4×RTX-3090 totals. Per-instance ms also reported for our hardware.
- Reference solver: **HiGHS** ILP optimum (analogue of the paper's Concorde/HGS).
- Gap is **not** bolded across rows with different feas% except on F\*; under the lexicographic
  (feasibility-first) objective, Feas% is the primary ranking.

## Planned auxiliary results (cheap, no new training/labelling)

1. **SIL self-improvement curve** — gap% & feas% vs epoch, from each run's `log.txt`, with the
   Phase-1 seed as a dashed baseline (feas 36.8%, gap 2.08%).
2. **Inference scaling vs measured wall-clock** — two panels (gap%, feas%) vs ms/inst (log x),
   TaSaR points averaged over 3 seeds with std-error.
3. **Feas%–vs–Gap% Pareto scatter** — one point per setting; never collapses gap to one number.

## Known correctness rules (from the alignment review)

- TaSaR ⇒ Top-p < 1. Top-p = 1.0 is **WOR-SBS / Gumbeldore**, not TaSaR (label honestly).
- Gap over feasible-only is **non-orderable** across methods → always report on F\*.
- The completed feas 37%→60% / gap 2.08→2.84 result is the textbook trap: those gaps are over
  different instance subsets; it is **not** a gap regression.
- `best_model.pt` is lex-selected `(1−feas)·1000 + gap` — verify all of greedy/BS/TaSaR for a
  group use the **same** checkpoint.
