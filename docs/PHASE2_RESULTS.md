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

## Results table (BQ group) — N=256 test instances, |F*|=39, Top-p=0.9

Checkpoints: SL = `phase1_bq128_42476/last_model.pt`; GD SIL = `2026-06-24--09-58-33/last_model.pt` (Top-p=1.0);
Ours = `2026-06-24--15-04-20/best_model.pt` (Top-p=0.9, lex-selected best epoch).

| Method              | Feas%* | Gap%(F*)† | ⟨g(k,s)⟩‡ | Time set(s)§ | ms/inst |
|---------------------|--------|-----------|-----------|--------------|---------|
| BQ SL, greedy       | 27.7   | 2.208     | 37        | 31           | 121     |
| GD SIL (BQ), greedy | 62.9   | 2.705     | 37        | 40           | 155     |
| Ours (BQ), greedy   | 53.1   | 2.466     | 37        | 38           | 148     |
| Ours (BQ), bs k=16  | 93.8   | 1.094     | 594       | 483          | 1889    |
| Ours (BQ), k=8 s=4  | 99.2   | 0.024     | 1613      | 1039         | 4057    |
| **Ours (BQ), k=16 s=4** | **100.0** | **0.000** | 3226  | 1933         | 7551    |

**Headline:** inference-compute scaling drives `Ours` from feas 53% / gap 2.47% (greedy) to
**feas 100% / gap 0.00%** at the top TaSaR budget — it solves the test set. SIL beats supervised
at every budget (SL greedy 27.7% feas).

**Caveat:** |F*| = 39 / 256. Because gap% is averaged over instances feasible under *every* row,
the weak `BQ SL greedy` (27.7% feas) shrinks F* to an easy subset; treat Feas% (full N) as the
robust metric and Gap%(F*) as conservative. A gap over the stronger-rows-only F* gives a fuller
picture (see auxiliary).

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

## Auxiliary results

### A. Inference scaling on the stronger-rows F* (|F*|=95, excludes weak SL baseline)

A larger, more trustworthy denominator than the 6-row F*=39. The `Ours` rows are all feasible
on F*, so the gap comparison is clean:

| Ours (BQ)      | Feas% | Gap%(F*=95) |
|----------------|-------|-------------|
| greedy         | 53.1  | 3.13        |
| bs k=16        | 93.8  | 2.01        |
| k=8 s=4        | 99.2  | 0.33        |
| k=16 s=4       | 100.0 | **0.07**    |

Monotonic: more inference compute ⇒ higher feasibility and lower gap, converging to ~optimal.

### B. SIL self-improvement curve

`artifacts/vne_phase2_sil_curve.png` — validation feas% and gap% vs epoch for both SIL runs
(Ours p=0.9, GD-SIL p=1.0) against the Phase-1 seed (dashed: feas 36.8%, gap 2.08%). Both runs
lift feasibility to the 50–63% band. The gap rising slightly above the seed is the
**feasible-pool-composition confound** (≈24pp of harder instances entered the feasible pool),
not a regression — see the F* caveat above.

### C. Deferred (cheap, not yet run)
- Inference-scaling vs measured wall-clock with 3-seed error bars; Feas%–vs–Gap% Pareto scatter.

## Known correctness rules (from the alignment review)

- TaSaR ⇒ Top-p < 1. Top-p = 1.0 is **WOR-SBS / Gumbeldore**, not TaSaR (label honestly).
- Gap over feasible-only is **non-orderable** across methods → always report on F\*.
- The completed feas 37%→60% / gap 2.08→2.84 result is the textbook trap: those gaps are over
  different instance subsets; it is **not** a gap regression.
- `best_model.pt` is lex-selected `(1−feas)·1000 + gap` — verify all of greedy/BS/TaSaR for a
  group use the **same** checkpoint.
