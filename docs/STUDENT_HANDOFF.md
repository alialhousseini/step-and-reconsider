# TaSaR for Cloud Network Flow — results handoff

Hi! Here's a short guide to what we ran and where to find everything.

## What this is
We applied **TaSaR** (Take a Step and Reconsider — a sequence-decoding + self-improving-learning
method) to the **Cloud Network Flow** problem (Llorca/Tulino's Cloud Service Distribution Problem:
place service-function chains on a cloud network with compute+communication resources, minimizing
transport+compute cost). Regime: **embed-all** (every request must be embedded; no admission).
Two policy architectures: **BQ** and **LEHD**.

## Start here
- **`docs/RESULTS_REPORT.html`** — open in a browser. Self-contained report with all methods,
  tables, figures, and interpretation. Best single entry point.

## Where the numbers live
| File | Content |
|---|---|
| `docs/PHASE2_RESULTS.md` | Main results: SIL, per-architecture tables (SL → GD-SIL → Ours × budgets) |
| `docs/PHASE3_RESULTS.md` | Inference scaling vs beam width (Table 3 + figure) |
| `docs/PHASE4_RESULTS.md` | Generalization to larger networks (gap to 1.35×, feasibility to ~4.6×) |
| `docs/BASELINE_COMPARISON.md`, `docs/BS_TABLES.md` | Beam-search comparison of all models (greedy/bs8/bs16) |
| `eval_out/{bq,lehd}/*.json` | Raw per-instance eval outputs (SL/GD-SIL/Ours) |
| `artifacts/*.png` | Figures (SIL curve, beam scaling, feasibility vs scale) |

## How to read the tables
- **Rows per architecture**: `SL` (supervised seed) → `GD SIL` (Gumbeldore self-improved baseline)
  → `Ours` (the genuine TaSaR runs). Higher rows are the method's contribution.
- **Budgets (columns/rows)**: `greedy` → `bs k` (beam search width k) → `TaSaR (k, s)`
  (sample-and-reconsider). More budget = more search = better solutions.
- **Feas%** = fraction of the 256 test instances fully embedded. **This is the primary metric**
  (every instance provably has a solution, so feas<100% means the model's search didn't find it).
- **Gap%(F\*)** = mean cost gap to the ILP optimum, averaged over **F\*** = instances feasible under
  *all* compared rows (a fixed denominator so rows are comparable). Report it *with* feas%.
- **⟨g(k,s)⟩** = compute budget (node transitions). Bigger = more search. Note: *smaller s = more*.

## The one-line story
SIL + TaSaR **solves** the in-distribution set (BQ → 100% feasible, ~0% gap at k=16,s=4); quality
scales monotonically with search budget; and it **generalizes** to much larger networks under beam
search (no retraining). At plain beam search the SIL models are on par — **Ours' distinct edge is
at the TaSaR budgets.**

## Caveats to keep in mind
- **LEHD GD-SIL diverged** during training (feas ~18–66%); it's a *broken baseline*, kept only for
  completeness. (This is itself a finding: un-truncated p=1.0 SIL is unstable for LEHD; TaSaR's
  Top-p is stable.)
- Timings are **single-GPU (RTX 5060)** — not comparable to cluster hardware.
- Small F\* denominators inflate some Phase-2 gap cells; prefer feasibility% + the larger-F\* tables.

## Reproduce
- Evaluate a checkpoint: `EVAL_PLAN="greedy,bs:8,bs:16" python scripts/vne_eval_table.py <ckpt> <bq|lehd> 256`
- Build grouped tables: `python scripts/vne_assemble_bs.py eval_out/bq BQ`
- Everything is on branch **`yorgo-work`**.
