# VNE Phase 3 — Multi-beam Evaluation

Search quality vs **beam width** (deterministic beam search) for the two Phase 2
`Ours` (TaSaR p=0.9) models — BQ and LEHD. No new models, no supervised baselines.
N=256 test instances. beam=1 and beam=16 reused from the Phase 2 eval; {4,8,32,64}
computed here. Figure: `artifacts/vne_phase3_beam_scaling.png`.

Gap% is averaged over each model's **all-beam-feasible set** (instances feasible at
*every* beam width for that model — BQ 116, LEHD 141), so the gap-vs-beam curve has a
fixed denominator. Feas% is over the full N=256.

## Table 3 — Multi-beam evaluation

**BQ Ours** (all-beam-feasible set = 116)

| beam | Feas% | Gap% | ms/inst |
|------|-------|------|---------|
| 1    | 53.1  | 3.227 | 148    |
| 4    | 80.5  | 3.298 | 460    |
| 8    | 87.5  | 3.149 | 882    |
| 16   | 93.8  | 2.233 | 1889   |
| 32   | 95.7  | 2.203 | 3489   |
| 64   | 99.2  | 1.751 | 6899   |

**LEHD Ours** (all-beam-feasible set = 141)

| beam | Feas% | Gap% | ms/inst |
|------|-------|------|---------|
| 1    | 55.5  | 3.782 | 158    |
| 4    | 83.2  | 2.552 | 542    |
| 8    | 91.0  | 1.831 | 1070   |
| 16   | 96.9  | 1.335 | 2024   |
| 32   | 99.2  | 0.809 | 3987   |
| 64   | 99.6  | 0.553 | 8000   |

## Findings

- **Wider search keeps improving VNE solutions** — the paper's headline holds. Feasibility
  rises monotonically with beam width (BQ 53→99%, LEHD 56→99.6%) and gap falls
  (BQ 3.23→1.75%, LEHD 3.78→0.55%). beam=1 is clearly **not** as good as beam=64, so
  SBS/TaSaR produces useful diverse alternatives for VNE — the "reconsider" mechanism works.
- **LEHD scales better than BQ** on gap (0.55% vs 1.75% at beam=64), with a cleaner monotonic
  decrease; BQ's gap is flat for small beams then drops.
- **TaSaR > plain beam search at matched budget:** Phase 2 TaSaR(k=16,s=4) reached gap ~0.07–0.32%,
  below the deterministic beam-64 gaps here — consistent with the paper's central claim that
  TaSaR's sample-and-reconsider beats deterministic beam search.
- Cost scales ~linearly with beam width (ms/inst roughly doubles per width doubling), giving a
  clean cost–quality Pareto curve.

## Status

Phase 3 complete. Remaining: **Phase 4 — Generalization** to larger substrates (120–400 nodes),
which needs new HiGHS ILP labeling at those sizes (evaluation-only, no retraining).
