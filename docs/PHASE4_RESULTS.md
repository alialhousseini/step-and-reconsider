# VNE Phase 4 — Generalization to Larger Substrates

Evaluation-only (no retraining): do the Phase 2 `Ours` models (BQ/LEHD TaSaR p=0.9),
trained on 60–80-node substrates, still produce feasible near-optimal embeddings on
*larger* substrates? Labels generated with **Gurobi** (academic license).

## Table 4 — Generalization

| Model | Scale            | greedy Feas% | bs16 Feas% | bs16 Gap%(g∩bs16) |
|-------|------------------|--------------|------------|-------------------|
| BQ    | 1× (60–80, N=256)   | 53.1 | 93.8  | 2.36 |
| BQ    | 1.35× (90–110, N=16)| 43.8 | 87.5  | 0.63 |
| LEHD  | 1× (60–80, N=256)   | 55.5 | 96.9  | 1.33 |
| LEHD  | 1.35× (90–110, N=16)| 43.8 | **100.0** | 0.18 |

(gap averaged over each row's greedy∩bs16-feasible set.)

## Findings

- **The models generalize to 1.35× larger substrates with strong feasibility under
  search.** Greedy feasibility drops (≈53–56% → 44%, as expected for harder OOD
  instances), but **beam-16 recovers it (BQ 87.5%, LEHD 100%)** — the "reconsider"
  search transfers to larger-than-trained sizes. LEHD generalizes especially well.

## Caveats (important)

- **Gap is not comparable across scales** — 1× labels are *exact* (mip_gap 0); 1.35×
  labels are *≤5% loose* (time-bounded Gurobi to keep labeling tractable). So the
  apparent gap *improvement* at 1.35× is partly a label-tightness artifact, not a real
  quality gain. **Feasibility% is the clean, label-independent metric** and the headline.
- **Small N=16** at 1.35× (estimates are noisy); **modest 1.35× scale.**

## Feasibility vs scale — extended to ~4.6× (no labels, no solver)

Feasibility% needs only a feasibility-possible instance + the model — **no ILP label**.
So we generate unlabeled substrates up to 320 nodes (`make_instance`, instant) and report
feasibility%, sidestepping the labeling wall entirely. Figure:
`artifacts/vne_phase4_feas_vs_scale.png`.

| nodes | ~× trained | BQ greedy | BQ bs16 | LEHD greedy | LEHD bs16 |
|-------|-----------|-----------|---------|-------------|-----------|
| 80    | 1.1×      | 37.5 | 75.0 | 31.2 | 87.5 |
| 120   | 1.7×      | 31.2 | 87.5 | 18.8 | 87.5 |
| 160   | 2.3×      | 43.8 | 87.5 | 31.2 | 87.5 |
| 240   | 3.4×      | 50.0 | 68.8 | 18.8 | 75.0 |
| 320   | 4.6×      | 31.2 | 56.2 | 12.5 | 50.0 |

**Headline:** **beam-16 feasibility holds ~75–87% out to 2.3×**, and is still **~50–56% at
4.6×**, with **no retraining** — search-based decoding generalizes far beyond the trained
60–80-node scale. **Greedy collapses** (12–50%), so the "reconsider" search is what carries
generalization.

Caveats: **N=16 per scale** (feas% in 6.25% steps — read the *trend*, not point values; the
non-monotonic wiggle at 80–160 is small-N noise). These are freshly generated
`make_instance` substrates (greedy-feasibility-accepted), so absolute feas% is **not** directly
comparable to the ILP-labelled Table 4 (e.g. "80n" here is the hard max of the trained range,
not the 60–80 average). No gap reported (unlabelled).

## Gap is the only labeling-bound part

Gap needs an optimum reference (Gurobi), and ILP labeling explodes with size on one machine
(~10s@80n, ~21s@100n; ≥120n the bounded solves time out → `generate_instance` resamples →
~28 min/instance). 90–110 nodes is the tractable ceiling here. So **gap stays at ≤1.35×
(Table 4); feasibility extends to ~4.6× (above).** Larger-scale *gap* would need many parallel
Gurobi licenses (not the GPU cluster, which has no Gurobi).

Tools: `scripts/vne_gen_parallel.py` (Gurobi labeler), `scripts/vne_gen_unlabeled.py`
(no-solver), `scripts/vne_genscale_table.py` (feas-vs-scale table+figure).
