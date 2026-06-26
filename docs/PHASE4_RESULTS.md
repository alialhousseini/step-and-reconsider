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

## Why only 1.35× here (not the paper's 2–5×)

ILP labeling cost explodes with substrate size on a single machine: ~10s at 80 nodes,
~21s at 100, and at 120–160 nodes the time-bounded solves routinely time out, which
makes `generate_instance` resample (it rejects non-optimal solves), exploding to
~28 min/instance. 90–110 nodes is the tractable ceiling here (≈106s/instance, no
resamples). The paper's 2–5× generalization needs the **SLURM cluster** (parallel
labeling across many nodes) — a clean follow-up, evaluation-only once labelled.

Tools: `scripts/vne_gen_parallel.py` (Gurobi, multiprocessing, `--threads`/`--workers`);
set `data/vne/vne_gen_135x.pickle`.
