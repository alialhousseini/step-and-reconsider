# VNE Baseline Matrix

> Generated 2026-06-13. Maps the TaSaR paper's comparison tables onto VNE equivalents.

## Paper Comparison Tables (reference)

**Table 1 (TSP/CVRP)** compared: AM+BS(1024), MDAM+BS(50), POMO(×8), SGBS, LEHD RL+SIL, Gumbeldore SIL (BQ/LEHD), BQ SL, LEHD SL → against TaSaR SIL.

**Table 2 (JSSP)** compared: L2D, ScheduleNet, L2S(500/5000), SPN, GD SIL → against TaSaR SIL.

The key design: **same architecture, different training paradigm / decoding algorithm**. This isolates the contribution of the method.

---

## Baselines Runnable with Zero New Code (config changes only)

All use the same LEHD architecture. Only `learning_type` and `search_type` change.

| # | Method | learning_type | search_type | What it measures |
|---|--------|--------------|-------------|-----------------|
| 1 | **ILP optimal (ceiling)** | — | — | HiGHS MILP exact optimum |
| 2 | **Supervised** | `supervised` | — | Train on ILP labels. Current plateau ~−23 |
| 3 | **Greedy policy** | eval only | `beam_search` k=1 | Pure policy, no search |
| 4 | **Deterministic BS** | eval only | `beam_search` k>1 | Beam search without stochasticity |
| 5 | **WOR SIL** | `gumbeldore` | `wor` | Sample without replacement. Simpler than TaSaR |
| 6 | **Gumbeldore SIL** | `gumbeldore` | `gumbeldore` | Advantage-based policy improvement. Prior method |
| 7 | **Theory-Gumbeldore SIL** | `gumbeldore` | `theory_gumbeldore` | Locally estimated advantage variant |
| 8 | **TaSaR SIL** | `gumbeldore` | `tasar` | **Our method.** Commit-reconsider cycle |

Baselines 5-8 are the critical internal comparison: they isolate what TaSaR's commit-reconsider adds over other decoding strategies, all else equal.

---

## Baselines Requiring New Implementation

### RL (high priority — needed for paper table)

| # | Method | Effort | New files needed |
|---|--------|--------|-----------------|
| 9 | **REINFORCE + baseline** | 2-3 days | `core/rl_train.py`, new fn in `vne_main.py` |
| 10 | **REINFORCE + critic** | 3-5 days | Value network head, advantage computation |

Requires: reward function (terminal objective), policy gradient over sampled trajectories, baseline for variance reduction, new `learning_type="rl"` branch in `main_train_cycle`.

### Non-neural (low effort, high value)

| # | Method | Effort |
|---|--------|--------|
| 11 | **Greedy shortest-path** | 1 day |
| 12 | **Random feasible** | few hours |
| 13 | **First-fit heuristic** | 1 day |

### External VNE methods (literature)

| # | Method | Effort | Reference |
|---|--------|--------|-----------|
| 14 | **DeepViNE** | High | Blenk et al. 2018 |
| 15 | **NCO-VNE** | Medium | Yousefnezhad et al. 2020 |

---

## Target Paper Table (VNE)

Following the paper's format:

| Method | Train Paradigm | Arch | Val Gap (20-40n) | Test Gap (20-40n) | Gen Gap (40-60n) |
|--------|---------------|------|------------------|-------------------|-----------------|
| ILP optimal (HiGHS) | — | — | 0% | 0% | — |
| Random feasible | — | — | ? | ? | ? |
| Greedy heuristic | — | — | ? | ? | ? |
| Supervised | SL on ILP | LEHD | ? | ? | ? |
| REINFORCE+baseline | RL | LEHD | ? | ? | ? |
| WOR SIL | SIL | LEHD | ? | ? | ? |
| Gumbeldore SIL | SIL | LEHD | ? | ? | ? |
| **TaSaR SIL (ours)** | **SIL** | **LEHD** | **?** | **?** | **?** |
| + inference decode (k>1, s>1) | SIL | LEHD | ? | ? | ? |

Plus ablation: k×s grid, architecture variants, data scaling curves.

---

## Priority Order

1. **Now** — Baselines 1-8 (config changes only)
2. **Phase A** — Baselines 11-13 (greedy/random heuristics; quick wins)
3. **Phase B** — Baselines 9-10 (RL; needed for paper parity with TaSaR paper)
4. **Phase C** — Baselines 14-15 (external methods; only if reviewers demand)
