# VNE Generalization Axes — Strategic Design

> The TaSaR paper used exactly 1 generalization axis per problem (size scaling).
> We must be equally disciplined. More axes ≠ stronger paper.

## The Paper's Formula

| Problem | Train | Test | Axes |
|---------|-------|------|------|
| TSP | N=100 | N=200, 500 | 1 (city count) |
| CVRP | N=100 | N=200, 500 | 1 (customer count) |
| JSSP | 15×{10,15,20} | 8 Taillard sizes up to 100×20 | 1 (job×machine count) |

**They tested exactly ONE thing: "does it generalize to larger instances?"**

They did NOT test: different distance metrics, constrained TSP, asymmetric TSP, time windows, multiple depots, heterogeneous vehicles, etc. They picked ONE axis and executed it perfectly.

## VNE Possible Axes — Ranked by Impact

### Axis A: Substrate size scaling ⭐⭐⭐ (MUST HAVE)

**Paper analog**: TSP N=100 → N=200, N=500

| | Substrate nodes | Requests | Topology | ILP feasible? |
|---|----------------|----------|----------|---------------|
| Train | 20–40 | 2–6 | line | Yes (<1s) |
| Test 1 | 40–60 | 2–6 | line | Yes (~4s) |
| Test 2 | 60–80 | 2–6 | line | Yes (~9s) |
| Test 3 | 80–100 | 2–6 | line | Marginal (~19s/inst) |

**Story**: "Trained on 20-40 node substrates, TaSaR maintains sub-X% gap on 80-100 node substrates — a 4× scale increase."

**Pros**: Direct paper analog, easy to understand, ILP labels available for all test sizes.
**Cons**: ILP doesn't actually "fail" until n>150 for loose regime. The "neural only option" story is weaker. Sequence length doesn't change (still ~10 decisions).

**Verdict**: ✅ Do this. It's the minimum expected by any reviewer who read the paper.

### Axis B: Topology generalization ⭐⭐⭐ (UNIQUE CONTRIBUTION)

**Paper analog**: NONE. This is genuinely new. The paper always used complete graphs.

| | Substrate nodes | Requests | Topology |
|---|----------------|----------|----------|
| Train | 20–40 | 2–6 | **line only** |
| Test 1 | 20–40 | 2–6 | **ER random (p=0.3)** |
| Test 2 | 40–60 | 2–6 | **ER random (p=0.3)** |
| Test 3 | 20–40 | 2–6 | **2D mesh/grid** |

**Story**: "TaSaR trained exclusively on line topologies achieves near-optimal embedding on random and mesh topologies — graph structures it has never seen. This demonstrates that the learned policy captures substrate-agnostic embedding principles rather than memorizing line-specific patterns."

**Pros**: 
- Genuinely novel — no NCO paper has tested topology generalization
- Reviewer can't say "this is just TSP but for VNE"
- ILP labels are still available on test topologies
- Line→ER is a hard test: ER has ~2× more edges, completely different path structure

**Cons**: 
- Requires implementing ER and mesh topology generators (not in `instance_generator.py` yet — currently only supports line)
- If it doesn't work (gap >>10%), it weakens the paper rather than strengthening it

**Verdict**: ✅ Do this, but test feasibility early. If line→ER gap is small (<5%), this becomes a headline result. If not, drop it.

### Axis C: Request load scaling ⭐ (WEAK SIGNAL)

| | Requests | Decisions (mean) |
|---|----------|-----------------|
| Train | 2–6 | 10 |
| Test 1 | 4–8 | 15 |
| Test 2 | 6–10 | 20 |

**Story**: "Handles longer sequences with more requests."

**Pros**: Direct JSSP analog (longer sequences).
**Cons**: Our sequences are already very short (10 decisions vs JSSP's 150-300). Going from 10→20 decisions is only 2×, while JSSP went 150→2000 (13×). The signal is weak.

**Verdict**: ❌ Skip. Not enough dynamic range to be impressive. If reviewers ask, can add later.

### Axis D: Resource contention ⭐ (DIFFERENT PROBLEM)

Requires admission control, different objective (lex/profit), trajectory changes. This is a separate paper, not a generalization test.

**Verdict**: ❌ Skip. Parked for follow-up work.

### Axis E: Multi-size training ⭐⭐ (REFINEMENT)

JSSP-style: train on mixed {20-30, 25-35, 30-40} substrates rather than single range.

**Verdict**: ❌ Skip for now. If single-size training already generalizes well, this is an ablation at best.

## Recommended Design: 2 Axes × 3 Test Columns Each

### Main Results Table (train: 20-40n, line topology, loose resources)

| Method | Val (20-40n, line) | Test (40-60n, line) | Test (60-80n, line) | Test (40-60n, ER) |
|--------|-------------------|---------------------|---------------------|-------------------|
| ILP optimal | 0% | 0% | 0% | 0% |
| Supervised (SL on ILP) | ? | ? | ? | ? |
| REINFORCE (RL) | ? | ? | ? | ? |
| WOR SIL | ? | ? | ? | ? |
| Gumbeldore SIL | ? | ? | ? | ? |
| **TaSaR SIL** | **?** | **?** | **?** | **?** |
| + inference (k>1, s>1) | ? | ? | ? | ? |

**Just 6 rows × 4 columns = 24 numbers. The paper's Table 1 had ~30 numbers. Comparable.**

### What this design achieves:

1. **Column 1 (val)**: Shows all methods on training distribution. Establishes baseline ordering.
2. **Column 2 (40-60n, line)**: First generalization test. Does the gap stay small?
3. **Column 3 (60-80n, line)**: Second generalization test. Does it continue to degrade gracefully?
4. **Column 4 (40-60n, ER)**: Topology generalization. The unique VNE result.

**The headline**: "TaSaR trained on 20-40 node line topologies achieves a X% optimality gap on 60-80 node substrates (3× larger) and Y% on random topologies never seen during training."

### What we explicitly SKIP (and why):

| Skipped | Reason |
|---------|--------|
| Request count scaling | Only 2× range (10→20 decisions), weak signal |
| Chain length scaling | Same — small dynamic range |
| Resource contention | Different problem, needs admission control |
| Multi-size training | Refinement, not headline |
| More than 3 substrate sizes | 3 is enough (paper used 2: N=200, 500) |
| More than 1 alternative topology | ER is the hardest; mesh is bonus if ER works |
| Full k×s grid on all axes | Do on val only, then best config on test |

## Experiment Count

```
Phase A (diagnostics):   
  A1: Optimality gap on current supervised model     → 0 GPU-hours (use existing checkpoint)

Phase B (main table):    
  B1-B4: TaSaR SIL (vary k, s)                      → 4 runs × 15 epochs
  B5: Gumbeldore SIL                                  → 1 run × 15 epochs  
  B6: WOR SIL                                         → 1 run × 15 epochs
  Total: ~6 SIL runs                                  → ~30-50 GPU-hours

Phase C (RL baseline):   
  C1: Implement REINFORCE+baseline                    → development time
  C2: REINFORCE training run                          → 1 run × 15 epochs

Phase D (generalization):
  D1: Best method on 40-60n line test set             → eval only (no training)
  D2: Best method on 60-80n line test set             → eval only
  D3: Best method on 40-60n ER test set               → eval only
  D4: Top-3 baselines on same test sets               → eval only

Phase E (topology generator):
  E1: Implement ER random topology generator          → development time
  E2: Generate ER test sets with ILP labels           → CPU time (HiGHS)

Total: ~6 training runs + eval passes + 2 new code modules (RL + ER topology)
```

## Go/No-Go Decision Points

1. **After Phase A**: If optimality gap > 20%, fix architecture before SIL
2. **After Phase B1** (first TaSaR run): If SIL degrades vs supervised, pivot to WOR/Gumbeldore or tune k,s
3. **After Phase B complete**: If TaSaR doesn't beat supervised by >2%, consider whether SIL is worth it for VNE
4. **After Phase E1** (ER generator): Test supervised model on ER. If gap > 15% even for supervised, topology generalization may not work — drop Axis B
