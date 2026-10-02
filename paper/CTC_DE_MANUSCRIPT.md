# Manuscript skeleton — CTC-DE

## Proposed title
**Caterpillar Threat-Conditioned Differential Evolution: Counterfactual Decoy Search and Selective Evaluation Reallocation for Continuous Optimization**

> Draft status: methodology scaffold only. Results/conclusions must be populated from frozen campaign artifacts.

## Abstract
Population-based optimizers often spend function evaluations uniformly even when candidate states differ markedly in stagnation, crowding, and solution quality. This work proposes Caterpillar Threat-Conditioned Differential Evolution (CTC-DE), a success-history differential-evolution framework in which a composite threat signal controls three search responses and the allocation of objective evaluations. Low-threat candidates use current-to-pbest refinement; medium-threat candidates construct counterfactual decoy anchors reflected away from the neighborhood of a promising point; high-threat candidates activate density-aware startle escape. Stable elite candidates may temporarily remain unevaluated, and the saved objective calls are reassigned to high-threat candidates under the same global function-evaluation budget. [RESULT SENTENCES TO BE INSERTED ONLY AFTER FROZEN CAMPAIGNS.] Code, seeds, evaluator provenance, and raw results are released for reproducibility.

## 1. Introduction
- Expensive black-box optimization makes evaluation allocation as important as move generation.
- Existing adaptive DE methods learn mutation/crossover parameters, but generally do not explicitly reallocate per-generation evaluations using a population-state threat signal.
- Bio-inspired narrative: caterpillar defense illustrates concealment, deceptive signaling, and startle/escape; the paper contribution is the mathematical search/evaluation architecture, not the metaphor.
- Contributions:
  1. threat signal combining stagnation, normalized crowding, and rank;
  2. counterfactual decoy anchor operator;
  3. density-aware startle escape;
  4. selective evaluation reallocation under exact FE accounting;
  5. frozen reproducibility protocol with strong reference baselines and ablations.

## 2. Related work
Cover separately:
- SHADE/L-SHADE and success-history adaptive differential evolution.
- jSO, L-SRTDE and recent competitive DE variants.
- Caterpillar/inchworm-inspired optimization literature.
- Dormancy/state-switching metaheuristics including APO.
- Population-state and camouflage-inspired methods including OCO.
Do not claim that state switching, dormancy, diversity feedback, or camouflage alone is novel.

## 3. Proposed CTC-DE
### 3.1 State variables
For population X and fitness f, compute normalized rank r_i, nearest-neighbor crowding c_i, and stagnation s_i.

### 3.2 Threat signal
T_i = clip(0.45 s_i + 0.35 c_i + 0.20 r_i, 0, 1).

### 3.3 Concealment/refinement state
Use success-history current-to-pbest/1 mutation and binomial crossover.

### 3.4 Counterfactual decoy state
For a selected p-best vector p and its local-neighborhood centroid mu:
d = clip[p + kappa(p-mu)].
Mutation becomes:
v_i = x_i + F(d-x_i) + F(x_r1-x_r2).

### 3.5 Startle escape state
Construct a normalized repulsion direction from the candidate to the centroid of its nearest neighbors and add a bounded heavy-tailed displacement to the standard DE mutation.

### 3.6 Selective evaluation reallocation
Stable elites satisfying low stagnation and low crowding may be frozen for one generation. Exactly the same number of saved evaluations is reassigned to the highest-threat active candidates. The global FE budget is unchanged.

### 3.7 Success history and population reduction
Use SHADE-style memories M_F/M_CR, external archive, p-best sampling and linear population-size reduction.

### 3.8 Complexity
State explicitly:
- objective calls dominate expensive black-box settings;
- pairwise distance construction is O(N^2 D) per generation;
- DE mutation/crossover is O(ND);
- memory is O(ND + N^2) in the reference implementation.
Discuss a kNN/partial-distance implementation for large N.

## 4. Experimental protocol
### 4.1 Development discipline
Tune only on non-CEC development functions, then freeze source hash before official CEC evaluation.

### 4.2 CEC-2022
F1-F12, D=10/20, 30 independent matched-seed runs, exact repository FE budgets and pinned evaluator.

### 4.3 Competitors
Reference/genuine L-SHADE, jSO, L-SRTDE and RDEx/RDE-family comparators under identical evaluator, budget and seeds.

### 4.4 Statistics
Descriptive statistics, Friedman average ranks, matched Wilcoxon signed-rank, Holm correction, rank-biserial effect sizes, win/tie/loss and FE-aligned convergence curves.

### 4.5 Ablation
Full vs no-decoy vs no-escape vs no-reallocation vs base.

### 4.6 Scalability and sensitivity
Add CEC-2017 D=10/30/50 and sensitivity of threat weights/thresholds without retuning against competitors.

### 4.7 Engineering applications
At least three constrained engineering design tasks, using a predeclared constraint-handling method.

## 5. Results
**Do not write superiority text until CSV artifacts pass validation.**
Tables:
1. CEC-2022 mean ± std.
2. CEC-2022 median and rank.
3. Friedman/Wilcoxon-Holm/effect sizes.
4. Ablation.
5. Runtime.
6. Engineering tasks.
Figures:
1. Conceptual mechanism.
2. Convergence curves.
3. Diversity/threat-state trajectories.
4. Evaluation reallocation over FEs.
5. Critical-difference/rank visualization if statistically appropriate.

## 6. Discussion
Explain where CTC-DE helps and where it loses. Test whether selective reallocation contributes independently from decoy/escape. Discuss O(N^2) state-estimation cost and sensitivity to population geometry.

## 7. Threats to validity
- Benchmark overfitting.
- Multiple-comparison inflation.
- implementation bias;
- stochastic variability;
- metaphor-driven post-hoc explanation.
Mitigations: frozen hashes, matched seeds, reference implementations, correction tests, independent suites/applications.

## 8. Conclusion
Populate only after experiments. Avoid "best", "state-of-the-art", or "superior" unless supported by the complete prespecified comparisons.

## Reproducibility statement
Release exact commit/tag, raw run CSVs, provenance hashes, seeds, evaluator source commit, analysis scripts and environment versions.
