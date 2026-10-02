# CTC-DE publication protocol

## Working title
Caterpillar Threat-Conditioned Differential Evolution (CTC-DE):
Counterfactual Decoy Search and Selective Evaluation Reallocation for Continuous Optimization

## Status
Development branch only. No superiority claim is permitted until the frozen full campaigns and
statistical tests are complete.

## Motivation and novelty boundary
The biological video motivates three defense states: concealment, deceptive signaling, and
startle/escape. The scientific contribution is not the caterpillar metaphor itself. CTC-DE is a
success-history differential-evolution framework with:
1. threat-conditioned state selection from stagnation, crowding, and fitness rank;
2. a counterfactual decoy anchor reflected away from the neighborhood of a promising solution;
3. density-aware startle escape;
4. selective evaluation reallocation: stable elites may be frozen for one generation and their
   saved function evaluations are reassigned to high-threat agents;
5. function-evaluation-driven population reduction and SHADE-style parameter memory.

Existing caterpillar foraging/swarm optimizers and camouflage/state-aware metaheuristics must be
discussed explicitly. Novelty claims must be restricted to the concrete mechanisms above.

## Development gate
Use only non-CEC classical functions for coding/tuning. Once the equations and constants are frozen,
record SHA-256 and do not tune on CEC-2022.

## Official CEC-2022 campaign
- Functions: F1-F12.
- Dimensions: D=10 and D=20.
- Bounds: [-100,100]^D.
- Budgets used by this repository: 200,000 FEs (D=10), 1,000,000 FEs (D=20).
- Runs: 30 independent runs per function/dimension.
- Matched seeds:
  base_seed = 2026220000 + dimension*10000 + function*100; runs add 0..29.
- Objective evaluator/data pinned to commit:
  fa7291054a83ce5f46132c4045c6a7878e9611e9
- Exact FE accounting is mandatory.
- Raw CSV, stdout, code/evaluator hashes, runtime and provenance are retained.

## Comparators
Use genuine/reference implementations under the identical evaluator, FE budget and matched seeds.
Minimum primary set in this repository:
- L-SHADE 1.0.1
- jSO
- L-SRTDE
- RDEx-SOP/RDE-family implementation already under validation

Do not claim state-of-the-art performance from comparisons against weak or reimplemented toy
baselines alone.

## Statistics
For every function and dimension report mean, median, standard deviation, best, worst and runtime.
Across problems report:
- Friedman average ranks and omnibus test;
- pairwise Wilcoxon signed-rank tests on matched run IDs;
- Holm correction;
- effect sizes;
- win/tie/loss counts;
- convergence curves using the same FE checkpoints.

## Ablation
After the full algorithm is frozen, run the same matched-seed protocol for:
- no_decoy
- no_escape
- no_reallocation
- base

The ablation is explanatory; it must not be used to retune the frozen full algorithm.

## Additional evidence required before submission
- CEC-2017 validation after CEC-2022, preferably D=10/30/50 (and D=100 if computationally feasible);
- at least 3 engineering design problems or one substantial real-world application family;
- complexity analysis and empirical runtime;
- parameter sensitivity for the few exposed structural constants;
- search-behavior diagnostics (diversity, threat-state proportions, evaluation allocation);
- reproducible repository release with frozen commit/tag.

## Claim policy
Accept the measured rank. If CTC-DE does not beat strong comparators, report that outcome and either
revise the algorithm in a new development version or position the paper around efficiency,
robustness, or mechanism insight. Never manufacture a rank-1 claim.
