# CTC-DE frozen publication manifest

## Frozen optimizer
- Algorithm: Caterpillar Threat-Conditioned Differential Evolution (CTC-DE)
- Frozen optimizer SHA-256: `70bc55e8cb21d1ad65ab0f3690c6786ee364591d05c6c37351fb66c5a55427a5`
- Frozen branch: `ctc-de-frozen-v1`
- Pilot workflow run: `37035421123`
- Pilot status: SUCCESS
- Pilot blocks: D={10,20}, F={1,2,3}, 5 runs/block
- Pilot checks: finite objective error; exact FE budget; unique runs/seeds; identical optimizer SHA across blocks

## CEC-2022 protocol
- Functions: F1-F12
- Dimensions: D=10 and D=20
- Runs: 30 independent matched-seed trials per block
- D=10 budget: 200000 FEs
- D=20 budget: 1000000 FEs
- Seed formula: `2026220000 + D*10000 + F*100 + (run-1)`
- CEC evaluator source commit: `fa7291054a83ce5f46132c4045c6a7878e9611e9`

## Reference campaigns
- L-SHADE 1.0.1 run: `37018775160`
- jSO run: `37023749798`
- RDEx-SOP run: `37026217103`
- matched L-SRTDE run: `37030142388`
- Expected reference blocks: 96 (4 algorithms × 2 dimensions × 12 functions)
- Expected reference rows: 2880

## Frozen CTC-DE campaigns
- Full workflow: `ctc-de-frozen-full-cec2022`
- Ablation workflow: `ctc-de-frozen-ablation-cec2022`
- Ablations: no_decoy, no_escape, no_reallocation, base
- Reference audit workflow: `ctc-de-reference-audit`

## Claim discipline
No statement of superiority, state-of-the-art status, rank-1 performance, or significant advantage may be added to the manuscript unless supported by the completed frozen campaign and prespecified statistical analysis.

## Novelty boundary
The paper must not claim novelty for caterpillar inspiration, periscopic/head-waving search, pheromone search, fractal search, dormancy, camouflage, diversity feedback, or generic state switching. The proposed contribution is the combination of threat-conditioned control, counterfactual decoy anchors, density-oriented startle escape, and selective objective-evaluation reallocation under an exact global FE budget.
