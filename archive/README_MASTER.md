# PMJMO-X2f CEC-2022 Archival Master Package

This package preserves the complete matched-seed five-algorithm CEC-2022 comparison used to evaluate frozen PMJMO-X2f.

## Dataset
- Algorithms: PMJMO-X2f-frozen, RDEx-SOP, jSO, L-SRTDE, L-SHADE-1.0.1
- Dimensions: D=10 and D=20
- Functions: CEC-2022 F1-F12
- Runs: 30 per function/dimension/algorithm
- Total run-level records: 3,600
- Matched seed formula: 2026220000 + dimension*10000 + function*100 + (run-1)
- PMJMO budget: 200,000 FEs at D=10 and 1,000,000 FEs at D=20
- PMJMO frozen SHA-256: 89d013851d408fb51930d670b95c93e7aeb5f3d186e100fd5a68bc0118ca6976

## Final rank result
1. RDEx-SOP: 2.145833
2. jSO: 2.708333
3. L-SRTDE: 2.895833
4. L-SHADE-1.0.1: 3.062500
5. PMJMO-X2f-frozen: 4.187500

PMJMO-X2f was not state of the art on this campaign. The archive is intentionally retained as a reproducibility, negative-result, benchmarking, and algorithm-development resource.

## Integrity
The master CSV contains exactly 720 records for each algorithm and has zero seed-alignment mismatches. The original comparator workflows and pinned source provenance are included.

Some comparator RESULT streams did not emit every auxiliary field (for example elapsed time or explicit NFE in all adapters). Those fields are left blank rather than inferred. The core algorithm, dimension, function, run, seed, and final error fields are retained from the completed workflow logs.

## Safe reuse
Appropriate uses include:
- reproducibility and negative-results research;
- benchmark methodology studies;
- statistical-analysis teaching;
- PMJMO-X3 and later-version baselines;
- supplementary material;
- Zenodo/OSF/Figshare archival deposition.

Do not describe PMJMO-X2f as SOTA or superior on CEC-2022.
