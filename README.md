# TFTMO CEC-2022 Benchmark Campaign

Reproducible benchmark repository for the frozen **TFTMO-ADG** candidate against source-faithful comparator implementations on the official CEC-2022 bound-constrained single-objective suite.

## Protocol

- Functions: CEC-2022 F1-F12
- Dimensions: D=10 and D=20
- Runs: 30 independent runs per function/dimension/algorithm
- Bounds: [-100, 100]^D
- Budget: 200,000 FEs at D=10; 1,000,000 FEs at D=20
- Frozen candidate: TFTMO-ADG
- Comparators: corrected L-SHADE 1.0.1, L-SRTDE, jSO, RDEx/RDE-family source package
- Search logic of comparator sources is preserved; only benchmark/evaluation adapters are permitted.

## Source provenance

- L-SHADE 1.0.1: Ryoji Tanabe's corrected CEC-2014 archive
- L-SRTDE: Vladimir Stanovov's public L-SRTDE_CEC-2024 repository (contains CEC-2022 evaluator/data)
- jSO: source-faithful CEC-2017 package mirrored in TBU-AILab/ResourceFiles_TEVC2021
- RDEx-SOP: Sichen Tao et al.'s public IEEE CEC 2025 RDEx series repository

Raw runs, convergence histories, hashes, and statistical summaries will be committed/uploaded as campaign artifacts.
