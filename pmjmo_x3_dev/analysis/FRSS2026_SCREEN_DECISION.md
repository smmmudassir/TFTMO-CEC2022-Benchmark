# PMJMO-X3 FRSS-2026 development-screen decision

## Freeze decision

**Do not freeze `X3-full` as PMJMO-X3.**

The 30-block fresh-suite screen does not establish statistically reliable superiority among the four tested variants. The strongest overall development variant is `X3-no-geometry`, not `X3-full`.

## Integrity

- 600 records = 30 D×F blocks × 5 matched runs × 4 variants
- dimensions: D=20, 50, 100
- functions: F1-F10
- exact budget: 1000×D evaluations in every record
- 0 duplicate variant/block/run keys
- 0 non-finite errors
- 0 negative errors
- matched algorithm seeds within each block/run
- matched instance seed within each D×F block
- all 30 artifacts carry the same source hashes

Source hashes:
- `pmjmo_x3_dev/pmjmo_x3_alpha.py`: `b967a8386475f88374884c1f063880482ee62b7af99c5260a76950111f93da27`
- `pmjmo_x3_dev/frss2026.py`: `074a7dce92d2096f813e0ef54ccfd0bdd7daf818a84a2cf8e17f9c62752faca4`
- `pmjmo_x3_dev/run_frss2026.py`: `91a414afeba510e6a9d9452014251e5641e185f2322107575901d3d1c3bb0d48`
- `PMJMO_X2F_FROZEN.py`: `89d013851d408fb51930d670b95c93e7aeb5f3d186e100fd5a68bc0118ca6976`

## Overall mean ranks

| Variant | Mean rank |
|---|---:|
| X3-no-geometry | 2.2667 |
| X3-full | 2.4667 |
| X3-backbone | 2.6333 |
| X2f-frozen | 2.6333 |

## Mean ranks by dimension

| Variant | D=20 | D=50 | D=100 |
|---|---:|---:|---:|
| X3-full | 2.1 | 2.5 | 2.8 |
| X3-no-geometry | 2.9 | 2.1 | 1.8 |
| X3-backbone | 2.4 | 2.9 | 2.6 |
| X2f-frozen | 2.6 | 2.5 | 2.8 |

## W/T/L on block medians

- X3-full vs X2f-frozen: 17/1/12
- X3-no-geometry vs X2f-frozen: 16/1/13
- X3-backbone vs X2f-frozen: 16/1/13
- X3-full vs X3-no-geometry: 10/5/15
- X3-no-geometry vs X3-backbone: 17/3/10

## Statistics

Friedman test across 30 matched problem blocks:
- chi-square = 1.640000
- p = 0.650354

Pairwise Wilcoxon signed-rank tests on block ranks with Holm correction:
- X3-no-geometry vs X3-backbone: raw p=0.152116, Holm p=0.912698
- X3-no-geometry vs X2f-frozen: raw p=0.297212, Holm p=1.0
- X3-full vs X3-no-geometry: raw p=0.439153, Holm p=1.0
- X3-full vs X3-backbone: raw p=0.470644, Holm p=1.0
- X3-full vs X2f-frozen: raw p=0.738698, Holm p=1.0
- X3-backbone vs X2f-frozen: raw p=0.925192, Holm p=1.0

No pair remains significant at alpha=0.05 after Holm correction.

## Mechanism interpretation

### Micro-geometry
X3-full vs X3-no-geometry:
- overall: 10/5/15
- D=20: 6/1/3
- D=50: 3/2/5
- D=100: 1/2/7

Geometry is useful at D=20 but becomes increasingly harmful with dimension. The current geometry gate is therefore not dimension-robust.

### Escape
X3-no-geometry vs X3-backbone:
- overall: 17/3/10
- D=20: 2/1/7
- D=50: 6/1/3
- D=100: 9/1/0

Escape is unfavorable at D=20 but increasingly valuable at D=50 and D=100.

## Gate outcome

**FAIL for X3-full.**

Advance a revised candidate from the no-geometry branch, with geometry disabled by default (or restricted to strongly supported low-dimensional cases) and a dimension-aware escape controller. Do not tune on CEC-2022. Run a larger fresh-suite confirmatory campaign before freezing any X3 version.
