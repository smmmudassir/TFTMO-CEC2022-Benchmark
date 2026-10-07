# PMJMO-X3β FRSS-2026B Confirmation Protocol

Date frozen: 2026-10-06

## Purpose

This campaign is an independent confirmation of the PMJMO-X3β candidate created after the FRSS-2026 development screen. CEC-2022 is not used for tuning or candidate selection in this phase.

## Primary candidate

`X3-beta` = `PMJMOX3Beta(seed, use_microgeom=False, use_escape=True)`.

It inherits the X3-no-geometry backbone and changes only the escape policy in the primary candidate. Escape pressure is a deterministic smooth function of dimension: conservative near D=20 and progressively stronger toward D=100.

Optional stricter geometry exists in the implementation for later ablation only, but it is disabled in the primary candidate and is not part of this confirmation comparison.

## Fresh confirmation suite

- Suite label: FRSS-2026B
- Dimensions: D = 20, 50, 100
- Functions: F1-F10 from `frss2026.py`
- Transformations: new fixed shift/rotation instances, independent from the FRSS-2026 development screen
- Budget: 1000 × D function evaluations
- Runs: 30 matched runs per D×F×variant
- Blocks: 30 D×F blocks
- Variants: 4
- Total optimizer runs: 30 × 30 × 4 = 3600
- Algorithm seeds are matched across variants within each D×F×run
- No CEC-2022 result is consulted during this confirmation

## Pre-specified variants

1. `X3-beta`
2. `X3-alpha-no-geometry`
3. `X3-alpha-full`
4. `X2f-frozen`

The direct parent comparison is `X3-beta` versus `X3-alpha-no-geometry`, because both have geometry disabled and therefore isolate the revised escape policy.

## Primary analysis

For each D×F×variant block, calculate the median final error across the 30 runs. Rank the four variants within each of the 30 blocks using average ranks for ties. The primary aggregate endpoint is overall mean block rank.

Run a Friedman test across the four variants over the 30 block medians. Run paired two-sided Wilcoxon signed-rank tests comparing `X3-beta` with each of the three pre-specified references, followed by Holm correction.

## Freeze rule

Freeze PMJMO-X3β only if all of the following hold:

1. all 3600 records pass integrity checks, including exact FE budgets, finite errors, complete matched seeds and no duplicate run keys;
2. `X3-beta` is rank 1 by overall mean block rank;
3. against `X3-alpha-no-geometry`, `X3-beta` has more block wins than losses;
4. the median signed block-error difference (`X3-beta - X3-alpha-no-geometry`) is negative;
5. the paired Wilcoxon test versus `X3-alpha-no-geometry` remains significant at α=0.05 after Holm correction over the three pre-specified X3β comparisons.

If any item fails, X3β remains a development candidate and must not be represented as frozen or superior.

## Post-freeze rule

Only after the candidate passes the FRSS-2026B freeze gate may it be evaluated on the official CEC-2022 benchmark. CEC-2022 results must not be used to modify X3β after freezing.
