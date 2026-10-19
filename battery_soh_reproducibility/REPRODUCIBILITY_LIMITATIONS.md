# Reproducibility boundaries and known limitations

This source release is intentionally conservative: it includes only scientific
implementations that could be tied to retained, validated project code and the
frozen Step 1--10B outputs. Two historical boundaries could not be filled
without inventing code, so they are documented rather than silently replaced.

## 1. NASA repository-native files versus validated study inputs

The final retained analysis code consumes the NASA measurement exports used in
the study:

- `B0005.csv`, `B0006.csv`, `B0007.csv`, `B0018.csv`; and
- `RW3.csv`, `RW4.csv`, `RW5.csv`, `RW6.csv`.

The release validates the measurement and operation columns in those files and
reconstructs all final SoH targets from current and time. In particular, the
final target pipeline does not use source `SOH` or `SOH_interpolated` values.

A separately validated script that converts the NASA repository's native
distribution format into these study-input CSV exports was not present in the
retained project files or prior revision record. It is therefore **not
re-created here**. Doing so would introduce an unverified data-ingestion path
into a publication release.

Accordingly, the raw-input boundary for this release is:

- original CALCE CS2 workbooks; and
- the NASA raw-measurement CSV exports actually consumed by the validated
  study code.

Once those inputs are supplied, every target reconstruction, feature-building,
modeling, statistical, robustness, table, and figure stage reported in the
revised manuscript is implemented in this package.

## 2. Preliminary correlation/stability heatmaps

The revised manuscript retains two development-stage heatmaps that predate the
strict feature-admissibility audit and final nested HGB procedure. The exact
historical raw-data grid runner was not recoverable from the retained source.

`scripts/analysis/preliminary_threshold_calibration.py` therefore contains the
preserved diagnostic matrices solely to regenerate those historical figures.
It is explicitly **not** presented as a re-execution of the original threshold
grid. The final reported HGB experiments do not depend on those matrices:
correlation threshold, retained feature count, and HGB configuration are
selected inside training-only nested validation in the final LOBO and transfer
scripts.

## 3. Full experimental re-execution during release assembly

The complete CALCE/NASA raw-study archive was not available in the release
assembly environment. The full raw-to-final workflow was therefore not
retrained during packaging. Source compilation, import/CLI checks, unit tests,
baseline recomputation, strict-feature contract checks, manuscript-table
reconstruction from the frozen outputs, and frozen-result regression checks
were executed. See `VERIFICATION.md` for the exact distinction.
