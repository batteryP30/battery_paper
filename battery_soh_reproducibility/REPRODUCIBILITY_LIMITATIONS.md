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

## Later experiment coverage and independent re-execution

The new local scripts are not a claim that every submitted result has been fully rerun on the current machine. The original 708 MB study raw-data ZIP and the archived fold-level checkpoints were **not** inside `battery_paper.zip`. A separate saved frozen dataset and several experiment workbooks were available in the conversation and used for selective smoke tests, not bundled into the final source archive.

- The executable true-3.6-V censoring and quality-screened rebuild code is new and must be compared with Step 11G and CALCE quality-sensitivity workbooks after full raw-data re-execution.
- The PyTorch LSTM/Transformer source is a transparent reimplementation of the documented architectures, not an original recovered source snapshot. Minor choices such as deterministic validation battery identity and optimizer/library versions can affect performance. Do not claim byte-level equality to the archived benchmark without fold-by-fold verification.
- The original full matched Step 11HB low-pass/noise LOBO trainer was not available. `lowpass_feature_diagnostic.py` verifies the filter's within-cycle/step behavior and effect on features, but **does not** reproduce Step 11HB prediction RMSE; the manuscript's matched-filter result remains supported by its saved workbook.
- The original calibrated NASA CSV input derivation from NASA native source distributions is not available. The reproducibility boundary remains the study-input CSVs inside the original `collab.zip`.
- Original feature-selection threshold heatmaps are intentionally reconstructed from preserved diagnostics and are not falsely represented as a rerun of a lost historical grid search.

No automatic claim of publication-scale performance is made by the `--help` and small-fixture tests. A complete release should include a version-pinned raw-data source checksum manifest, a full local execution record, and links to immutable experiment outputs.
