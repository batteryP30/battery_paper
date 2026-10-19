# Verification summary

This document distinguishes checks actually executed while assembling this ZIP
from experiments that were only inspected because the complete raw-data archive
was not available in the packaging environment.

## Checks executed successfully

The following checks were run against the final source tree in this release:

- The latest supplied LaTeX manuscript (`mdpi (1)(2).tex`) was inspected and its
  computational reporting contract was reconciled with the package: all 12
  manuscript table labels and all referenced descriptive/preliminary figure
  files are mapped in `MANUSCRIPT_CODE_MAP.md`.
- The package was installed in editable mode locally with build isolation and
  dependency downloads disabled, confirming that the package metadata and
  import structure are valid in the available environment.
- `pytest -q` completed successfully: **11 tests passed**.
- `python -m compileall -q src scripts tests` completed successfully.
- Every executable research/reporting/validation script completed `--help`
  successfully, including the top-level `scripts/run_all.py` launcher.
- The release tree was scanned for prohibited packaged artifacts: no notebooks,
  result CSV/XLSX files, figures, models, logs, or Git metadata are included.
- The frozen strict feature-name contracts were compared directly with the
  validated supplied feature-name outputs and matched exactly, including order:
  **335 constant-protocol** strict-safe features and **314 variable-protocol**
  strict-safe features.
- The feature-admissibility count contract is unit-tested:
  - constant: 335 strict-safe, 105 partial-observation, 30 history-only,
    66 inadmissible, 85 review/excluded;
  - variable: 314 strict-safe, 91 partial-observation, 30 history-only,
    91 inadmissible, 46 review/excluded.
- The retained base feature extractor was exercised on synthetic charge/CV/
  discharge cycles. It produced **105 engineered measurement features** plus
  `Cycle_Index`; adding `battery_id`, `source_dataset`, and `protocol` produced
  the historical **109-column base-table schema**. Structural identity columns
  are not model predictors.
- Constant and variable trivial/history-assisted baselines were recomputed from
  the supplied frozen corrected observation tables. Their macro RMSE and MAE
  values matched the validated Step 5 and Step 3 summaries to numerical
  precision (maximum absolute differences below `8e-15`).
- The final validation script was run against the supplied frozen Step 3--10B
  outputs arranged into the consolidated release schema. All frozen manuscript
  regression checks passed, including dataset sizes, LOBO metrics, transfer,
  3.6 V results, ablations, robustness scenarios, and clean raw-reconstruction
  tolerance.
- The manuscript-table generator was run against those frozen outputs and
  produced **12 CSV tables and 12 LaTeX fragments** (24 generated artifacts in
  the temporary verification directory). Those generated verification files
  are not included in this ZIP.
- The preliminary threshold-calibration figure script was executed and produced
  the four historical heatmap files referenced by the manuscript. Those
  generated verification figures are not included in this ZIP.

## What was not re-executed during ZIP assembly

The complete combined CALCE/NASA raw-study archive was not available in the
packaging runtime. Therefore the full raw-to-final workflow was **not** rerun
while assembling this ZIP. In particular, this packaging session did not
re-execute:

- reconstruction from every original CALCE/NASA measurement file;
- the complete nested LOBO HGB training runs;
- cross-dataset model fitting;
- strict feature-family ablation retraining; or
- the full raw-perturbation feature reconstructions for all Monte Carlo
  replicates.

Those implementations were inspected, compiled, import/CLI checked, and tied to
the supplied validated output files, but this document does not claim a fresh
raw-data reproduction that was not actually performed.

After supplying the validated study-input archive, execute:

```bash
python scripts/run_all.py --raw-zip /path/to/collab.zip
python scripts/validation/validate_reproduced_results.py --outputs-root outputs
```

The second command is the release regression check against the frozen manuscript
headline values.

## Historical limitations that are deliberately not fabricated

Two retained project boundaries could not be recovered as independently
validated implementations:

1. a converter from the NASA repository-native distribution format into the
   sample-level NASA CSV exports consumed by the validated study code; and
2. the original raw-data runner that produced the preliminary
   correlation/stability threshold-grid heatmaps.

The first is documented as the NASA input boundary. The second is represented
only by a figure-reconstruction script containing the preserved diagnostic
matrices. Neither limitation changes the final Step 1--10B model-selection,
validation, transfer, ablation, or robustness procedures. See
`REPRODUCIBILITY_LIMITATIONS.md` for details.
