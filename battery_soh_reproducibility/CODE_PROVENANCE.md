# Code provenance and consolidation decisions

This release is a source-only consolidation of the computational work supporting
the current manuscript. It is not a dump of every development notebook.

## Retained scientific logic

The shared raw-data and feature-engineering implementation preserves the
validated feature extraction used in the final experiments, including the
cycle reconstruction, voltage/current statistics, dynamic-response features,
protocol-phase descriptors, partial-voltage candidates, incremental-capacity
candidates, causal transformations, and source-specific loaders.

The publication workflows retained as executable Python are:

- feature-admissibility audit;
- NASA randomized-use reference-target reconstruction;
- corrected variable-protocol observation construction and trivial baselines;
- corrected constant-protocol cycle construction and trivial baselines;
- strict variable nested LOBO HGB;
- strict constant nested LOBO HGB;
- constant-protocol NASA/CALCE cross-dataset transfer;
- constant-protocol 3.6 V partial-observation analysis;
- whole-battery bootstrap confidence intervals;
- strict feature-family ablations;
- clean-to-raw reconstruction validation;
- systematic voltage/current perturbations;
- the Step 10B lower-severity perturbation extension;
- descriptive manuscript figures and manuscript table generation.

The Step 10 and Step 10B perturbation code is consolidated into one script so
the final robustness suite is generated from a single implementation.

## Superseded development code not included

Earlier exploratory Run 2/3/4/5 notebooks and scripts are intentionally not
packaged. They contained model comparisons, feature policies, target-proxy
experiments, variable-transfer formulations, ablations, or noise studies that
were superseded by the final methodology. In particular, the release excludes:

- the old target-proxy/3.2 V feature formulation;
- old visual target-spike filtering and charge-duration cleaning;
- obsolete XGBoost/LightGBM/random-forest/stacking headline experiments;
- obsolete variable cross-dataset transfer;
- old feature-family ablations whose reference model used the superseded
  feature space;
- the earlier white-noise-only perturbation study;
- Colab/Google-Drive wrappers, checkpoints, cached features, and notebooks.

This keeps the public source aligned with the experiments actually reported in
the manuscript rather than preserving obsolete paths that could be mistaken
for current methodology.

## Historical preliminary threshold-calibration limitation

The manuscript retains development-stage correlation/stability heatmaps that
predate the strict feature-admissibility audit and final nested HGB procedure.
The original raw-data grid runner that created those historical matrices was
not present in the retained project files. Reconstructing a new runner and
claiming it was the historical implementation would therefore be speculative.

For that reason, the release includes
`scripts/analysis/preliminary_threshold_calibration.py` only to regenerate the
two manuscript heatmaps from the preserved diagnostic matrices. The script is
explicitly marked as figure reconstruction, not as a rerun of the historical
raw-data experiment. The final correlation-threshold, top-K, and HGB
configuration selection is fully executable in the strict LOBO and transfer
scripts and does not depend on those historical matrices.

## Base-feature accounting

The validated `build_cycle_features` implementation emits 105 engineered
measurement-feature columns. With `Cycle_Index` plus the three structural
identity fields added at the base-table stage (`battery_id`, `source_dataset`,
`protocol`), the historical table contains 109 columns before the later robust
and causal transformations. Identity columns are not admitted as predictors.
The release preserves this validated implementation and the frozen 335/314
strict feature lists rather than modifying scientific logic to make a prose
feature count line up artificially.

## NASA ingestion boundary

The final retained code consumes the NASA sample-level CSV exports used in the
study. No validated native-distribution-to-CSV converter was found in the
retained project history. The release therefore documents that boundary instead
of introducing a newly invented parser.
