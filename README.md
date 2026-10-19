# Battery SoH Code

This repository contains the Python source used to reproduce the computational
workflow reported in **Battery-Level and Cross-Dataset Generalization of
Data-Driven Lithium-Ion Battery State-of-Health Estimation**.

The final analysis is deliberately organized around information availability.
The primary models are strict capacity-independent models: they do not use
same-cycle full discharge capacity, energy or duration, endpoint-equivalent
capacity proxies, previous-cycle full capacity, or current-integrated
incremental-capacity quantities. Partial-observation and history-assisted
information are reported separately.

No datasets, generated result files, figures, trained models, checkpoints, or
notebooks are included.

## Python environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e .
```

For repository tests:

```bash
pip install -r requirements-dev.txt
pytest
```

## Raw-data input boundary

The validated study pipeline takes one combined ZIP as input. It may contain an
extra top-level directory because discovery is recursive. The ZIP must resolve:

- original CALCE CS2 workbooks for `CS2_33`--`CS2_38`, `CS2_3`, and `CS2_9`;
- NASA raw-measurement CSV exports used by the study: `B0005.csv`, `B0006.csv`,
  `B0007.csv`, `B0018.csv`, and `RW3.csv`--`RW6.csv`.

The pipeline validates the required CALCE sheets and NASA measurement/operation
columns before modelling and reconstructs the final capacity-based targets from
current and time. Source-provided NASA `SOH` and `SOH_interpolated` fields are
not used by the final target pipeline.

A separately validated converter from the NASA repository-native distribution
format into these study-input CSV exports was not present in the retained
project source and has not been invented for this release. See
`REPRODUCIBILITY_LIMITATIONS.md` and `config/raw_data_layout.md` for the exact
input contract.

## Full workflow

Run from the repository root after installing the package:

```bash
python scripts/run_all.py --raw-zip /path/to/collab.zip
```

By default, derived files are written to `outputs/` and temporary extracted
raw files to `work/`. Both locations are ignored by Git.

The measurement-robustness stage is intentionally expensive because it
rebuilds features from perturbed raw measurements for every held-out battery
and Monte Carlo replicate. For a shorter structural run:

```bash
python scripts/run_all.py \
  --raw-zip /path/to/collab.zip \
  --skip-measurement-robustness
```

That option is for development only and does **not** reproduce the complete
manuscript.

## Individual stages

The stages can also be executed independently in manuscript order:

```bash
python scripts/data/feature_admissibility_audit.py --raw-zip /path/to/collab.zip
python scripts/data/nasa_rw_target_provenance.py --raw-zip /path/to/collab.zip
python scripts/data/prepare_variable_dataset.py --raw-zip /path/to/collab.zip
python scripts/data/prepare_constant_dataset.py --raw-zip /path/to/collab.zip

python scripts/modeling/variable_strict_lobo.py
python scripts/modeling/constant_strict_lobo.py
python scripts/modeling/constant_cross_dataset_transfer.py
python scripts/modeling/constant_partial_observation_3p6v.py --raw-zip /path/to/collab.zip

python scripts/analysis/bootstrap_and_ablations.py
python scripts/analysis/measurement_robustness.py --raw-zip /path/to/collab.zip

python scripts/analysis/preliminary_threshold_calibration.py
python scripts/reporting/generate_descriptive_figures.py --raw-zip /path/to/collab.zip
python scripts/reporting/generate_manuscript_tables.py
python scripts/validation/validate_reproduced_results.py
```

Scripts use relative default locations under `outputs/`; each path can be
overridden from the command line.

## Feature-schema count used by the implementation

The retained base cycle extractor emits 105 engineered numeric measurement
features. At the base cycle-table stage it is accompanied by `Cycle_Index`,
`battery_id`, `source_dataset`, and `protocol`, giving the 109-column schema
that appeared in the historical project accounting before the larger derived
candidate space was created. The three identity fields are structural metadata,
not model predictors. The final strict candidate spaces are the frozen audited
sets of 335 constant-protocol and 314 variable-protocol features. This source
release preserves the validated extractor rather than changing feature logic to
force a manuscript count.

## Methodological safeguards encoded in the source

- Capacity-based targets are reconstructed from raw current and time.
- NASA randomized-use targets are defined only by independent reference
  discharges; source `SOH` / interpolated target fields are not used.
- The 70--100% SoH interval is a declared analysis scope, not a
  target-inspection outlier filter.
- Feature admissibility is frozen before model evaluation.
- Feature availability filtering, median imputation, correlation pruning,
  feature ranking, top-K selection, and HGB configuration selection are
  training-only.
- LOBO splits hold out entire batteries.
- Each battery receives equal total training weight.
- Cross-dataset transfer uses target-source labels only for final evaluation.
- The 3.6 V experiment is explicitly partial-observation-assisted and is
  restricted to constant protocols because variable-protocol coverage was
  inadequate.
- Bootstrap resampling is at the battery level.
- Ablations retrain the complete nested pipeline after feature-family removal.
- Measurement perturbations are applied to raw voltage/current before feature
  extraction. Clean raw reconstruction must reproduce the frozen unperturbed
  fold before perturbed results are accepted.

## Preliminary threshold-calibration figures

The manuscript retains two early development heatmaps documenting the
chronology of feature-selection threshold calibration. The exact historical
grid-execution script was not present in the retained source set. Therefore
`scripts/analysis/preliminary_threshold_calibration.py` regenerates those
figures from the diagnostic matrices preserved in the original manuscript; it
does **not** claim to rerun the historical raw-data experiment. Those
preliminary RMSEs are not used as final model evidence and do not determine
the nested final results.

## Reproducibility check

After a complete run:

```bash
python scripts/validation/validate_reproduced_results.py \
  --outputs-root outputs
```

This checks the principal frozen manuscript metrics, battery counts,
cross-dataset results, 3.6 V results, ablations, robustness results, and the
clean raw-reconstruction tolerance.

See `MANUSCRIPT_CODE_MAP.md` for a table/figure-to-script mapping,
`CODE_PROVENANCE.md` for consolidation decisions, `VERIFICATION.md` for the
checks actually executed while assembling this package, and
`REPRODUCIBILITY_LIMITATIONS.md` for the two historical input/parameter-study
boundaries that could not be reconstructed without fabricating code.
