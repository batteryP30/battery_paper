# Manuscript-to-Code Map

The mapping below refers to the current manuscript supplied with this release.

## Data, targets, and feature policy

| Manuscript component | Reproduction code |
|---|---|
| CALCE/NASA raw loading and cycle reconstruction | `src/battery_soh_repro/data_pipeline.py` |
| Base cycle feature schema and derived causal representations | `src/battery_soh_repro/data_pipeline.py` (105 engineered measurement features plus `Cycle_Index` and three structural identity fields at the historical 109-column base-table stage) |
| Feature-admissibility audit | `src/battery_soh_repro/feature_policy.py`, `scripts/data/feature_admissibility_audit.py` |
| Table `tab:feature_audit` | `scripts/reporting/generate_manuscript_tables.py` |
| NASA RW3--RW6 independent reference-discharge target provenance | `scripts/data/nasa_rw_target_provenance.py`, `src/battery_soh_repro/data_pipeline.py` |
| Corrected variable independent observations and trivial baselines | `scripts/data/prepare_variable_dataset.py` |
| Corrected constant observations and trivial baselines | `scripts/data/prepare_constant_dataset.py` |
| Table `tab:dataset_summary` | preprocessing scripts plus `scripts/reporting/generate_manuscript_tables.py` |

## Main modelling

| Manuscript experiment | Reproduction code |
|---|---|
| Finite H0--H2 configuration table `tab:hgb_configs` | `scripts/reporting/generate_manuscript_tables.py` |
| Variable strict electrical HGB and strict electrical + age HGB | `scripts/modeling/variable_strict_lobo.py` |
| Constant strict electrical HGB and strict electrical + age HGB | `scripts/modeling/constant_strict_lobo.py` |
| Table `tab:lobo_macro` | both LOBO scripts, preprocessing baseline outputs, `scripts/reporting/generate_manuscript_tables.py` |
| Tables `tab:constant_perbattery`, `tab:variable_perbattery` | the two LOBO scripts plus `scripts/reporting/generate_manuscript_tables.py` |
| Constant NASA -> CALCE and CALCE -> NASA transfer / `tab:transfer_results` | `scripts/modeling/constant_cross_dataset_transfer.py`, bootstrap output, `scripts/reporting/generate_manuscript_tables.py` |
| 3.6 V observation-depth audit / `tab:partial_depth` | `scripts/modeling/constant_partial_observation_3p6v.py`, `scripts/reporting/generate_manuscript_tables.py` |
| 3.6 V partial-observation-assisted model comparison / `tab:partial_3p6` | `scripts/modeling/constant_partial_observation_3p6v.py`, `scripts/reporting/generate_manuscript_tables.py` |

## Statistical and robustness analysis

| Manuscript experiment | Reproduction code |
|---|---|
| 20,000 whole-battery bootstrap confidence intervals / `tab:bootstrap_primary` | `scripts/analysis/bootstrap_and_ablations.py`, `scripts/reporting/generate_manuscript_tables.py` |
| Strict voltage-shape, dynamic-response, temporal/trend, and explicit-age ablations / `tab:feature_ablation` | `scripts/analysis/bootstrap_and_ablations.py`, `scripts/reporting/generate_manuscript_tables.py` |
| Clean-to-raw reconstruction validation | `scripts/analysis/measurement_robustness.py` |
| Systematic voltage offset, current gain, and combined errors | `scripts/analysis/measurement_robustness.py` |
| 1 mV + 0.1% and 5 mV + 0.5% stochastic perturbations / `tab:measurement_perturbation` | `scripts/analysis/measurement_robustness.py`, `scripts/reporting/generate_manuscript_tables.py` |
| Preliminary correlation/stability heatmaps | `scripts/analysis/preliminary_threshold_calibration.py` |

The preliminary correlation/stability file regenerates the historical
development heatmaps from the preserved diagnostic matrices. The original
raw-data threshold-grid runner could not be recovered and is not fabricated in
this release. See `CODE_PROVENANCE.md` and `VERIFICATION.md`.

## Manuscript figures

| Figure files referenced by LaTeX | Reproduction code |
|---|---|
| `charge.png`, `Discharge.png` | `scripts/reporting/generate_descriptive_figures.py` |
| `resistance.png`, `dv_dt.png` | `scripts/reporting/generate_descriptive_figures.py` |
| `const_rmse.png`, `const_feat.png`, `var_rmse.png`, `var_feat.png` | `scripts/analysis/preliminary_threshold_calibration.py` |

## Manuscript tables

`scripts/reporting/generate_manuscript_tables.py` creates a CSV and LaTeX
fragment for all twelve table labels in the current manuscript:

- `dataset_summary`
- `feature_audit`
- `hgb_configs`
- `lobo_macro`
- `constant_perbattery`
- `variable_perbattery`
- `bootstrap_primary`
- `transfer_results`
- `partial_depth`
- `partial_3p6`
- `feature_ablation`
- `measurement_perturbation`

The reporting script derives values from pipeline outputs rather than embedding
published result numbers.

## Integrity checks

`scripts/validation/validate_reproduced_results.py` compares reproduced
headline metrics against the frozen manuscript values and enforces the
clean-to-raw reconstruction tolerance. The tests under `tests/` check strict
feature contracts, static release hygiene, and the preserved preliminary
heatmap diagnostics.

## Input-boundary note

All manuscript computational stages after the validated study-input archive are
covered above. The retained source does not contain a separately validated
converter from NASA repository-native packaging into the NASA CSV measurement
exports used by the study. This is documented in
`REPRODUCIBILITY_LIMITATIONS.md` rather than replaced with an unverified parser.
