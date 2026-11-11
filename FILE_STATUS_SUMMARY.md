# Source file change inventory

Compared by exact SHA-256 file content against the supplied `battery_paper.zip`, excluding the pre-existing `.git/` internals (which represent a machine-specific repository state, not research source) and generated caches.

- **Unchanged original files:** 29
- **Modified original files:** 4
- **New files:** 28 (plus this summary and the CSV inventory)
- **Removed original files:** 0

The unchanged files listed here have exactly the same bytes as those in the original ZIP. Other files may be functionally equivalent but are classified as modified whenever their content changed. The original `.git` history is **not** shipped in the updated source-code package; keep your separate Git repository and commit the updated files there.

## Unchanged original files

- `LICENSE`
- `battery_soh_reproducibility/.gitignore`
- `battery_soh_reproducibility/config/raw_data_layout.md`
- `battery_soh_reproducibility/pyproject.toml`
- `battery_soh_reproducibility/requirements-dev.txt`
- `battery_soh_reproducibility/requirements.txt`
- `battery_soh_reproducibility/scripts/analysis/bootstrap_and_ablations.py`
- `battery_soh_reproducibility/scripts/analysis/measurement_robustness.py`
- `battery_soh_reproducibility/scripts/analysis/preliminary_threshold_calibration.py`
- `battery_soh_reproducibility/scripts/data/feature_admissibility_audit.py`
- `battery_soh_reproducibility/scripts/data/nasa_rw_target_provenance.py`
- `battery_soh_reproducibility/scripts/data/prepare_constant_dataset.py`
- `battery_soh_reproducibility/scripts/data/prepare_variable_dataset.py`
- `battery_soh_reproducibility/scripts/modeling/constant_cross_dataset_transfer.py`
- `battery_soh_reproducibility/scripts/modeling/constant_partial_observation_3p6v.py`
- `battery_soh_reproducibility/scripts/modeling/constant_strict_lobo.py`
- `battery_soh_reproducibility/scripts/modeling/variable_strict_lobo.py`
- `battery_soh_reproducibility/scripts/reporting/generate_descriptive_figures.py`
- `battery_soh_reproducibility/scripts/reporting/generate_manuscript_tables.py`
- `battery_soh_reproducibility/scripts/run_all.py`
- `battery_soh_reproducibility/scripts/validation/validate_reproduced_results.py`
- `battery_soh_reproducibility/src/battery_soh_repro/__init__.py`
- `battery_soh_reproducibility/src/battery_soh_repro/data_pipeline.py`
- `battery_soh_reproducibility/src/battery_soh_repro/feature_policy.py`
- `battery_soh_reproducibility/src/battery_soh_repro/utils.py`
- `battery_soh_reproducibility/tests/test_feature_contracts.py`
- `battery_soh_reproducibility/tests/test_preliminary_calibration.py`
- `battery_soh_reproducibility/tests/test_release_contracts.py`
- `battery_soh_reproducibility/tests/test_static_repository.py`

## Modified original files

- `README.md`
- `battery_soh_reproducibility/CODE_PROVENANCE.md`
- `battery_soh_reproducibility/MANUSCRIPT_CODE_MAP.md`
- `battery_soh_reproducibility/REPRODUCIBILITY_LIMITATIONS.md`

## New files (excluding this inventory)

- `battery_soh_reproducibility/VERIFICATION.md`
- `battery_soh_reproducibility/requirements-audits.txt`
- `battery_soh_reproducibility/requirements-sequence.txt`
- `battery_soh_reproducibility/scripts/analysis/battery_balanced_ranking.py`
- `battery_soh_reproducibility/scripts/analysis/common_subset_baselines.py`
- `battery_soh_reproducibility/scripts/analysis/full_discharge_proxy_audit.py`
- `battery_soh_reproducibility/scripts/analysis/lowpass_feature_diagnostic.py`
- `battery_soh_reproducibility/scripts/analysis/paired_primary_uncertainty.py`
- `battery_soh_reproducibility/scripts/analysis/variable_temperature_feasibility.py`
- `battery_soh_reproducibility/scripts/data/build_quality_screened_dataset.py`
- `battery_soh_reproducibility/scripts/data/calce_protocol_completeness.py`
- `battery_soh_reproducibility/scripts/modeling/age_only_hgb.py`
- `battery_soh_reproducibility/scripts/modeling/censored_observation_lobo.py`
- `battery_soh_reproducibility/scripts/modeling/electrical_linear_ridge.py`
- `battery_soh_reproducibility/scripts/modeling/fixed_600s_archived.py`
- `battery_soh_reproducibility/scripts/modeling/quality_screened_lobo.py`
- `battery_soh_reproducibility/scripts/modeling/sequence_3p6.py`
- `battery_soh_reproducibility/scripts/modeling/source_only_electrical_hgb.py`
- `battery_soh_reproducibility/scripts/modeling/transfer_age_mean_baselines.py`
- `battery_soh_reproducibility/scripts/reporting/build_comparison_metrics.py`
- `battery_soh_reproducibility/scripts/reporting/generate_new_figures.py`
- `battery_soh_reproducibility/scripts/run_extensions.py`
- `battery_soh_reproducibility/src/battery_soh_repro/baselines.py`
- `battery_soh_reproducibility/src/battery_soh_repro/censoring.py`
- `battery_soh_reproducibility/src/battery_soh_repro/experiment_utils.py`
- `battery_soh_reproducibility/src/battery_soh_repro/protocol_qc.py`
- `battery_soh_reproducibility/src/battery_soh_repro/sequence_models.py`
- `battery_soh_reproducibility/tests/test_extensions.py`

## Verification scope

Test suite: 19 passed. The source-code package contains no `.ipynb`, Colab API calls, Google Drive mount assumptions, local cache directories, or the large raw-data archive. Full experiment reproduction requires raw inputs and selected historical result records. See `battery_soh_reproducibility/VERIFICATION.md` for which numerical outputs were re-checked.
