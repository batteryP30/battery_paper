# Battery-level lithium-ion SoH estimation — research code

Local Python implementation accompanying **Battery-Level and Cross-Dataset Generalization of Data-Driven Lithium-Ion Battery State-of-Health Estimation**.

This revision **preserves the original experimental code** and adds source-code implementations of the later battery-held-out, partial-observation, sequence-model, protocol-completeness, and diagnostic studies. It contains **no Jupyter notebooks and no Google Colab runtime dependencies**.

## Install

The installable project is in `battery_soh_reproducibility/`:

```bash
cd battery_soh_reproducibility
python -m venv .venv
source .venv/bin/activate                   # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -r requirements-dev.txt
pytest -q
```

For neural-model training, additionally run `pip install -r requirements-sequence.txt`; for the temperature audit, `pip install -r requirements-audits.txt`.

## Run the published primary workflow

Supply the **original raw battery ZIP**, including CALCE workbooks and the NASA CSV exports used for the study. This ~708 MB archive is not included in source control.

```bash
python scripts/run_all.py --raw-zip /data/collab.zip
```

This retains the historical 4,590 constant-protocol observations and 165 variable-protocol independent observations. It does **not** apply the later CALCE protocol screen to the primary analysis.

## Run additional experiments locally

From the same installable project directory:

```bash
# Target-integral proxy audit from already derived constant observations
python scripts/run_extensions.py --stages proxy

# Nonlinear age baseline and source-only transfer controls
python scripts/run_extensions.py --stages age transfer_baselines common_subset

# Post hoc CALCE screen and screened-data reconstruction
python scripts/run_extensions.py --raw-zip /data/collab.zip --stages calce_qc screened_features

# Nested quality-screened HGB on 4,501 retained cycles
python scripts/run_extensions.py --stages screened_hgb

# Leakage-safe censored current-cycle experiments
python scripts/run_extensions.py --raw-zip /data/collab.zip --stages censored_600s censored_3p6

# Sequence benchmarks (requires PyTorch; CPU supported, GPU recommended)
python scripts/run_extensions.py --raw-zip /data/collab.zip --stages sequence_prepare
python scripts/run_extensions.py --stages sequence_train --device cpu

# Variable-protocol temperature feasibility
python scripts/run_extensions.py --raw-zip /data/collab.zip --stages temperature
```

Run stages in dependency order. Only explicitly selected stages run; no hidden network calls or uploading. Large experiments use substantial time/memory. The raw source ZIP and derived CSV results remain in user-controlled paths.

The exact archived Step 11L 600-s experiment also has a local entry point:

```bash
python scripts/modeling/fixed_600s_archived.py \
  --raw-zip /data/collab.zip \
  --frozen-data outputs/constant_dataset/constant_observations.csv \
  --frozen-features outputs/constant_dataset/constant_strict_feature_names.csv \
  --frozen-predictions outputs/constant_hgb/constant_hgb_predictions.csv \
  --output-dir outputs/fixed600_archived
```

`fixed_600s_archived.py` preserves the saved Step 11L scientific routines with hosted-runtime I/O replaced by local paths. `censored_observation_lobo.py` is the newer generalized local implementation; outputs from it should be checked against the archived Step 11G/11L records before claiming exact numerical reproduction.

## Rebuild the seven manuscript figures

```bash
python scripts/reporting/generate_new_figures.py \
  --original-predictions outputs/constant_hgb/constant_hgb_predictions.csv \
  --screened-predictions outputs/quality_screened_lobo/screened_STRICT_ELECTRICAL_PLUS_AGE_HGB_predictions.csv \
  --output-dir outputs/manuscript_figures
```

The comparison plots require independent metric CSVs from the corresponding experiments:

```bash
python scripts/reporting/build_comparison_metrics.py
python scripts/reporting/generate_new_figures.py \
  --comparison-metrics outputs/manuscript_figures/figure_comparison_metrics.csv \
  --output-dir outputs/manuscript_figures
```

The generated files use the exact `Fig1_...png` through `Fig7_...png` names in the submitted manuscript. No metric is estimated or hardcoded just to make a plot.

## Scientific interpretation

- The historical "strict" full-discharge features exclude explicit target integrals but **are not claimed statistically capacity-independent**. Full-discharge slopes can encode indirect reciprocal-capacity information.
- The 600-s and true-3.6-V studies compute current-cycle features **only** from the observed discharge prefix, and do not transfer post-boundary feature values.
- The CALCE charge/discharge screen is **post hoc**. A missing logger record is not proof of physical charging failure. Screened results supplement, rather than replace, the primary study.
- LSTM and Transformer models use 96-point raw discharge-prefix sequences while HGB uses engineered indicators; their differences do not isolate architecture alone.
- Persistence uses a prior measured reference target from the same test battery. Its first observation is excluded from common-subset comparisons.
- Original data, MATLAB/Excel workbook examples, NASA native-to-CSV conversions, and original checkpoints are not bundled or invented.

See `battery_soh_reproducibility/MANUSCRIPT_CODE_MAP.md`, `CODE_PROVENANCE.md`, `REPRODUCIBILITY_LIMITATIONS.md`, and `VERIFICATION.md` for exact provenance and validation status. A file-by-file preserved/modified/added list is provided in `FILE_CHANGE_REPORT.csv`.
