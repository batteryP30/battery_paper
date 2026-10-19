"""Execute the complete battery-SoH workflow in manuscript order.

The raw dataset archive is the only required external input. The workflow writes
all derived datasets, model results, statistical analyses, tables, and figures
under the chosen output directory. Nothing in the repository depends on
pre-generated result files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def repository_root() -> Path:
    """Return the project root containing ``scripts`` and ``src``."""
    return Path(__file__).resolve().parents[1]


def _run(script: Path, *args: str) -> None:
    command = [sys.executable, str(script), *map(str, args)]
    print("\n" + "=" * 100)
    print("RUNNING:", " ".join(command))
    print("=" * 100)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-zip", type=Path, required=True,
                        help="ZIP containing the original CALCE and NASA datasets.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument(
        "--skip-measurement-robustness", action="store_true",
        help="Skip the expensive raw-perturbation study. All other stages still run.",
    )
    args = parser.parse_args()

    root = repository_root()
    scripts = root / "scripts"
    out = args.output_dir
    work = args.work_dir
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    # 1. Feature policy and corrected datasets / baselines.
    _run(scripts/"data/feature_admissibility_audit.py",
         "--raw-zip", args.raw_zip, "--output-dir", out/"feature_audit",
         "--work-dir", work/"feature_audit")
    _run(scripts/"data/nasa_rw_target_provenance.py",
         "--raw-zip", args.raw_zip, "--output-dir", out/"nasa_rw_provenance",
         "--work-dir", work/"nasa_rw_provenance")
    _run(scripts/"data/prepare_variable_dataset.py",
         "--raw-zip", args.raw_zip, "--output-dir", out/"variable_dataset",
         "--work-dir", work/"variable_dataset")
    _run(scripts/"data/prepare_constant_dataset.py",
         "--raw-zip", args.raw_zip, "--output-dir", out/"constant_dataset",
         "--work-dir", work/"constant_dataset")

    # 2. Strict nested LOBO and constant cross-dataset transfer.
    _run(scripts/"modeling/variable_strict_lobo.py",
         "--data", out/"variable_dataset/variable_independent_observations.csv",
         "--baselines", out/"variable_dataset/variable_baseline_macro.csv",
         "--output-dir", out/"variable_hgb")
    _run(scripts/"modeling/constant_strict_lobo.py",
         "--data", out/"constant_dataset/constant_observations.csv",
         "--features", out/"constant_dataset/constant_strict_feature_names.csv",
         "--baselines", out/"constant_dataset/constant_baseline_macro.csv",
         "--output-dir", out/"constant_hgb")
    _run(scripts/"modeling/constant_cross_dataset_transfer.py",
         "--data", out/"constant_dataset/constant_observations.csv",
         "--features", out/"constant_dataset/constant_strict_feature_names.csv",
         "--output-dir", out/"constant_transfer")

    # 3. Secondary 3.6 V information setting.
    _run(scripts/"modeling/constant_partial_observation_3p6v.py",
         "--raw-zip", args.raw_zip,
         "--constant-data", out/"constant_dataset/constant_observations.csv",
         "--strict-features", out/"constant_dataset/constant_strict_feature_names.csv",
         "--strict-macro", out/"constant_hgb/constant_hgb_macro.csv",
         "--output-dir", out/"constant_3p6",
         "--work-dir", work/"constant_3p6")

    # 4. Battery bootstrap and strict feature-family ablations.
    _run(scripts/"analysis/bootstrap_and_ablations.py",
         "--constant-data", out/"constant_dataset/constant_observations.csv",
         "--strict-features", out/"constant_dataset/constant_strict_feature_names.csv",
         "--constant-macro", out/"constant_hgb/constant_hgb_macro.csv",
         "--variable-per-battery", out/"variable_hgb/variable_hgb_per_battery.csv",
         "--constant-per-battery", out/"constant_hgb/constant_hgb_per_battery.csv",
         "--transfer-per-battery", out/"constant_transfer/constant_transfer_per_battery.csv",
         "--partial-per-battery", out/"constant_3p6/constant_3p6_per_battery.csv",
         "--output-dir", out/"statistical_analysis")

    # 5. Raw-measurement perturbations with clean reconstruction validation.
    if not args.skip_measurement_robustness:
        _run(scripts/"analysis/measurement_robustness.py",
             "--raw-zip", args.raw_zip,
             "--constant-data", out/"constant_dataset/constant_observations.csv",
             "--selected-features", out/"constant_hgb/constant_hgb_selected_features.csv",
             "--fold-metrics", out/"constant_hgb/constant_hgb_per_battery.csv",
             "--output-dir", out/"measurement_robustness",
             "--work-dir", work/"measurement_robustness")

    # 6. Manuscript figures and tables.
    _run(scripts/"analysis/preliminary_threshold_calibration.py",
         "--output-dir", out/"manuscript_figures")
    _run(scripts/"reporting/generate_descriptive_figures.py",
         "--raw-zip", args.raw_zip,
         "--constant-data", out/"constant_dataset/constant_observations.csv",
         "--output-dir", out/"manuscript_figures",
         "--work-dir", work/"manuscript_figures")

    if not args.skip_measurement_robustness:
        _run(scripts/"reporting/generate_manuscript_tables.py",
             "--outputs-root", out, "--output-dir", out/"manuscript_tables")
        _run(scripts/"validation/validate_reproduced_results.py",
             "--outputs-root", out)

    print("\nWorkflow complete.")


if __name__ == "__main__":
    main()
