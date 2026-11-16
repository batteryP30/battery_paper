#!/usr/bin/env python3
"""Step 11N — training-cohort-matched 3.6-V baselines.

Re-train the equal-battery training-mean, age-only linear regression, and
age-only HistGradientBoostingRegressor on the EXACT eligible 4,589 constant-
protocol cycles that were used in the true first-3.6-V censored HGB study.

The original censored HGB predictions are used unchanged. Compare all five
models on the same held-out battery/cycle keys and compute battery-paired
bootstrap uncertainty (20,000 resamples; no retraining in the bootstrap).

No raw archive or previous code execution is required: input is a compact,
traceable bundle of archived Step 5 age/target columns and the Step 11G
censored predictions. Notebook edition embeds that bundle automatically.

Outputs: CSV summaries/predictions/intervals + report + ZIP for review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

RUN_TAG = "step11n-matched3p6-v1"
TARGET = "Calculated_SOH(%)"
KEYS = ["battery_id", "Cycle_Index"]
AGE_FEATURES = ["Cycle_Age", "Log_Cycle_Age"]
LINEAR_FEATURES = ["Cycle_Age"]  # Match archived AgeOnlyLinear: linear on observed age only.
BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 42
SEED = 42
MODEL_CENSORED_E = "TRUE_3P6V_CENSORED_ELECTRICAL_HGB"
MODEL_CENSORED_EA = "TRUE_3P6V_CENSORED_ELECTRICAL_PLUS_AGE_HGB"
BASELINE_MEAN = "TRAINING_MEAN_EQUAL_BATTERY"
BASELINE_LINEAR = "AGE_ONLY_LINEAR"
BASELINE_HGB = "AGE_ONLY_HGB_TUNED"
HGB_CONFIGS = {
    "H0": dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
               min_samples_leaf=5, l2_regularization=1.0),
    "H1": dict(learning_rate=0.04, max_iter=500, max_leaf_nodes=15,
               min_samples_leaf=10, l2_regularization=2.0),
    "H2": dict(learning_rate=0.03, max_iter=700, max_leaf_nodes=31,
               min_samples_leaf=10, l2_regularization=3.0),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def integer_cycles(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Missing/nonfinite Cycle_Index")
    if not np.allclose(numeric.to_numpy(dtype=float), np.rint(numeric), rtol=0, atol=1e-6):
        raise ValueError("Non-integer Cycle_Index detected")
    return numeric.round().astype("int64")


def require_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    duplicate = frame.duplicated(columns, keep=False)
    if duplicate.any():
        print(frame.loc[duplicate, columns].head(8).to_string(index=False))
        raise ValueError(f"Duplicate keys in {label}: {columns}")


def load_inputs(bundle_path: Path, strict_counts: bool = True):
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)
    with zipfile.ZipFile(bundle_path) as archive:
        expected = {"study_age_and_targets.csv", "true_3p6_censored_hgb_predictions.csv"}
        if not expected.issubset(archive.namelist()):
            raise ValueError(f"Input bundle must contain {sorted(expected)}")
        study = pd.read_csv(io.BytesIO(archive.read("study_age_and_targets.csv")))
        old = pd.read_csv(io.BytesIO(archive.read("true_3p6_censored_hgb_predictions.csv")))

    study_needed = set(KEYS + AGE_FEATURES + [TARGET, "source_dataset"])
    pred_needed = set(KEYS + ["Formulation", "Prediction_Raw", TARGET,
                               "Cycle_Age", "source_dataset"])
    if not study_needed.issubset(study.columns):
        raise ValueError(f"Missing study columns: {sorted(study_needed - set(study.columns))}")
    if not pred_needed.issubset(old.columns):
        raise ValueError(f"Missing prediction columns: {sorted(pred_needed - set(old.columns))}")

    for table in (study, old):
        table["battery_id"] = table["battery_id"].astype(str).str.strip()
        table["Cycle_Index"] = integer_cycles(table["Cycle_Index"])
        for key in (TARGET, "Cycle_Age"):
            table[key] = pd.to_numeric(table[key], errors="raise")
        if not np.isfinite(table[[TARGET, "Cycle_Age"]].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite age/target in input")
    for key in AGE_FEATURES:
        study[key] = pd.to_numeric(study[key], errors="raise")
    if not np.isfinite(study[AGE_FEATURES].to_numpy(float)).all():
        raise ValueError("Nonfinite age features in study")
    old["Prediction_Raw"] = pd.to_numeric(old["Prediction_Raw"], errors="raise")
    if not np.isfinite(old["Prediction_Raw"]).all():
        raise ValueError("Nonfinite archived HGB prediction")
    require_unique(study, KEYS, "archived Step 5 study")
    require_unique(old, ["Formulation"] + KEYS, "archived Step 11G predictions")

    names = set(old.Formulation.unique())
    if names != {MODEL_CENSORED_E, MODEL_CENSORED_EA}:
        raise ValueError(f"Unexpected archived censored HGB formulations: {names}")
    keys_by_model = {
        name: set(map(tuple, g[KEYS].itertuples(index=False, name=None)))
        for name, g in old.groupby("Formulation")
    }
    common_keys = keys_by_model[MODEL_CENSORED_E]
    if common_keys != keys_by_model[MODEL_CENSORED_EA]:
        raise ValueError("Censored electrical and electrical-plus-age predictions have different test sets")
    all_keys = set(map(tuple, study[KEYS].itertuples(index=False, name=None)))
    if not common_keys.issubset(all_keys):
        raise ValueError("Archived censored predictions include keys absent from Step 5")
    excluded_keys = sorted(all_keys - common_keys)

    eligible = study.set_index(KEYS).loc[list(common_keys)].reset_index()
    eligible = eligible.sort_values(KEYS).reset_index(drop=True)
    eligible["Log_Cycle_Age"] = eligible["Log_Cycle_Age"].astype(float)
    if strict_counts:
        if len(study) != 4590 or len(common_keys) != 4589:
            raise ValueError(f"Expected 4590 study and 4589 censored observations; got {len(study)}, {len(common_keys)}")
        if excluded_keys != [("CS2_34", 91)]:
            raise ValueError(f"Unexpected censored eligibility exclusion: {excluded_keys}")
        if eligible.battery_id.nunique() != 10:
            raise ValueError("Expected 10 held-out batteries")

    merged = old.merge(
        eligible[KEYS + [TARGET, "Cycle_Age", "source_dataset"]],
        on=KEYS, validate="many_to_one", how="left", suffixes=("_HGB", "_Study"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Censored prediction-to-study matching failed")
    for c in [TARGET, "Cycle_Age"]:
        if not np.allclose(merged[f"{c}_HGB"], merged[f"{c}_Study"], rtol=0, atol=1e-8):
            raise ValueError(f"Archived HGB/study mismatch in {c}")
    if not (merged["source_dataset_HGB"].astype(str) == merged["source_dataset_Study"].astype(str)).all():
        raise ValueError("Source labels disagree for predictions and study")

    expected_scores = {
        MODEL_CENSORED_E: 2.282087480660201,
        MODEL_CENSORED_EA: 1.715946556127656,
    }
    for name, g in old.groupby("Formulation"):
        per = [float(np.sqrt(np.mean((v[TARGET] - v["Prediction_Raw"]) ** 2)))
               for _, v in g.groupby("battery_id")]
        mean_rmse = float(np.mean(per))
        if abs(mean_rmse - expected_scores[name]) > 1e-7:
            raise ValueError(f"Censored HGB reproduction gate FAILED: {name}: {mean_rmse}")
        print(f"Validated archived {name}: RMSE {mean_rmse:.6f}")

    print(f"Eligibility gate passed: {len(study)} archived cycles; {len(eligible)} censored-eligible; "
          f"{eligible.battery_id.nunique()} batteries; excluded {excluded_keys}")
    return eligible, old, excluded_keys


def equal_battery_weights(battery_ids: pd.Series) -> np.ndarray:
    ids = battery_ids.astype(str)
    n_per = ids.value_counts()
    w = ids.map(lambda x: 1.0 / n_per[x]).to_numpy(float)
    return w / w.mean()


def fit_linear(train: pd.DataFrame):
    model = LinearRegression()
    model.fit(train[LINEAR_FEATURES].to_numpy(float),
              train[TARGET].to_numpy(float),
              sample_weight=equal_battery_weights(train.battery_id))
    return model


def fit_age_hgb(train: pd.DataFrame, config: str):
    model = HistGradientBoostingRegressor(
        loss="squared_error", early_stopping=False, random_state=SEED,
        **HGB_CONFIGS[config],
    )
    model.fit(train[AGE_FEATURES].to_numpy(float),
              train[TARGET].to_numpy(float),
              sample_weight=equal_battery_weights(train.battery_id))
    return model


def tune_hgb_on_inner_batteries(train: pd.DataFrame):
    """Exact archived age-HGB strategy: 3 configs; inner LOBO by battery."""
    values = []
    for name in sorted(HGB_CONFIGS):
        held_rmse = []
        for held in sorted(train.battery_id.unique()):
            tr = train.loc[train.battery_id.ne(held)]
            va = train.loc[train.battery_id.eq(held)]
            candidate = fit_age_hgb(tr, name)
            prediction = candidate.predict(va[AGE_FEATURES].to_numpy(float))
            rmse = float(np.sqrt(np.mean((va[TARGET].to_numpy(float) - prediction) ** 2)))
            held_rmse.append(rmse)
        values.append({"Config": name, "Inner_EqualBattery_RMSE": float(np.mean(held_rmse)),
                       "Inner_Batteries": len(held_rmse)})
    winner = min(values, key=lambda z: (z["Inner_EqualBattery_RMSE"], z["Config"]))
    return winner["Config"], values


def read_checkpoint(path: Path, expected_holdout: str, expected_cycles: set[int]):
    if not path.exists():
        return None
    cached = pd.read_csv(path)
    needed = {"Model", "battery_id", "Cycle_Index", TARGET, "Prediction_Raw", "HGB_Config"}
    if not needed.issubset(cached.columns):
        raise ValueError(f"Corrupt checkpoint {path}: missing columns")
    cached["Cycle_Index"] = integer_cycles(cached.Cycle_Index)
    if cached.battery_id.nunique() != 1 or cached.battery_id.iloc[0] != expected_holdout:
        raise ValueError(f"Checkpoint battery mismatch: {path}")
    for model, g in cached.groupby("Model"):
        if set(g.Cycle_Index.tolist()) != expected_cycles:
            raise ValueError(f"Incomplete checkpoint {path}, model {model}")
    if set(cached.Model.unique()) != {BASELINE_MEAN, BASELINE_LINEAR, BASELINE_HGB}:
        raise ValueError(f"Incomplete model set in {path}")
    require_unique(cached, ["Model"] + KEYS, "checkpoint")
    return cached


def run_training(eligible: pd.DataFrame, output_dir: Path, bundle_sha: str):
    all_rows = []
    selection_rows = []
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps({
        "version": RUN_TAG, "configs": HGB_CONFIGS, "age_features": AGE_FEATURES,
        "linear_features": LINEAR_FEATURES,
        "seed": SEED, "loss": "squared_error", "early_stopping": False,
        "train_population": "3p6 4589", "evaluation": "battery-mean-RMSE",
    }, sort_keys=True).encode()).hexdigest()[:12]

    batteries = sorted(eligible.battery_id.unique())
    for i, heldout in enumerate(batteries, 1):
        start = time.monotonic()
        tr = eligible.loc[eligible.battery_id.ne(heldout)].copy()
        te = eligible.loc[eligible.battery_id.eq(heldout)].copy()
        if tr.battery_id.nunique() != 9:
            raise ValueError(f"Expected nine training batteries for {heldout}")
        checkpoint = checkpoint_dir / f"fold_{heldout}_{bundle_sha[:12]}_{signature}.csv"
        expected_cycles = set(te.Cycle_Index.tolist())
        cached = read_checkpoint(checkpoint, heldout, expected_cycles)
        if cached is not None:
            print(f"[{i}/{len(batteries)}] {heldout}: using validated Drive checkpoint")
            all_rows.append(cached)
            pick = cached.loc[cached.Model.eq(BASELINE_HGB), "HGB_Config"].iloc[0]
            selection_rows.append({"Holdout_Battery": heldout, "Best_HGB_Config": pick,
                                   "Cached": True, "Training_Batteries": 9,
                                   "N_Test": len(te)})
            continue

        print(f"[{i}/{len(batteries)}] {heldout}: tuning 3 age-only HGB configurations "
              f"on 9 training-battery holdouts ...", flush=True)
        train_means = tr.groupby("battery_id")[TARGET].mean()
        mean_prediction = np.full(len(te), train_means.mean(), dtype=float)
        linear_model = fit_linear(tr)
        linear_prediction = linear_model.predict(te[LINEAR_FEATURES].to_numpy(float))
        config, inner_scores = tune_hgb_on_inner_batteries(tr)
        age_hgb_model = fit_age_hgb(tr, config)
        age_hgb_prediction = age_hgb_model.predict(te[AGE_FEATURES].to_numpy(float))
        selection_rows.append({"Holdout_Battery": heldout, "Best_HGB_Config": config,
                               "Cached": False, "Training_Batteries": 9,
                               "N_Test": len(te),
                               **{f"Inner_{x['Config']}": x["Inner_EqualBattery_RMSE"] for x in inner_scores}})

        rows = []
        for label, values in [
            (BASELINE_MEAN, mean_prediction),
            (BASELINE_LINEAR, linear_prediction),
            (BASELINE_HGB, age_hgb_prediction),
        ]:
            for (_, row), yhat in zip(te.iterrows(), values):
                rows.append({"Model": label, "battery_id": heldout,
                             "source_dataset": str(row.source_dataset),
                             "Cycle_Index": int(row.Cycle_Index),
                             "Cycle_Age": float(row.Cycle_Age),
                             TARGET: float(row[TARGET]),
                             "Prediction_Raw": float(yhat),
                             "HGB_Config": config if label == BASELINE_HGB else ""})
        result = pd.DataFrame(rows)
        temp = checkpoint.with_suffix(".tmp")
        result.to_csv(temp, index=False)
        os.replace(temp, checkpoint)
        all_rows.append(result)
        print(f"[{i}/{len(batteries)}] {heldout} completed; n={len(te)}, "
              f"HGB={config}; {time.monotonic() - start:.1f}s; saved checkpoint")

    joined = pd.concat(all_rows, ignore_index=True)
    require_unique(joined, ["Model"] + KEYS, "matched baseline predictions")
    return joined, pd.DataFrame(selection_rows)


def results_and_uncertainty(eligible, archived_hgb, baseline, out: Path, bundle_sha: str,
                            bootstrap_reps: int = BOOTSTRAP_REPS, bootstrap_seed: int = BOOTSTRAP_SEED):
    hgb = archived_hgb.rename(columns={"Formulation": "Model"})
    long = pd.concat([
        baseline[["Model", "battery_id", "source_dataset", "Cycle_Index", "Cycle_Age", TARGET, "Prediction_Raw"]],
        hgb[["Model", "battery_id", "source_dataset", "Cycle_Index", "Cycle_Age", TARGET, "Prediction_Raw"]],
    ], ignore_index=True)
    require_unique(long, ["Model"] + KEYS, "five-model prediction output")
    models = [BASELINE_MEAN, BASELINE_LINEAR, BASELINE_HGB, MODEL_CENSORED_E, MODEL_CENSORED_EA]
    total = len(eligible)
    for name in models:
        g = long.loc[long.Model.eq(name)]
        if len(g) != total or g.battery_id.nunique() != 10:
            raise RuntimeError(f"Unpaired/incomplete model results: {name}: {len(g)}")
        require_unique(g, KEYS, f"test cycle predictions for {name}")
        merged = eligible[KEYS + [TARGET]].merge(g[KEYS + [TARGET]], on=KEYS,
                    how="left", validate="one_to_one", indicator=True,
                    suffixes=("_Original", "_Predictions"))
        if not merged._merge.eq("both").all() or not np.allclose(
            merged[f"{TARGET}_Original"], merged[f"{TARGET}_Predictions"], rtol=0, atol=1e-8):
            raise RuntimeError(f"Unmatched target data for {name}")

    long["Error_PredMinusMeasured"] = long.Prediction_Raw - long[TARGET]
    long.sort_values(["Model", "battery_id", "Cycle_Index"], inplace=True)
    per = []
    for (name, battery), g in long.groupby(["Model", "battery_id"], sort=True):
        err = g.Error_PredMinusMeasured.to_numpy(float)
        per.append({"Model": name, "battery_id": battery,
                    "source_dataset": str(g.source_dataset.iloc[0]),
                    "N_Test": len(g), "RMSE": float(np.sqrt(np.mean(err**2))),
                    "MAE": float(np.mean(np.abs(err)))})
    per = pd.DataFrame(per)
    summary = per.groupby("Model", as_index=False).agg(
        Batteries=("battery_id", "nunique"), Observations=("N_Test", "sum"),
        EqualBattery_RMSE=("RMSE", "mean"), EqualBattery_MAE=("MAE", "mean"))
    if len(summary) != 5 or not summary.Observations.eq(4589).all():
        raise RuntimeError("Model summary incomplete")
    baseline_reference = per.pivot(index="battery_id", columns="Model", values="RMSE")
    rng = np.random.default_rng(bootstrap_seed)
    n = len(baseline_reference)
    # The same draws are shared across contrasts; no model is retrained.
    indices = rng.integers(0, n, size=(bootstrap_reps, n))
    contrast_rows = []
    for censored in [MODEL_CENSORED_E, MODEL_CENSORED_EA]:
        for reference in [BASELINE_MEAN, BASELINE_LINEAR, BASELINE_HGB]:
            deltas = (baseline_reference[censored] - baseline_reference[reference]).to_numpy(float)
            estimates = deltas[indices].mean(axis=1)
            lo, hi = np.percentile(estimates, [2.5, 97.5])
            contrast_rows.append({
                "Censored_Model": censored, "Baseline": reference,
                "Mean_Paired_RMSE_Difference_Censored_Minus_Baseline": float(deltas.mean()),
                "CI95_Lower": float(lo), "CI95_Upper": float(hi),
                "Batteries": n, "Bootstrap_Repetitions": bootstrap_reps,
                "Bootstrap_Seed": bootstrap_seed,
                "CI_Excludes_Zero": bool((hi < 0) or (lo > 0)),
            })
    contrasts = pd.DataFrame(contrast_rows)
    long.to_csv(out / "Step11N_AllFive_Model_Predictions.csv", index=False)
    per.to_csv(out / "Step11N_FiveModel_PerBattery.csv", index=False)
    summary.to_csv(out / "Step11N_FiveModel_EqualBattery_Summary.csv", index=False)
    contrasts.to_csv(out / "Step11N_Paired_Battery_Bootstrap_CIs.csv", index=False)

    metadata = {
        "run": RUN_TAG,
        "study": "true first downward 3.6-V constant-protocol discharge boundary",
        "input_bundle_sha256": bundle_sha,
        "eligibility_count": 4589,
        "excluded_vs_original_4590": [{"battery_id": "CS2_34", "Cycle_Index": 91}],
        "target_based_scope": "original 70-100% SoH study subset; NOT a target-unrestricted deployment test",
        "baseline_training_subset": "All 9 non-held-out batteries, 3.6V-eligible cycles only",
        "test_subset": "The retained 3.6V-eligible cycles for the held-out battery",
        "hgb_age_covariates": AGE_FEATURES,
        "linear_age_covariates": LINEAR_FEATURES,
        "training_mean": "equal arithmetic mean of 9 training-battery mean SoHs",
        "linear": "battery-weighted sklearn LinearRegression; no clipping",
        "age_hgb": "training-only inner battery LOBO among H0/H1/H2; no electrical inputs",
        "original_hgb": "frozen TRUE_3P6V_CENSORED Step11G predictions; no retraining",
        "metric": "arithmetic mean across battery-specific RMSEs / MAEs; NOT pooled cycle RMSE",
        "paired_uncertainty": "20,000 battery resamples of the 10 per-battery RMSE differences; no refits",
        "python": sys.version,
        "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "platform": platform.platform(),
    }
    (out / "Step11N_Reproducibility_Manifest.json").write_text(json.dumps(metadata, indent=2))
    lines = [
        "Step 11N: 3.6 V censored observation — matched-training cohort baseline experiment",
        "",
        "All models evaluated on the same 4,589 held-out battery/cycle observations.",
        "The baseline models were retrained on the same 3.6 V-eligible training cycle population.",
        "Their inputs are only age or a constant. HGB 3.6 V predictions are frozen from Step 11G.",
        "This study is still conditional on the originally selected 70-100% SoH range.",
        "",
        summary.sort_values("EqualBattery_RMSE").to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "", "Paired differences: censored electrical HGB minus indicated baseline (negative = better):",
        contrasts.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "", "The battery bootstrap resamples 10 fixed held-out battery errors without refitting."
    ]
    (out / "Step11N_Reviewer_Ready_Results.txt").write_text("\n".join(lines) + "\n")
    return summary, contrasts


def run(bundle: Path, output_dir: Path, bootstrap_reps: int = BOOTSTRAP_REPS,
        bootstrap_seed: int = BOOTSTRAP_SEED, verify_only: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    input_sha = sha256(bundle)
    print(f"Step 11N | data bundle SHA256 {input_sha}")
    eligible, archived_hgb, excluded = load_inputs(bundle)
    eligible.to_csv(output_dir / "Step11N_3p6_Eligible_Population.csv", index=False)
    if verify_only:
        print("INPUT VERIFICATION COMPLETE; training not started.")
        return
    baseline, selection = run_training(eligible, output_dir, input_sha)
    selection.to_csv(output_dir / "Step11N_InnerAgeHGB_Selection.csv", index=False)
    summary, contrasts = results_and_uncertainty(eligible, archived_hgb, baseline,
                                                  output_dir, input_sha,
                                                  bootstrap_reps, bootstrap_seed)
    print("\nMATCHED 3.6 V EXPERIMENT COMPLETED.\n")
    print(summary.to_string(index=False, float_format=lambda x:f"{x:.6f}"))
    print("\nPaired RMSE differences (Censored model minus baseline):")
    print(contrasts.to_string(index=False, float_format=lambda x:f"{x:.6f}"))
    final_zip = output_dir / "Step11N_Matched_3p6V_Results.zip"
    files = sorted(p for p in output_dir.iterdir() if p.is_file() and p.name != final_zip.name)
    with zipfile.ZipFile(final_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=8) as z:
        for f in files:
            z.write(f, arcname=f.name)
    print(f"\nRESULTS: {final_zip}")
    print("Upload the results ZIP (or the summary and per-battery CSVs) to this chat.")
    return final_zip


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-bundle", type=Path, required=True,
                   help="ZIP with archived 4590-cycle Step 5 ages and 4589-cycle Step 11G predictions")
    p.add_argument("--output-dir", type=Path, default=Path("step11n_3p6_matched_baselines"))
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    run(a.data_bundle, a.output_dir, a.bootstrap_reps, a.bootstrap_seed, a.verify_only)


if __name__ == "__main__":
    cli()
