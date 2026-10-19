"""Regenerate the manuscript's preliminary feature-selection heatmaps.

The historical threshold-grid execution script was not retained with the
available project sources. The matrices below are the diagnostic values shown
in the manuscript development figures and are retained solely to regenerate
those development-stage heatmaps. They are not final-model results
and are not used by the nested HGB pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CORRELATION_THRESHOLDS = [0.85, 0.90, 0.93, 0.95, 0.98]
STABILITY_THRESHOLDS = [0.60, 0.70, 0.80, 0.90]

CONSTANT_RMSE = np.array([
    [2.289, 1.801, 1.724, 1.752, 1.922],
    [2.385, 2.017, 1.810, 1.889, 1.932],
    [2.385, 2.017, 1.810, 1.889, 1.932],
    [2.448, 2.022, 1.819, 2.176, 2.045],
])
CONSTANT_FEATURE_COUNT = np.array([
    [13.1, 12.6, 12.0, 11.1, 12.3],
    [6.9, 8.5, 7.5, 6.9, 7.4],
    [6.9, 8.5, 7.5, 6.9, 7.4],
    [4.7, 5.2, 4.7, 4.2, 3.9],
])
VARIABLE_RMSE = np.array([
    [2.070, 1.915, 1.821, 2.312, 2.324],
    [2.009, 1.899, 1.862, 2.297, 2.241],
    [2.009, 1.899, 1.862, 2.297, 2.241],
    [1.949, 3.131, 3.138, 3.308, 3.019],
])
VARIABLE_FEATURE_COUNT = np.array([
    [14.7, 15.0, 14.3, 14.2, 15.2],
    [10.8, 8.2, 8.5, 9.2, 7.7],
    [10.8, 8.2, 8.5, 9.2, 7.7],
    [5.7, 5.0, 5.2, 4.8, 2.3],
])


def _save_heatmap(values: np.ndarray, title: str, colorbar_label: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = ax.imshow(values, aspect="auto")
    ax.set_xticks(range(len(CORRELATION_THRESHOLDS)),
                  [f"{x:.2f}" for x in CORRELATION_THRESHOLDS])
    ax.set_yticks(range(len(STABILITY_THRESHOLDS)),
                  [f"{x:.1f}" for x in STABILITY_THRESHOLDS])
    ax.set_xlabel("Correlation threshold")
    ax.set_ylabel("Stability threshold")
    ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, f"{values[row, col]:.3f}" if "RMSE" in title
                    else f"{values[row, col]:.1f}",
                    ha="center", va="center")
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        ("const_rmse.png", CONSTANT_RMSE, "Constant protocol preliminary RMSE", "RMSE"),
        ("const_feat.png", CONSTANT_FEATURE_COUNT, "Constant protocol selected features", "Mean feature count"),
        ("var_rmse.png", VARIABLE_RMSE, "Variable protocol preliminary RMSE", "RMSE"),
        ("var_feat.png", VARIABLE_FEATURE_COUNT, "Variable protocol selected features", "Mean feature count"),
    ]
    paths = []
    for filename, values, title, label in outputs:
        path = output_dir / filename
        _save_heatmap(values, title, label, path)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/manuscript_figures"))
    args = parser.parse_args()
    for path in run(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
