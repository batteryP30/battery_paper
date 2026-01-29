# battery_paper

## Overview

This project presents a data-driven machine learning framework for estimating the State of Health (SoH) of lithium-ion batteries using cycle-level features extracted from charge–discharge data. The approach is designed to generalize across batteries by leveraging degradation-relevant electrical signatures and strict validation strategies.

The framework is evaluated on CS2 prismatic lithium-ion cells from the CALCE dataset under both constant-discharge and variable-discharge protocols. A strong emphasis is placed on feature engineering, feature selection stability, and battery-level generalization rather than fitting individual cell trajectories.

---

## Methodology Summary

Raw time-series battery cycling data are transformed into a cycle-level dataset in which each row represents one complete charge–discharge cycle. State of Health (SoH) is computed on a per-cycle basis from discharge capacity using current integration.

A total of 109 cycle-level features are engineered to capture degradation behavior, including:
- Voltage derivative and dynamic resistance proxy features
- Charge and discharge capacity, energy, duration, and voltage statistics
- Coulombic and energy efficiency metrics
- Step-specific diagnostics for constant-current and constant-voltage phases
- Incremental capacity (IC) curve statistics and peak-based features

To reduce redundancy and improve generalization, feature selection is applied only on training data using a two-stage pipeline:
1. Correlation pruning based on absolute Spearman correlation
2. Stability selection using grouped cross-validation by battery ID and permutation importance

Two regression models are evaluated: a single regularized baseline and a stacked ensemble combining multiple tree-based learners.

---

## Evaluation Strategy

Model performance is assessed using two complementary evaluation settings:

**Battery Holdout Evaluation**  
One battery is completely held out for testing while all remaining batteries are used for training. Feature selection is performed only on training batteries, and the selected feature set is applied unchanged to the held-out battery. This setting evaluates the model’s ability to generalize to unseen cells.

**Early-Life Evaluation**  
For each battery, the first 30% of cycles are used for training and the remaining cycles are used for testing. Additional features such as cycle index and normalized cycle progression are included to encode early-life position within the aging trajectory. This setting evaluates the model’s ability to forecast future degradation from limited early-life data.

Model performance is measured using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

