# Verification performed for this repository update

## Directly executed checks

- Original test suite: **11 passed** when `src` was on the Python import path before modifications.
- Extended suite: **19 passed**, including CALCE CV/cutoff screening, actual-boundary censoring, per-step five-sample filtering, pairing guards, bootstrap unit, and optional PyTorch model output shapes.
- All added Python source files compiled successfully using Python 3.13 `compileall` (does not require raw data).
- Source-to-figure smoke test successfully regenerated original Figures 1 and 2 from supplied frozen held-out predictions, without editing source observations.
- The full-discharge proxy association audit on the supplied frozen Step 5 CSV reproduced median absolute within-battery Spearman **0.963198** for `LEGACY__Discharge_Step7_CC_slope` versus reciprocal capacity.
- Age-only HGB variable-protocol LOBO completed with six batteries and 165 observations; obtained equal-battery RMSE **2.382257** and MAE **1.820588**, matching the submitted manuscript's rounded 2.382 and 1.821.
- Common-subset variable baselines reproduced 159 observations and RMSEs: age-only linear **1.973513**, age-only HGB **2.431914**, and persistence **2.347010**, matching the manuscript after rounding.
- Local `--help` interface checked for the converted fixed-duration, electrical-linear/Ridge, and temperature auditing source files.

## Not completed in this environment

- A constant-protocol age-only HGB full numerical rerun exceeded the local execution time limit; its code compiled, but the 5.641 headline RMSE has not been independently reproduced in this update.
- The original ~708 MB raw `collab.zip` was not attached to the current source-code request. Therefore all raw-archive extraction stages, the 4501-cycle reconstruction, new 600-s/3.6-V model reruns, train/test matched filter sensitivity, and full neural training remain to be re-executed with that archive.
- No assertion of exact reproduction is made for the newly authored general censored or neural implementations until tested against archived outputs and actual fold selections.

`pytest` validates unit contracts; it is not equivalent to a complete paper reproduction. Preserve final run manifests and fold-level results before citing newly generated numeric scores.
