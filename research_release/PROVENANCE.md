# Evidence provenance

The `eb32c73` source baseline came from a `git archive` of the user's live
GitHub clone. Files already tracked in that snapshot are preserved except
for a README addendum explaining the recovered material. No `.git/` folder
is supplied, and this package does not commit or push any changes.

`research_release/results/step11n/*.csv` are the supplied completed Step 11N
output files. `inputs/step11n_frozen_input_bundle.zip` contains study age/target
columns and the original HGB censored predictions. `results/step11o/*.csv` are
content-preserving **CSV extractions** of the named sheets from
`Step11O_Preserved_Results_Workbook.xlsx`, using the standard OOXML sheet XML
when the spreadsheet renderer could not load the large sheet. No model was
refitted for extraction. The original workbook is retained for inspection.

`source_exports/` retains exact Colab `.py` exports. The local Step 11O script
under `battery_soh_reproducibility/scripts/experiments/` differs only in its
`MODEL_DIR` resolution to find the existing scripts/modeling path.

`data/step11o_constant_6299_soh_target_values.csv` deduplicates the archived
OOF labels by `(battery_id, Cycle_Index)`. It does **not** reconstruct
reference capacities from raw current integrations. The original 4,590 study
labels were recovered from Step 11N's preserved archived input bundle.

`splits/` exports the documented deterministic nested battery-holdout DESIGN
from the released battery IDs; it is not an original inner-CV execution trace.

`archived_workbooks/` are exact byte copies of previously supplied result
workbooks, and `figures/` are exact copies of available figures.
