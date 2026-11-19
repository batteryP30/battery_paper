# Academic-editor traceability evidence

**Status:** Prepared for a future GitHub/Zenodo version *after* the published v0.9.0 release. These materials are **not included** in DOI `10.5281/zenodo.24286732` until uploaded as a new record version. Do not cite v0.9.0 as containing the files in this directory.

## Contents and direct provenance

- `step11a_battery_reference_summary.csv` — preserved Step 11A battery-level Ah references for 16 cells.
- `step11a_source_input_registry.csv` — historical input file paths, workbook names and expected sheets.
- `step11a_constant_cycle_capacity_audit.csv` — 6,322 candidate cycles (6,299 passed raw QC), integrated Ah, reference Ah and SoH.
- `step11a_nasa_rw_reference_events.csv` — 73 NASA RW reference discharges and raw row boundaries.
- `step11a_nasa_rw_pre_reference_cycle_map.csv` — 2,607 operating cycles assigned to subsequent references.
- `step11a_calce_variable_reference_audit.csv` and `step11a_regression_checks.csv` — other archived Step 11A audits.
- `nasa_csv_v2/` — actual 2026-11-19 Colab verification reports, eight per-cell PASS records, regenerated constant cycle/reference tables, regenerated RW cycle mappings, source CSV file sizes/SHA-256 identifiers, audit environment, original checker stderr, and precise manifest discrepancy classifications. See `nasa_csv_v2/EVIDENCE.sha256` for **file-byte** hashes of these included reports.
- `HISTORICAL_RELEASE_NOTICES.md` — flags 2026-11-16 pre-deposit status text in the earlier v0.9.0 documentation as historical.
- `REPRODUCIBILITY_SCOPE.md` — audit scope and refit/archived-output distinction.

The executable Colab notebook used for the *independent study-CSV re-integration* is located at `research_release/notebooks/NASA_Study_CSV_Traceability_Verification_Colab_v2.ipynb`. It requires the author's original measurement archive `collab.zip`, which is not included because it is large and redistribution conditions require review. The notebook was run against the CSV files located in that archive; its exact source sizes and SHA-256 hashes were recorded.

## Source verification result (2026-11-19)

B0005, B0006, B0007 and B0018 each matched their Step 11A cycle-level measurements and reconstructed SoH. RW3, RW4, RW5 and RW6 each matched their Step 11A reference-event integration and cycle-to-following-reference assignments. **Eight out of eight NASA study-format source CSV checks passed**. The largest observed absolute SoH difference was about `1.42e-14` percentage points. Across RW3--RW6, 73 reference events and 2,607 operating-cycle assignments were checked.

### NASA first-valid integrated reference discharge capacities (Ah)

| Cell | Qref (Ah) |
| --- | ---: |
| B0005 | 1.8619519503655222 |
| B0006 | 2.0519781081621304 |
| B0007 | 1.9186612348565528 |
| B0018 | 1.8685276687897638 |
| RW3 | 4.011886890277778 |
| RW4 | 3.948668852777778 |
| RW5 | 3.932915277777778 |
| RW6 | 3.848791011111112 |

The reference capacities above are *derived first-valid-reference* baselines, not manufacturer ratings. The reconstruction functions are in `battery_soh_reproducibility/src/battery_soh_repro/data_pipeline.py`. Constant-cell discharge Ah is calculated from finite time/current samples, using positive time increments and a nonnegative discharge-current convention. The RW procedure identifies recorded `reference...discharge` operations, integrates discharge-current magnitudes over valid increments, and assigns each random-walk interval to the **subsequent** reference event. The first reference cannot be paired with earlier usage. See the archived Step 11A columns and source code for the exact QC logic and the differences between constant and RW integration schemes.

## Running the checks

From the repository root (with the project's Python dependencies installed):

```bash
python research_release/tools/verify_nasa_v2_evidence.py --repo-root .
python research_release/tools/verify_editor_chain.py --repo-root .
python research_release/tools/verify_release_portable.py --repo-root .
```

The first script checks the saved Colab outputs against the Step 11A tables by battery, cycle and reference-event keys; the second verifies capacity-to-SoH labels, OOF observations, split designs and saved configurations; the third checks the v0.9.0 frozen manifest with an exact reversibility rule for the **12 known CRLF/LF changes**, followed by the existing archived numerical verifier. It does **not** ignore checksum differences.

The original `verify_release.py` is deliberately unchanged. On a Linux Git checkout it may still report `SHA256 changed` for the twelve CSVs, while the narrowly scoped portable check confirms exact reconstruction of their original CRLF bytes. On a Windows checkout preserving CRLF it should report exact matches. This issue concerns Git checkout portability rather than numerical differences.

## Explicit scope boundaries

- **Verified from study-format CSVs:** eight NASA CSV inputs under `collab.zip` to integrated reference-capacity and target/mapping tables, using the released reconstruction code.
- **Verified from archived outputs:** linked Step 11A records, 6,299 Step 11O targets, 4,589 Step 11N targets, saved LOBO fold designs, selected-feature/configuration records and preserved OOF scores.
- **Not independently verified:** conversion from upstream NASA native MATLAB data to the historical study-format CSVs, raw CALCE workbook reconstruction in this Colab exercise, and all historical model training/tuning/refits. The deterministic 90 inner split rows are reconstructed validation *design*, not original optimizer execution logs.

An archived PASS must not be represented as original-model refit verification. The original v0.9.0 DOI remains valid for its contents only.
