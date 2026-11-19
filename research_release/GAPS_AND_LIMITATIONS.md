# Artifact limitations and outstanding reviewer requirements

This release candidate is a **transparent recovered-record deposit**. It is
not yet the fully reproducible, independently verified release requested by
Reviewer Comment 4. The following absences are not replaced by invented data.

| Item | Status | Impact / remediation |
|---|---|---|
| Original ~708 MB `collab.zip` input archive | NOT AVAILABLE here | Cannot certify a fresh raw-to-final model rerun or compute original input SHA-256. Recover from personal backups or exact official source exports; otherwise document availability boundary. |
| Historical university-Drive experiment folders | NOT AVAILABLE | Some original model snapshots, cached candidate tables and log files cannot be recovered. |
| NASA native-to-measurement-CSV conversion | UNVERIFIED historical source | Python source starts at expected measurement CSV exports. Do not claim end-to-end NASA native ingestion is reproduced. |
| CALCE/NASA original input file hashes | NOT AVAILABLE | List source download identifiers and filenames where known; SHA-256 cannot be extracted without the original bytes. |
| Battery-specific Qref values in Ah / complete target provenance | PARTIAL | The released SoH values and archived Step 11A workbook are preserved; full original raw capacity and QC provenance needs separate verification. |
| Primary constant/variable archived OOF files | PARTIAL | Original-run summaries exist in archived workbooks. |
| Inner-validation fold configuration trials for all experiments | PARTIAL | Step 11N HGB searches and Step 11O selected configs/features are available; the exported inner-fold design is deterministic rather than a saved run trace. |
| Step 11O training-only feature selection logs | PARTIAL | Selected final outer-fold feature names and HGB configurations recovered; original inner scoring traces absent. |
| Original complete training checkpoints | NOT RECOVERED | Saved OOF predictions support verification of reported metrics, not model recreation. |
| Neural and transfer exact fold-level OOF verification | PARTIAL | Archived workbooks and code are preserved but completeness of all individual records may require further audit. |


## Scientific scope

The original 70-100% cohort was target-selected. Step 11O evaluates 6,299
reconstructed reference-labelled cycles with raw-record-based eligibility and
full-discharge diagnostic features, but does not demonstrate early-prediction
performance on the unrestricted population. Full-discharge derivative
features can encode indirect capacity information; do not call them capacity-
independent. CALCE variable observations describe retrospective macro-cycle
characterization, whereas NASA RW3-RW6 use pre-reference measurements. Their
combined average is descriptive, not a homogeneous prospective benchmark.


