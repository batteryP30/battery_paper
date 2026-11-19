# CALCE source-level traceability addendum — 19 November 2026

**Evidence scope:** 231 original study-format CALCE Excel workbooks (six constant cells and two variable cells) were read in Google Colab from the author's existing `collab.zip`. This directory archives the regenerated targets, per-battery comparison results, workbook-level source SHA-256 identifiers, and validation reports. **The 231 original workbooks are not redistributed here.**

- The **six constant cells** contribute **5,686 candidate cycles**, of which **5,663** pass the original raw-quality conditions and **4,023** are within the original 70–100% SoH cohort. These values connect to the 5,663 CALCE rows in the separately retrained Step 11O evaluation.
- The **two variable cells** contribute **251 candidate reference observations**, of which **238** meet the valid-reference condition and **118** fall within the original 70–100% range. Do not conflate these counts with the six-battery variable-protocol comparison (which combines CALCE and NASA).
- Every regenerated row matches the archived Step 11A audit, including source diagnostic fields, capacity, QC, and reference-candidate status. On the 5,663 CALCE QC-valid constant cycles, the largest SoH label difference from Step 11O is ~1.421e-14 percentage points.

Reconstruction rules are distinct: constant-protocol cycles use the archived right-endpoint rectangular current-time integration procedure; variable-protocol cells use the selected 0.22-A reference-discharge block and trapezoidal integration. The original Step 11A functions are provided as `research_release/tools/calce_step11a_exact_target_functions.py` and the **executable source-level** notebook as `research_release/notebooks/CALCE_Source_to_Target_Verification_Colab.ipynb`.

## Verify against released artifacts

From the **repository root**, with the project's Python dependencies installed:

```bash
python research_release/tools/verify_calce_xlsx_evidence.py --repo-root .
python research_release/tools/verify_nasa_v2_evidence.py --repo-root .
python research_release/tools/verify_editor_chain.py --repo-root .
python research_release/tools/verify_release_portable.py --repo-root .
```

The first command checks hashes of archived CALCE outputs, all eight source-vs-Step11A records across every column, workbook fingerprints, Qref values, QC counts, and the Step11O labels. It does **not** read the 231 raw XLSX files; the Colab notebook performs that upstream work when run with `collab.zip` mounted in Google Drive.

## Data availability and limitations

The workbook hashes identify the author's particular input files and allow an independent holder of the same files to confirm identity. The checks are not proof of the original laboratory logger-to-XLSX export, and raw source files are not redistributed. Historical full model refits have not been independently completed. The native NASA MATLAB-to-study-CSV conversion is also not independently validated. The original Zenodo v0.9.1 DOI identifies the prior frozen release and **does not include this CALCE addendum**. The addendum should be deposited in a new version after validation and publication.
