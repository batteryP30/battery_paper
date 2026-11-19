# Read with the Colab reports

These 37 files were produced by the **2026-11-19T09:07:19 UTC** Colab v2 verification of the preserved eight NASA study CSV exports under the author's original `collab.zip` archive. The raw measurements are not redistributed in this directory.

- B0005, B0006, B0007, B0018: `*_reintegrated_constant_targets.csv` and per-cell check JSONs; **636** rows combined.
- RW3--RW6: `*_reintegrated_reference_events.csv` and `*_regenerated_cycle_reference_map.csv`; **73** reference events and **2,607** mapped usage cycles combined.
- `NASA_Source_Manifest_SHA256.csv`: the eight input-file names, sizes and the SHA-256 digests of the **source CSVs used in Colab**, facilitating re-identification of the exact exports; they are **not** native NASA `.mat` hashes.
- `Release_Manifest_Diagnostics.csv`: 45 exact checksum matches and 12 CRLF/LF-only reversible differences against the original v0.9.0 manifest; no unexplained mismatches.
- `verify_release_original.log`: original verifier exits nonzero on exact-byte mismatch, so **do not** claim it passed in this Colab run.
- `Archive_Checks_Classification.json`: records that the archived numerical computations passed after a separate, narrowly proven EOL classification.

`EVIDENCE.sha256` hashes the **original 37 Colab report files**, not this interpretation text.

Source-level verification rerun requires the exact eight study-format CSVs, which can be identified by the recorded SHA-256 digests. Native NASA `.mat` conversion, raw CALCE source reconstruction and all historical refits are **not** established by these files.
