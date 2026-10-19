# Raw data layout

The loader recursively discovers the validated study inputs, so an exact
top-level folder name is not required. A single supplied ZIP must contain:

- one folder for each CALCE cell:
  - `CS2_33`, `CS2_34`, `CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`
  - `CS2_3`, `CS2_9`
- one CSV for each NASA cell:
  - `B0005.csv`, `B0006.csv`, `B0007.csv`, `B0018.csv`
  - `RW3.csv`, `RW4.csv`, `RW5.csv`, `RW6.csv`

The validation routine checks workbook-sheet availability for CALCE and
required measurement columns for NASA before proceeding. It also sorts CALCE
workbooks chronologically using the date encoded in the filename, matching the
validated preprocessing workflow.

No raw or processed battery data are distributed with this source package.

## NASA input provenance note

The final retained analysis begins from the NASA measurement CSV exports listed
above. These files preserve the sample-level current, voltage, time, cycle, and
operation metadata needed by the corrected target and robustness pipelines. A
validated repository-native NASA archive-to-CSV conversion script was not found
in the retained source history and is therefore not fabricated here.

The final target code independently integrates current over time and does not
trust source-provided SoH labels.
