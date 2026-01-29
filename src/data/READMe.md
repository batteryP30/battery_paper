## Data Description

The battery cycling data used in this work were obtained from the Center for Advanced Life Cycle Engineering (CALCE) at the University of Maryland, College Park. The dataset is publicly available at:

https://calce.umd.edu/data

The data consist of CS2 prismatic lithium-ion cells cycled under a constant current–constant voltage (CC–CV) charging protocol with a discharge cut-off voltage of 2.7 V. The selected cells include experiments with both constant current discharge and variable current discharge, providing a range of operating conditions for state-of-health (SoH) analysis.

The raw experimental data are provided as time-series measurements stored across multiple Microsoft Excel files for each battery. For each cell, all Excel files within the corresponding folder were imported in chronological order, sorted by file creation time. This ordering preserves a continuous aging trajectory when cycling tests span multiple data files. The relevant worksheet(s) from each file were extracted and concatenated to form a single continuous time series per battery.

To ensure a monotonically increasing cycle count across concatenated files, the cycle index in each newly imported file was offset by the maximum cycle index from the previously imported file. This resulted in a continuous and strictly increasing cycle index for each battery throughout its lifetime. A `battery_id` column was added to all records, derived from the folder name, and used as a unique identifier for each individual cell.

### Notes
- Time-series variables include current, voltage, test time, step index, and related measurements.
- Both constant-discharge and variable-discharge protocols are included.
- All preprocessing steps preserve cycle continuity across file boundaries.
