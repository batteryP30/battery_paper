# Local archival evidence checks (November 16, 2026)

Validation performed against the user's archived Git repository at `eb32c73`,
with the recovered reviewer outputs that were present in this conversation.
No original raw battery archive was supplied and no complete model retraining
was attempted or claimed.

## Confirmed checks

- The unchanged nested Python research project passed **19 pytest tests**
  (one non-failing PyTorch warning in the sequence encoder test).
- The recovered Step 11N and Step 11O scripts opened with `--help` using the
  nested project as their Python environment.
- All **five Step 11N models** have 4,589 unique battery/cycle predictions,
  and all five Step 11O models have 6,299 unique predictions. No repeated
  model/battery/cycle keys or target contradictions were detected.
- For every one of the 4,589 Step 11N observation keys, the Step 11O
  saved target value is identical to numerical precision.
- Independently recalculated equal-battery mean RMSE values from released
  OOF predictions: Step 11N electrical/electrical+age HGB
  2.282087/1.715947, Step 11O electrical/electrical+age HGB
  2.288309/2.194821 (SoH percentage points).
- Step 11O has 1,709 recorded cycles below 70% SoH and 4,590 between
  70% and 100% inclusive, and recovered 20 outer-fold electrical feature lists.
- The deterministic 10 outer / 90 inner battery split designs were generated
  from the documented leave-one-battery-out algorithm. Original execution
  logs were not available to independently check search chronology.
- SHA-256 manifest checks passed for all bundled research assets in the
  staging copy. These hashes must continue to match after author upload.

## Not verified

- Source-native NASA conversion and original CALCE/NASA raw file digests.
- Complete computational reproduction of every experimental table/figure
  from original raw measurements.
- Fidelity of every historical model selection run not accompanied by saved
  inner fold search logs or model checkpoint files.
- Actual GitHub release creation, license clearance, Zenodo publication, DOI
  resolution or public artifact download access.

Repeat after locally applying the patch:

`python research_release/tools/verify_release.py`

Only publish as a completed computational release after resolving or explicitly
accepting the unverified boundaries above in consultation with the coauthors
and the reviewer/editor.
