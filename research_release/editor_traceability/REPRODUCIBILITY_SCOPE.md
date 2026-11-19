# Verified scope, executable routes, and limitations

| Component | Evidence available | Executable verification route | Status / limitation |
| --- | --- | --- | --- |
| NASA B0005--B0018 study CSV to integrated capacity and SoH | 4 generated cycle-level tables; source CSV SHA-256 and sizes | `research_release/notebooks/NASA_Study_CSV_Traceability_Verification_Colab_v2.ipynb`, using original `collab.zip` | 4/4 passed on 2026-11-19; study CSVs not distributed |
| NASA RW3--RW6 study CSV to reference Ah and following-reference assignment | 4 event tables + 4 cycle maps; 73 events/2,607 assignments; eight source hashes | Same Colab notebook; saved results independently checked by `verify_nasa_v2_evidence.py` | 4/4 passed on 2026-11-19; native `.mat` conversion not verified |
| CALCE source workbooks to integrated targets | Archived Step 11A CALCE audit and code | Original pipeline with original CALCE workbooks | Archived outputs available; no new independent raw-source rerun in this addendum |
| Step 11A capacity audits to Step 11O label keys | 6,322 archived candidates, 6,299 valid records | `verify_editor_chain.py` | Archived chain PASS; not a substitute for source-level CALCE reconstruction |
| NASA RW pre-reference target matching | 73 reference events, 69 intervals after first references, 47 primary observations, 2,607 cycle maps | `verify_nasa_v2_evidence.py` plus `verify_editor_chain.py` | Cross-check PASS for study CSVs and archived maps |
| Constant-protocol LOBO outer/inner design | 10 outer and 90 **reconstructed** inner design records | `verify_editor_chain.py` | Design verified; original inner-optimization event logs not preserved |
| Step 11O OOF model predictions/configurations | 6,299 held-out observations x five models; 20 nonempty selected-feature configurations | `verify_release_portable.py` and `verify_editor_chain.py` | Archived outputs checked; raw full model refits NOT verified |
| Step 11N true 3.6-V OOF comparisons | 4,589 held-out observations x five models, compact age-baseline input bundle | `verify_release_portable.py`; optional Step 11N experiment script for recovered baselines | Archived metrics pass; not certified rerun of every historical censored electrical model |
| Published v0.9.0 manifest | 57 hashes: 45 exact on Linux, 12 LF/CRLF-only by reverse-transform | `verify_release_portable.py` | Exact published hash held intact; no unexplained mismatch |
| Original NASA upstream native format | NASA native `.mat` not present in `collab.zip` | Requires separate native-to-study CSV reconstruction with independent SHA-256 / row comparison | Not verified |

**Never phrase the status as “all experiments independently reproduced”**. The essential traceability achieved is from eight *preserved study-format NASA CSVs* through re-integrated reference capacities and archived targets to keyed held-out results, with the above limitations. Model refitting and native source conversion are separate reproducibility levels.
