# Author-controlled publication (PowerShell / VS Code)

**Do not publish until you've inspected this package and run validation.**

1. Make sure `git status --short` is empty and `git rev-parse --short HEAD`
   begins with `eb32c73`. If either has changed, stop and review the diff.
2. Extract the update ZIP outside the repository, preview and apply using the
   bundled `apply_update.py`, then run the release verifier and main project tests.
3. Review the gap register. If the original raw inputs or certain original
   outputs cannot be recovered, decide with your coauthor/editor how the
   manuscript's **complete reproducibility** claim can be narrowed, or which
   artifacts must still be regenerated or sourced. Do not promise missing files.
4. Commit and push from your VS Code terminal using **your own configured Git
   name and email** (commands below). No bot or ChatGPT commits are performed.
5. After the package is genuinely complete, set a final tag (e.g. `v1.0.0`),
   push it, and create a GitHub Release for that tag with a description that
   lists the actual deposited assets. `v1.0.0` is only an example, **not a
   release that currently exists**.
6. Upload exactly the versioned source and derived data to Zenodo as a new
   research dataset/software deposit. Set title, authors, date, license(s),
   manuscript title, GitHub URL and tag. Publish the record and copy its
   **actual version-specific DOI**; do not paste the manuscript placeholder.
7. Verify the Zenodo DOI resolves, the archive can be downloaded, the manifest
   verifies against those downloaded bytes, and GitHub's release/tag matches.
8. Only then update Section 4.13 and the Data Availability Statement of the
   manuscript, including a transparent statement of any input boundaries.

```powershell
# in C:\...\battery_paper (AFTER applying and reviewing the update)
python research_release/tools/verify_release.py
cd battery_soh_reproducibility
python -m pip install -e .
python -m pip install -r requirements-dev.txt
python -m pytest -q
cd ..
git status --short
git diff --stat
git config user.name
git config user.email
git add README.md research_release battery_soh_reproducibility/scripts/experiments
git commit -m "Archive verified SoH reviewer experiment evidence and release tooling"
git push origin main
```

For the actual GitHub Release: visit your repository's **Releases** page,
choose **Draft a new release**, select an existing final tag or create it,
write release notes and attach a ZIP of any outputs not represented in Git
history. Choose **Publish release**. A release is a stable, named snapshot
of a Git commit, optionally with downloadable attached artifacts; making a
commit alone does **not** create a GitHub Release.

For Zenodo: visit zenodo.org, sign in, make **New upload**, add the verified
archive ZIP (and any permitted supplemental data), title, creators, access,
licenses and related identifiers, and then **Publish** to mint the DOI.
The result must contain the preserved archived results, not only source code.
GitHub's automatic source ZIP normally does not include separately attached
Release assets. Use a manual Zenodo deposit if release assets are important.

No user authentication details or GitHub credentials need to be provided to
ChatGPT. **Never include `.git`, secret files, or unpublished personal data**
in a public research archive.
