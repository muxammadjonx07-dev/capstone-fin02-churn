# Project Status

## Current stage
- **Module/Class:** M8C4 → final submission
- **Stage:** Data Audit, modeling, evaluation, and documentation complete — ready for submission
- **Data Gate status:** Green
- **Last reviewed:** 2026-08-16

## Evidence map
- **Data source and limitations:** `data/README.md`
- **Audit, issue log, split, leakage and preprocessing:** `docs/data_audit.md`
- **Implementation path:** `demo.ipynb`
- **Split/manifest evidence:** `demo.ipynb`, data-splitting section (train 4507 / val 1127 / test 1409)
- **Latest milestone commit:** (fill in after pushing this update)

## Current correction or blocker
- None. DQ-04 confirmed 2026-08-16 (0 exact duplicate rows, 0 duplicate `customerID`s); Data Gate
  is Green; baseline-vs-final comparison and experiment log are in `demo.ipynb`.

## Next action
Upload `demo.ipynb`, `README.md`, `data/`, `docs/`, and this file to the repository, then do one
clean `Runtime → Disconnect and delete runtime` → `Run all` pass in Colab to confirm everything
reproduces from scratch. Commit and push before the deadline (2026-08-16, 23:59) — do not commit
after that.
