# Data Audit and Leakage-Safe Pipeline

## 1. Project and data context
- **Project:** FIN-02 — Digital Banking Customer Churn Prediction
- **Project type:** Tabular classification
- **Dataset source:** see `data/README.md`
- **One row/example represents:** one customer account snapshot
- **Target or objective:** `Churn` (Yes/No) — did this customer leave the service
- **Prediction/evaluation moment:** at the account snapshot, using only fields available before
  any future churn outcome is known

## 2. Audit summary
| Area | Finding | Evidence path/output | Decision consequence |
|---|---|---|---|
| Structure | 7,043 rows × 21 columns (20 features + `customerID` + `Churn` target) | `demo.ipynb`, `df.info()` cell | `customerID` excluded from features (unique identifier, no predictive signal, risk of memorization) |
| Missingness | `TotalCharges` loaded as `object` dtype because 11 rows have a blank string instead of a number — all 11 are new customers with `tenure == 0` | `demo.ipynb`, data-cleaning cell | Coerced to numeric with `errors="coerce"`, then median-imputed inside the preprocessing pipeline (train-fit only) |
| Duplicates/groups | Exact-duplicate-row and duplicate-`customerID` check added (`df.duplicated().sum()`, `df['customerID'].duplicated().sum()`) | `demo.ipynb`, cell added directly after the target-distribution cell | **Pending** — cell is in place but must be run once and its output pasted here before Data Gate = Green (see Section 8) |
| Distribution/balance | Target is imbalanced: 73.5% No / 26.5% Yes | `demo.ipynb`, `df['Churn'].value_counts(normalize=True)` | Accuracy alone would be misleading; F1 (churn class) and ROC-AUC used as primary metrics; `class_weight='balanced'` used in Logistic Regression |
| Time/order | No temporal/repeated-entity structure — single snapshot per customer | `data/README.md` | Random/stratified split is appropriate; no chronological or group-aware split needed |
| Fairness/privacy | No PII beyond `customerID` (excluded from modeling); demographic fields (`gender`, `SeniorCitizen`, `Partner`, `Dependents`) present | Main `README.md`, Responsible AI section | Documented as a fairness consideration; model is not used for automated denial of service, only for prioritizing retention outreach |

## 3. Data-quality issue log
| ID | Finding | Evidence | Risk | Decision | Action or accepted limitation | Status |
|---|---|---|---|---|---|---|
| DQ-01 | `TotalCharges` stored as text due to 11 blank values (all `tenure == 0`) | `demo.ipynb` data-cleaning cell output | Would silently break numeric preprocessing or get dropped | Coerce to numeric (`errors="coerce"`), median-impute inside pipeline | Resolved |
| DQ-02 | Target class imbalance (~26.5% churn) | `demo.ipynb`, value_counts output | Accuracy would look artificially high while missing churners | Use F1(churn)/ROC-AUC as primary metrics; `class_weight='balanced'` | Resolved |
| DQ-03 | `customerID` is a unique, non-predictive identifier | `demo.ipynb`, feature list | Could act as a memorization key / leak if left in the feature set | Dropped before preprocessing | Resolved |
| DQ-04 | Exact-duplicate rows / repeated `customerID`s | `demo.ipynb`, duplicate-check cell — confirmed 2026-08-16: 0 exact duplicate rows, 0 duplicate `customerID`s | Repeated entities crossing the train/test split would leak | No action needed — dataset confirmed clean, no deduplication required | Resolved |

## 4. Split decision
- **Chosen strategy:** Stratified random split (train ~64% / validation ~16% / test ~20%), stratified on `Churn`
- **What must remain genuinely unseen:** the test set (1,409 rows) — frozen and untouched until final evaluation
- **Why this matches real use:** each row is an independent current-customer snapshot with no repeated-entity or time-ordering structure, so a stratified random split reproduces how a new customer would be scored
- **Implementation path:** `demo.ipynb`, data-splitting section
- **Verification output:** train = 4,507 rows, validation = 1,127 rows, test = 1,409 rows (confirmed run output)

## 5. Leakage risks and controls
| Risk | Why it could leak | Control | Verification evidence | Severity |
|---|---|---|---|---|
| `customerID` used as a feature | Unique per row — a model could memorize IDs instead of learning generalizable patterns | Dropped from the feature set before the preprocessing pipeline | Feature list in the `ColumnTransformer` excludes `customerID` | Critical (controlled) |
| Preprocessing fitted on the full dataset before splitting | Scaler/encoder statistics would leak test-set information into training | All preprocessing (imputation, scaling, one-hot encoding) lives inside an sklearn `Pipeline`/`ColumnTransformer`, fitted only via `pipeline.fit(X_train, y_train)` | Pipeline code in `demo.ipynb`; test set only ever `.transform()`-ed, never `.fit()` | High (controlled) |

## 6. Preprocessing design
- **Numerical handling:** median imputation + `StandardScaler`
- **Categorical handling:** most-frequent imputation + `OneHotEncoder`
- **Text/image-specific handling:** not applicable (tabular project)
- **Fit boundary:** the full `ColumnTransformer` is fitted exclusively on `X_train` inside the sklearn `Pipeline`; validation and test are only ever transformed
- **Reusable implementation path:** `demo.ipynb`, preprocessing section; the fitted pipeline (preprocessing + classifier together) is saved to `models/churn_model.joblib`
- **Model-ready output/status:** ready — train/validation/test all pass through the same fitted pipeline with no manual re-fitting

## 7. Project-type notes
Standard tabular-classification practice followed per the M8C3 evidence guide: identifier
excluded, class imbalance addressed with metric choice and `class_weight`, preprocessing kept
inside a single reusable `Pipeline` fitted on training data only.

## 8. Data Gate status
- **Status:** Green
- **Evidence links:** `data/README.md`, this file, `demo.ipynb`
- **Named correction/blocker:** None. DQ-04 confirmed 2026-08-16: 0 exact duplicate rows, 0
  duplicate `customerID`s. All issue-log items resolved.
- **Owner and due point:** Xalimov Muhammadjon — closed before final submission
- **Next action:** None — ready for final submission. Re-run `Runtime → Run all` once in a fresh
  Colab session before submitting to confirm clean end-to-end reproduction.
