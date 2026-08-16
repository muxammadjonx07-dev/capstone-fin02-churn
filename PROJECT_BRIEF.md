# FIN-02 — Digital Banking Customer Churn Prediction
**FinTech | Technical Project Brief — Field-Based Scenario**

| Field | FinTech |
|---|---|
| Client | Digital Bank / FinTech Platform |
| Track | Field-Based Scenario |
| Student | Xalimov Muhammadjon |

---

## 1. Client Background
A digital banking platform wants to retain valuable customers before they stop using its
products. Customer activity changes over time: some reduce transactions gradually, some stop
using the app, while others close accounts or move their primary activity elsewhere.

## 2. Business Problem
The retention team does not have a reliable early-warning mechanism for likely churn. A major
challenge is defining churn in a way that reflects actual product behavior and is measurable
from available data.

## 3. Requested Solution
Develop an ML-based solution that identifies customers at elevated risk of churn within a
clearly defined future period, using information that would be available before the churn event.

## 4. Available Information
The organization may have access to account tenure and product usage, transaction/activity
summaries, balance or usage trends, digital engagement or support activity, and public banking
churn datasets or comparable customer-behavior datasets.

---

## 5. Data & Problem Discovery

| Decision / Question | Response |
|---|---|
| **Selected dataset and source** | Telco Customer Churn (IBM sample dataset), used as a public, well-documented **comparable customer-behavior dataset** standing in for direct banking transaction data, which was not available. Source: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`. |
| **What does one record represent?** | One customer account snapshot at a point in time — a single row per customer, not a longitudinal/time-series record. |
| **Proposed target / ML objective** | Binary target `Churn` (Yes/No) — whether the customer has left the service. Objective: output a churn probability usable as a risk score. |
| **Key information available at prediction time** | Account tenure, contract type, billing method, monthly/total charges, subscribed services (phone, internet, streaming, etc.), and basic demographics (gender, senior citizen status, partner/dependents). All fields represent pre-outcome account state. |
| **Main data quality issues** | (1) `TotalCharges` stored as text due to 11 blank values, all corresponding to customers with `tenure == 0` — coerced to numeric and median-imputed. (2) Target class imbalance (~26.5% churn / 73.5% no-churn) — addressed via `class_weight='balanced'` and F1/ROC-AUC as primary metrics instead of accuracy. (3) No exact-duplicate rows or duplicate `customerID`s (verified). |
| **Potential leakage risks** | `customerID` is a unique, non-predictive identifier — excluded from the feature set before preprocessing. All preprocessing (imputation, scaling, encoding) is fit exclusively on the training split, inside a single `sklearn.Pipeline`, so no validation/test information leaks into training. |
| **Privacy / fairness / licensing concerns** | Dataset is public and de-identified; licensed for educational/demo use. Slice-level fairness evaluation (by gender, senior-citizen status, and contract type) was performed on the frozen test set — see Responsible AI section of the main README for the actual results and their implications. |

## 6. Technical Proposal

| Decision / Question | Response |
|---|---|
| **ML problem formulation** | Supervised binary classification, scored via `predict_proba` to produce a continuous churn-risk probability rather than a bare label. |
| **Proposed baseline** | `DummyClassifier` (always predicts the majority class) — establishes the naive floor (F1 = 0.0 on the churn class) that any real model must beat. |
| **Main modeling approaches investigated** | Logistic Regression (`class_weight='balanced'`) as the primary candidate model, compared against Random Forest (`class_weight='balanced'`, tuned `max_depth`). |
| **Data splitting / validation strategy** | Stratified random split into train / validation / test (~64% / 16% / 20%), stratified on `Churn`. The test set is frozen and evaluated exactly once, after model selection is finalized on the validation set. |
| **Primary evaluation metric(s) and why** | F1 (churn class) and ROC-AUC. Accuracy is misleading under ~27% class imbalance; missing an actual churner (false negative) is costlier for a retention team than a false alarm, so F1 on the minority class is prioritized over overall accuracy. |
| **Expected inference input** | A dict of customer attributes matching the training schema (tenure, contract, billing, services, demographics — `customerID` and `Churn` excluded). |
| **Expected inference output** | `{'churn_probability': float, 'risk_flag': 'High'|'Low'}` — a probability plus a thresholded risk flag (threshold 0.5) usable for prioritizing retention outreach. |
| **Main technical risks / assumptions** | (1) A telecom dataset is used as a stand-in for banking behavior — churn drivers may not transfer exactly. (2) Single historical snapshot — no temporal/concept-drift modeling. (3) Model performance is uneven across customer segments (notably much weaker on long-contract customers), documented explicitly rather than hidden. |

---

## 7. Functional Requirements
- Churn is defined operationally before modeling (see Section 5).
- The solution scores a customer using only information available before the churn outcome.
- Output is usable for prioritizing retention activity (probability + risk flag).
- Leakage from account closure or post-churn information is prevented (`customerID` excluded, pipeline fit on train only).
- Fairness, privacy, and potential misuse of the risk score are discussed in the Responsible AI section of the main README.

## 8. Deliverables (this repository)
- Working ML pipeline: `demo.ipynb` (Colab-first, reproducible end-to-end)
- Terminal-runnable equivalent: `src/train.py`, `src/predict_cli.py`
- Documented dataset source, assumptions, preprocessing: `data/README.md`, `docs/data_audit.md`
- Evaluation results on frozen, unseen test data: `demo.ipynb` Sections 13-14, summarized in `README.md`
- Usable inference interface: `predict_churn_risk()` (notebook) / `predict_cli.py` (terminal)
- Reproducible repository with run instructions: `README.md`, `TERMINAL_DEMO.md`
- Limitations, risks, and next steps: `README.md` "Known limitations" and "Responsible AI considerations"

## 9. Acceptance Criteria
- Processes previously unseen input in the expected dict/row format.
- Output (churn probability + risk flag) is meaningful for a retention team's prioritization workflow.
- Methodology and evaluation are documented and defensible (baseline comparison, error analysis, fairness slices).
- Fully reproducible from the repository (`Runtime → Run all` in Colab, or `python src/train.py` + `python src/predict_cli.py` in a terminal).
- Known limitations, risks, and assumptions are documented (see Section 6 and main README).

## 10. Constraints
| IN SCOPE | OUT OF SCOPE |
|---|---|
| Prototype ML solution | Direct production integration |
| Public / legally usable data | Use of unauthorized private data |
| Reproducible inference workflow (notebook + terminal) | Enterprise-scale infrastructure / deployment |
| Reasonable student-scale demo | Claims beyond what the model can validly support |

## 11. Questions Resolved
- **What counts as churn, and over what horizon?** `Churn = Yes/No` as recorded in the account snapshot — a point-in-time label, not a forward-looking horizon (limitation noted in Section 6).
- **Which indicators could leak post-churn information?** None used — all features represent pre-outcome account state; `customerID` excluded as a non-predictive identifier.
- **How is class imbalance handled?** `class_weight='balanced'` in both candidate models; F1/ROC-AUC used instead of accuracy.
- **Which errors are more expensive?** False negatives (missed churners) — costlier than false alarms for a retention team; reflected in the metric choice.
- **What action would the business take at different risk levels?** `risk_flag = 'High'` (probability ≥ 0.5) is intended to trigger prioritized human-led retention outreach; `'Low'` requires no immediate action.
- **How is performance variation across segments assessed?** Slice-level F1/recall computed by gender, senior-citizen status, and contract type on the frozen test set — see Responsible AI section of the main README for results.

## 12. Optional Directions (not pursued, noted for transparency)
Risk bands, SHAP/feature importance, temporal validation, and a full retention-prioritization
demo app were considered out of scope for this submission — the core required deliverables
(Sections 8-9) were prioritized over these optional extensions given project timeline.

---
*General capstone requirements for experiment tracking, documentation, and grading are defined
in the official Capstone Evaluation Criteria and Implementation Helper documents.*
