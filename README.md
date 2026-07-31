# FIN-02 — Digital Banking Customer Churn Prediction

**Student:** <your full name>
**Track:** Field-Based Scenario — FIN-02
**Course:** AI/ML Fundamentals — Capstone Project

## Problem statement
A digital banking / telecom platform needs to identify customers at elevated risk of churn
*before* they leave, so the retention team can prioritize outreach. See `demo.ipynb` Section 1-2
for the full problem framing and ML task definition.

## ML task
- Type: Binary classification
- Input: customer account, contract, and usage attributes
- Target: `Churn` (Yes/No)
- Primary metric: F1 on the churn class (accuracy is misleading — ~27% churn rate)

## Dataset
- **Source:** Telco Customer Churn (IBM sample dataset), loaded from:
  `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- ~7,043 rows, 20 features + target
- License: publicly available sample dataset for educational use — cite the source above
- Known limitation: single historical snapshot from one provider; not necessarily representative
  of all telecom/banking customer populations

## Pipeline / architecture
1. Load data → clean `TotalCharges`, drop `customerID` (pure identifier, no signal)
2. Stratified train / validation / test split (test set frozen until final evaluation)
3. Preprocessing: median imputation + scaling (numeric), most-frequent imputation + one-hot
   encoding (categorical) — all inside a single `sklearn.Pipeline` to avoid leakage and keep
   training/inference preprocessing identical
4. Models compared: naive baseline (majority class) → Logistic Regression (baseline model) →
   Random Forest (main model)
5. Final model selected based on validation F1 (churn class), then evaluated once on the frozen
   test set

## Models / approaches tested
| Model | Notes |
|---|---|
| DummyClassifier | naive baseline |
| Logistic Regression | simple model baseline, `class_weight=balanced` |
| Random Forest | main model, compared against baseline |

(Fill in your actual metric numbers here after running `demo.ipynb`.)

## Final model & justification
<Fill in: which model won, on what metric, and why — 2-3 sentences>

## Evaluation results
- Test F1 (churn class): <value>
- Test ROC-AUC: <value>
- Baseline comparison: <value>
- See `demo.ipynb` Sections 13-14 for the confusion matrix and error analysis.

## Installation
```bash
pip install -r requirements.txt
```

## Running the demo (Colab-first)
1. Open `demo.ipynb` in Google Colab (Runtime → Run all, on a fresh runtime).
2. The notebook downloads the dataset directly from the public URL — no manual download needed.
3. Sections 1-15 reproduce training and evaluation end-to-end.
4. Section 16 (`predict_churn_risk`) is the inference demo — run it with any customer dict shaped
   like the training columns to get a churn probability + risk flag.

## Example input / output
```python
predict_churn_risk({
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 5, "PhoneService": "Yes", ... # (see notebook for full field list)
})
# -> {'churn_probability': 0.71, 'risk_flag': 'High'}
```

## Known limitations
- Trained on one historical snapshot; real deployment needs periodic retraining (concept drift)
- Public dataset may not represent all customer segments equally
- No causal claims — this is a predictive risk score, not an explanation of *why* a customer churns

## Responsible AI considerations
- Model outputs a risk score to prioritize human-led retention outreach — not an automated
  decision to cancel or modify a customer's service
- Slice-level fairness should be checked across contract type / tenure / demographic fields
  before any real use
- Underlying data is public and de-identified; a production system would need explicit consent
  and a data-retention policy

## Repository structure
```
.
├── README.md
├── requirements.txt
├── demo.ipynb
└── models/
    └── churn_model.joblib   (generated after running the notebook)
```
