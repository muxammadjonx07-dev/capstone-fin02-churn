# FIN-02 — Digital Banking Customer Churn Prediction

**Student:** Xalimov Muhammadjon
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
4. Models compared: naive baseline (majority class) → Logistic Regression → Random Forest
5. Final model selected based on validation F1 and ROC-AUC — Logistic Regression matched
   Random Forest on F1 but had a higher ROC-AUC and simpler structure, so it was chosen as
   the final model, then evaluated once on the frozen test set

## Models / approaches tested
| Model | Notes |
|---|---|
| DummyClassifier | naive baseline |
| Logistic Regression | **final model** — best ROC-AUC on validation, simple & interpretable |
| Random Forest | alternative approach, comparable F1 but lower ROC-AUC |

## Final model & justification
Logistic Regression was selected as the final model. On the validation set it achieved 
the same F1-score as Random Forest while obtaining a slightly higher ROC-AUC, so the 
simpler and more interpretable model was chosen. On the frozen test set, it achieved an 
F1-score of 0.62 for the churn class and a ROC-AUC of 0.842.
## Evaluation results
- Test F1 (churn class): 0.62
- Test ROC-AUC: 0.842
- Baseline comparison: since Logistic Regression was already used as the baseline, the 
  final model and baseline are the same model — see Section 12 for the justification.
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
