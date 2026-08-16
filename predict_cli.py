"""
predict_cli.py — FIN-02 Churn Prediction: terminal demo.

Loads the trained model (run train.py first) and predicts churn risk for a
customer, either interactively or from a JSON file/string. This is what you
run live in the terminal during your defense to "show the project running".

Usage:
    python train.py                     # train once, saves models/churn_model.joblib
    python predict_cli.py               # interactive prompts
    python predict_cli.py --example     # runs one built-in example, no typing needed
    python predict_cli.py --json '{"gender": "Female", ...}'
"""
import argparse
import json
import sys

import joblib
import pandas as pd

MODEL_PATH = "models/churn_model.joblib"

FIELDS = [
    ("gender", ["Female", "Male"]),
    ("SeniorCitizen", ["0", "1"]),
    ("Partner", ["Yes", "No"]),
    ("Dependents", ["Yes", "No"]),
    ("tenure", "int (months, e.g. 12)"),
    ("PhoneService", ["Yes", "No"]),
    ("MultipleLines", ["Yes", "No", "No phone service"]),
    ("InternetService", ["DSL", "Fiber optic", "No"]),
    ("OnlineSecurity", ["Yes", "No", "No internet service"]),
    ("OnlineBackup", ["Yes", "No", "No internet service"]),
    ("DeviceProtection", ["Yes", "No", "No internet service"]),
    ("TechSupport", ["Yes", "No", "No internet service"]),
    ("StreamingTV", ["Yes", "No", "No internet service"]),
    ("StreamingMovies", ["Yes", "No", "No internet service"]),
    ("Contract", ["Month-to-month", "One year", "Two year"]),
    ("PaperlessBilling", ["Yes", "No"]),
    ("PaymentMethod", ["Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"]),
    ("MonthlyCharges", "float (e.g. 70.35)"),
    ("TotalCharges", "float (e.g. 845.20)"),
]

EXAMPLE_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 420.0,
}


def prompt_for_customer() -> dict:
    print("Enter customer details (press Enter to see valid options where shown):\n")
    customer = {}
    for name, hint in FIELDS:
        if isinstance(hint, list):
            hint_str = "/".join(hint)
        else:
            hint_str = hint
        while True:
            val = input(f"  {name} [{hint_str}]: ").strip()
            if not val:
                print("    -> required field, try again")
                continue
            break
        if name in ("tenure", "SeniorCitizen"):
            val = int(val)
        elif name in ("MonthlyCharges", "TotalCharges"):
            val = float(val)
        customer[name] = val
    return customer


def main():
    parser = argparse.ArgumentParser(description="Predict churn risk from the terminal.")
    parser.add_argument("--example", action="store_true", help="use a built-in example customer")
    parser.add_argument("--json", type=str, help="customer data as a JSON string")
    args = parser.parse_args()

    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Model not found at {MODEL_PATH}. Run `python train.py` first.")
        sys.exit(1)

    if args.example:
        customer = EXAMPLE_CUSTOMER
        print("Using built-in example customer:\n", json.dumps(customer, indent=2))
    elif args.json:
        customer = json.loads(args.json)
    else:
        customer = prompt_for_customer()

    row = pd.DataFrame([customer])
    proba = model.predict_proba(row)[0, 1]
    risk = "High" if proba >= 0.5 else "Low"

    print("\n=== PREDICTION ===")
    print(f"Churn probability: {proba:.3f}")
    print(f"Risk flag:         {risk}")


if __name__ == "__main__":
    main()
