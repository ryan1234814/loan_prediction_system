import os
import sys
import torch
import numpy as np
import pandas as pd
import joblib

# Add current dir to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from loan_approval import LoanModel

def test_probabilities():
    artifacts_dir = 'model_artifacts'
    artifacts = joblib.load(os.path.join(artifacts_dir, 'preprocessing_artifacts.joblib'))
    input_dim = len(artifacts['feature_names'])
    model = LoanModel(input_dim)
    model.load_state_dict(torch.load(os.path.join(artifacts_dir, 'loan_model.pth')))
    model.eval()

    # Create some test cases
    # Typical features: Gender, Married, Dependents, Education, Self_Employed, 
    # ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
    
    test_cases = [
        # Case 1: High probability (Good credit, high income)
        {
            'Gender': 'Male', 'Married': 'Yes', 'Dependents': '0', 'Education': 'Graduate',
            'Self_Employed': 'No', 'ApplicantIncome': 5000, 'CoapplicantIncome': 2000,
            'LoanAmount': 150, 'Loan_Amount_Term': 360, 'Credit_History': 1.0, 'Property_Area': 'Urban'
        },
        # Case 2: Low probability (Bad credit)
        {
            'Gender': 'Male', 'Married': 'No', 'Dependents': '0', 'Education': 'Not Graduate',
            'Self_Employed': 'No', 'ApplicantIncome': 2000, 'CoapplicantIncome': 0,
            'LoanAmount': 100, 'Loan_Amount_Term': 360, 'Credit_History': 0.0, 'Property_Area': 'Rural'
        },
        # Case 3: Edge case (Good credit but very low income/high loan)
        {
            'Gender': 'Female', 'Married': 'No', 'Dependents': '1', 'Education': 'Graduate',
            'Self_Employed': 'No', 'ApplicantIncome': 1500, 'CoapplicantIncome': 0,
            'LoanAmount': 300, 'Loan_Amount_Term': 360, 'Credit_History': 1.0, 'Property_Area': 'Semiurban'
        },
        # Case 4: No credit history (imputed as 1.0 or mean?)
        {
            'Gender': 'Female', 'Married': 'Yes', 'Dependents': '2', 'Education': 'Graduate',
            'Self_Employed': 'Yes', 'ApplicantIncome': 4000, 'CoapplicantIncome': 1000,
            'LoanAmount': 120, 'Loan_Amount_Term': 360, 'Credit_History': 0.5, 'Property_Area': 'Urban'
        }
    ]

    df = pd.DataFrame(test_cases)
    
    # Preprocess
    X = df.copy()
    cat_cols = artifacts['cat_cols']
    num_cols = artifacts['num_cols']
    
    X[cat_cols] = artifacts['imputer_cat'].transform(X[cat_cols])
    X[num_cols] = artifacts['imputer_num'].transform(X[num_cols])
    
    for col in cat_cols:
        le = artifacts['label_encoders'][col]
        X[col] = le.transform(X[col].astype(str))
        
    # Apply log transformation to highly skewed numeric columns
    skewed_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for col in skewed_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col])
            
    X_scaled = artifacts['scaler'].transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        probs = model(X_tensor).numpy()
        
    for i, case in enumerate(test_cases):
        print(f"Case {i+1} Probability: {probs[i][0]:.4f}")

if __name__ == "__main__":
    test_probabilities()
