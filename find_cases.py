import os
import sys
import torch
import numpy as np
import pandas as pd
import joblib

# Add current dir to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from loan_approval import LoanModel

def find_moderate_cases():
    artifacts_dir = 'model_artifacts'
    artifacts = joblib.load(os.path.join(artifacts_dir, 'preprocessing_artifacts.joblib'))
    input_dim = len(artifacts['feature_names'])
    model = LoanModel(input_dim)
    model.load_state_dict(torch.load(os.path.join(artifacts_dir, 'loan_model.pth')))
    model.eval()

    # Base case: All moderate values
    base_case = {
        'Gender': 'Male', 'Married': 'Yes', 'Dependents': '0', 'Education': 'Graduate',
        'Self_Employed': 'No', 'ApplicantIncome': 5000, 'CoapplicantIncome': 0,
        'LoanAmount': 150, 'Loan_Amount_Term': 360, 'Credit_History': 1.0, 'Property_Area': 'Semiurban'
    }

    results = []
    
    # Try different Credit_History and Income combinations
    for ch in [0, 0.5, 1]:
        for inc in [1500, 3000, 5000]:
            case = base_case.copy()
            case['Credit_History'] = float(ch)
            case['ApplicantIncome'] = float(inc)
            
            # Preprocess
            df = pd.DataFrame([case])
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
                prob = model(X_tensor).numpy()[0][0]
            
            results.append({'Credit_History': ch, 'Income': inc, 'Prob': prob})

    for r in results:
        print(f"CH: {r['Credit_History']}, Income: {r['Income']} => Prob: {r['Prob']*100:.2f}%")

if __name__ == "__main__":
    find_moderate_cases()
