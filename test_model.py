import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import os

# Import the model class from loan_approval
from loan_approval import LoanModel

def run_test_predictions():
    # Paths
    base_dir = os.path.abspath(os.path.dirname(__file__))
    train_path = os.path.join(base_dir, 'data', 'train_u6lujuX_CVtuZ9i.csv')
    test_path = os.path.join(base_dir, 'data', 'test_Y3wMUE5_7gLdaTN.csv')
    output_dir = os.path.join(base_dir, 'test_csv')
    
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Save Loan_IDs for the output
    test_ids = test_df['Loan_ID']
    
    # Drop Loan_ID
    train_df = train_df.drop('Loan_ID', axis=1)
    test_df = test_df.drop('Loan_ID', axis=1)
    
    target = 'Loan_Status'
    y_train_orig = train_df[target].map({'Y': 1, 'N': 0})
    X_train_df = train_df.drop(target, axis=1)
    X_test_df = test_df.copy()
    
    cat_cols = X_train_df.select_dtypes(include=['object']).columns
    num_cols = X_train_df.select_dtypes(exclude=['object']).columns
    
    print("Preprocessing...")
    # Impute
    imputer_cat = SimpleImputer(strategy='most_frequent')
    imputer_num = SimpleImputer(strategy='mean')
    
    X_train_df[cat_cols] = imputer_cat.fit_transform(X_train_df[cat_cols])
    X_train_df[num_cols] = imputer_num.fit_transform(X_train_df[num_cols])
    
    X_test_df[cat_cols] = imputer_cat.transform(X_test_df[cat_cols])
    X_test_df[num_cols] = imputer_num.transform(X_test_df[num_cols])
    
    # Encode
    for col in cat_cols:
        le = LabelEncoder()
        X_train_df[col] = le.fit_transform(X_train_df[col])
        # Handle unseen labels in test set if any (though usually we'd use a more robust encoder)
        X_test_df[col] = le.transform(X_test_df[col])
        
    # Apply log transformation to highly skewed numeric columns
    skewed_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for col in skewed_cols:
        if col in X_train_df.columns:
            X_train_df[col] = np.log1p(X_train_df[col])
        if col in X_test_df.columns:
            X_test_df[col] = np.log1p(X_test_df[col])
            
    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    
    # Convert to Tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_orig.values, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    print(f"Training model on all training data ({X_train_scaled.shape[0]} samples)...")
    model = LoanModel(X_train_scaled.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    
    print("Generating predictions for test data...")
    model.eval()
    with torch.no_grad():
        test_probs = model(X_test_t).numpy().flatten()
    
    # Threshold at 0.5
    test_preds = (test_probs >= 0.5).astype(int)
    test_status = ['Y' if p == 1 else 'N' for p in test_preds]
    
    # Create output dataframe
    results_df = pd.DataFrame({
        'Loan_ID': test_ids,
        'Probability': test_probs,
        'Loan_Status_Predicted': test_status
    })
    
    output_file = os.path.join(output_dir, 'test_predictions.csv')
    results_df.to_csv(output_file, index=False)
    
    print(f"SUCCESS: Test predictions saved to {output_file}")
    
    # Also save a summary of feature importance for the test run if needed
    # (But user just asked for outputs in csv files)

if __name__ == "__main__":
    run_test_predictions()
