import os
import sys
import torch
import numpy as np
import pandas as pd
import shap
import joblib

# Add the parent directory to sys.path to import LoanModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from loan_approval import LoanModel

class LoanPredictor:
    def __init__(self, artifacts_dir='../model_artifacts'):
        self.artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), artifacts_dir))
        self.model = None
        self.artifacts = None
        self.explainer = None
        self.load_artifacts()

    def load_artifacts(self):
        print(f"Loading artifacts from {self.artifacts_dir}...")
        try:
            self.artifacts = joblib.load(os.path.join(self.artifacts_dir, 'preprocessing_artifacts.joblib'))
            input_dim = len(self.artifacts['feature_names'])
            self.model = LoanModel(input_dim)
            self.model.load_state_dict(torch.load(os.path.join(self.artifacts_dir, 'loan_model.pth')))
            self.model.eval()
            print("Model and artifacts loaded successfully.")
            
            # Initialize SHAP explainer
            # We need some background data to initialize KernelExplainer
            # In a real app, you might want to save a sample of training data for this.
            # For now, let's use a dummy background or just wait for the first prediction to init if needed.
            # But SHAP KernelExplainer needs background data.
            # Let's load the training data to get a background sample.
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            train_path = os.path.join(base_dir, 'data', 'train_u6lujuX_CVtuZ9i.csv')
            train_df = pd.read_csv(train_path)
            if 'Loan_ID' in train_df.columns:
                train_df = train_df.drop('Loan_ID', axis=1)
            if 'Loan_Status' in train_df.columns:
                train_df = train_df.drop('Loan_Status', axis=1)
            
            # Preprocess background
            X_bg = self.preprocess_input(train_df)
            background = shap.sample(X_bg, 50)
            
            def model_predict_wrapper(x):
                x_tensor = torch.tensor(x, dtype=torch.float32)
                with torch.no_grad():
                    outputs = self.model(x_tensor)
                return outputs.numpy().flatten()
                
            self.explainer = shap.KernelExplainer(model_predict_wrapper, background)
            print("SHAP explainer initialized.")

        except Exception as e:
            print(f"Error loading artifacts: {e}")

    def preprocess_input(self, df):
        # Create a copy to avoid modifying original
        X = df.copy()
        
        # Ensure all columns are present
        for col in self.artifacts['feature_names']:
            if col not in X.columns:
                X[col] = np.nan
        
        # Select and reorder columns
        X = X[self.artifacts['feature_names']]
        
        # Impute
        cat_cols = self.artifacts['cat_cols']
        num_cols = self.artifacts['num_cols']
        
        X[cat_cols] = self.artifacts['imputer_cat'].transform(X[cat_cols])
        X[num_cols] = self.artifacts['imputer_num'].transform(X[num_cols])
        
        # Label Encode
        for col in cat_cols:
            le = self.artifacts['label_encoders'][col]
            try:
                # Need to handle potential strings that were not in training
                # Convert to string if expected by LabelEncoder
                X[col] = X[col].astype(str)
                X[col] = le.transform(X[col])
            except Exception:
                # If unknown category, use the most frequent (mode) or first class
                X[col] = le.transform([le.classes_[0]] * len(X))
                
        # Apply log transformation to highly skewed numeric columns
        skewed_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
        for col in skewed_cols:
            if col in X.columns:
                X[col] = np.log1p(X[col])
                
        # Scale
        X_scaled = self.artifacts['scaler'].transform(X)
        return X_scaled

    def predict(self, input_data):
        """
        input_data: list of dictionaries
        """
        df = pd.DataFrame(input_data)
        X_scaled = self.preprocess_input(df)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            probs = self.model(X_tensor).numpy()
            
        predictions = (probs > 0.5).astype(int)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_scaled)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # For KernelExplainer on single output, it might return a list [values]
            shap_values = shap_values[0]
        
        # Determine base value (expected value)
        base_val = self.explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 0:
            # If it's a list or array, take the first element (scalar)
            # If it's a list of arrays, handle that too
            if isinstance(base_val[0], (np.ndarray, list)):
                base_val = base_val[0][0]
            else:
                base_val = base_val[0]
        
        results = []
        for i in range(len(input_data)):
            # Convert SHAP values to a dict for response
            explanation = {}
            for j, feat in enumerate(self.artifacts['feature_names']):
                val = shap_values[i][j]
                # Ensure it's a float scalar
                if isinstance(val, (np.ndarray, list)):
                    val = val[0]
                explanation[feat] = float(val)
            
            results.append({
                'prediction': 'Approved' if predictions[i][0] == 1 else 'Rejected',
                'probability': float(probs[i][0]),
                'explanation': explanation,
                'base_value': float(base_val)
            })
            
        return results
