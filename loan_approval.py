import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt
import os


# 3. Define PyTorch Model
class LoanModel(nn.Module):
    def __init__(self, input_dim):
        super(LoanModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

def preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Drop Loan_ID
    train_df = train_df.drop('Loan_ID', axis=1)
    test_df = test_df.drop('Loan_ID', axis=1)
    
    target = 'Loan_Status'
    y = train_df[target].map({'Y': 1, 'N': 0})
    X = train_df.drop(target, axis=1)
    
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(exclude=['object']).columns
    
    imputer_cat = SimpleImputer(strategy='most_frequent')
    imputer_num = SimpleImputer(strategy='mean')
    
    X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])
    X[num_cols] = imputer_num.fit_transform(X[num_cols])
    
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, X_scaled, y

def train_model(X_train, y_train, X_val, y_val):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    
    model = LoanModel(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    
    return model

def run_shap_explanations(model, X_train, X_val, feature_names):
    model.eval()
    def model_predict_wrapper(x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(x_tensor)
        return outputs.numpy()

    print("  Calculating SHAP values (this may take a minute)...")
    background = shap.sample(X_train, 50) 
    explainer = shap.KernelExplainer(model_predict_wrapper, background)
    
    # We'll explain a small sample of the validation set for performance
    test_sample = X_val[:20]
    shap_values = np.squeeze(explainer.shap_values(test_sample))
    
    os.makedirs('explanation_plots', exist_ok=True)
    plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})
    
    # 1. Summary Plot (Important Features Overall + Direction)
    print("  Generating Summary Plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, test_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Global Summary Plot: Feature Impact on Loan Approval", fontsize=16, pad=20)
    plt.savefig('explanation_plots/shap_summary_plot.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Global Feature Importance Ranking (Bar Plot)
    print("  Generating Bar Plot (Importance Ranking)...")
    exp_val = explainer.expected_value
    if isinstance(exp_val, (list, np.ndarray)) and len(exp_val) > 0:
        exp_val = exp_val[0]
        
    explanation = shap.Explanation(
        values=shap_values, 
        base_values=np.array([exp_val]*len(shap_values)), 
        data=test_sample, 
        feature_names=feature_names
    )
    
    plt.figure(figsize=(12, 8))
    shap.plots.bar(explanation, show=False)
    plt.title("Global Feature Importance Ranking", fontsize=16, pad=20)
    plt.savefig('explanation_plots/shap_feature_importance_ranking.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Local Explainability: Force Plot (Individual Applicant)
    print("  Generating Force Plot for instance 0...")
    instance_idx = 0
    # Use matplotlib for the force plot to save it
    shap.force_plot(
        exp_val, 
        shap_values[instance_idx], 
        test_sample[instance_idx], 
        feature_names=feature_names, 
        matplotlib=True, 
        show=False
    )
    plt.title(f"SHAP Local Force Plot: Applicant {instance_idx}", fontsize=14, pad=30)
    plt.savefig(f'explanation_plots/shap_force_plot_applicant_{instance_idx}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 4. Local Explainability: Waterfall Plot (Individual Applicant)
    print("  Generating Waterfall Plot for instance 0...")
    plt.figure(figsize=(12, 8))
    # Create an explanation object for a single instance
    single_explanation = shap.Explanation(
        values=shap_values[instance_idx],
        base_values=exp_val,
        data=test_sample[instance_idx],
        feature_names=feature_names
    )
    shap.plots.waterfall(single_explanation, show=False)
    plt.title(f"SHAP Waterfall Plot: Contribution Analysis (Applicant {instance_idx})", fontsize=14, pad=20)
    plt.savefig(f'explanation_plots/shap_waterfall_plot_applicant_{instance_idx}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 5. Local Explainability: Feature Contribution Table
    print("  Generating Feature Contribution Table...")
    contributions = pd.DataFrame({
        'Feature': feature_names,
        'Feature Value': test_sample[instance_idx],
        'SHAP Value (Contribution)': shap_values[instance_idx]
    })
    # Sort by absolute contribution
    contributions['Abs Contribution'] = contributions['SHAP Value (Contribution)'].abs()
    contributions = contributions.sort_values(by='Abs Contribution', ascending=False).drop('Abs Contribution', axis=1)
    contributions.to_csv('explanation_plots/feature_contribution_table.csv', index=False)
    print(f"  Feature contribution table saved to explanation_plots/feature_contribution_table.csv")

if __name__ == "__main__":
    # Determine absolute paths relative to this script's location
    base_dir = os.path.abspath(os.path.dirname(__file__))
    train_path = os.path.join(base_dir, 'data', 'train_u6lujuX_CVtuZ9i.csv')
    test_path = os.path.join(base_dir, 'data', 'test_Y3wMUE5_7gLdaTN.csv')

    print("Preprocessing data...")
    X_orig, X_scaled, y = preprocess_data(train_path, test_path)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"Training model on {X_train.shape[0]} samples...")
    model = train_model(X_train, y_train, X_val, y_val)
    
    print("Running SHAP explanations...")
    run_shap_explanations(model, X_train, X_val, X_orig.columns.tolist())
    print("-" * 30)
    print("SUCCESS: All SHAP and Explainability outputs generated.")
    print("Check 'explanation_plots/' directory for results.")
    print("-" * 30)
