import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from loan_approval import preprocess_data, train_model

def calculate_fairness_metrics(y_true, y_pred, sensitive_feature):
    # Performance by group
    groups = np.unique(sensitive_feature)
    metrics = {}
    
    for group in groups:
        mask = (sensitive_feature == group)
        acc = accuracy_score(y_true[mask], y_pred[mask])
        prec = precision_score(y_true[mask], y_pred[mask], zero_division=0)
        rec = recall_score(y_true[mask], y_pred[mask], zero_division=0)
        selection_rate = np.mean(y_pred[mask])
        
        metrics[group] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'selection_rate': selection_rate
        }
    
    # Disparate Impact (Ratio of selection rates)
    # Demographic Parity Difference
    if len(groups) >= 2:
        sr0 = metrics[groups[0]]['selection_rate']
        sr1 = metrics[groups[1]]['selection_rate']
        disparate_impact = min(sr0, sr1) / max(sr0, sr1) if max(sr0, sr1) > 0 else 1.0
        demographic_parity_diff = abs(sr0 - sr1)
    else:
        disparate_impact = 1.0
        demographic_parity_diff = 0.0
        
    return metrics, disparate_impact, demographic_parity_diff

def run_fairness_check():
    # 1. Load and preprocess
    # Determine absolute paths relative to this script's location
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_path = os.path.join(base_dir, 'data', 'train_u6lujuX_CVtuZ9i.csv')
    test_path = os.path.join(base_dir, 'data', 'test_Y3wMUE5_7gLdaTN.csv')
    
    X_orig, X_scaled, y = preprocess_data(train_path, test_path)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 2. Train/Load Model
    print("Training/Loading model for comprehensive fairness evaluation...")
    model = train_model(X_train, y_train, X_val, y_val)
    model.eval()
    
    # 3. Predict
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        # Get binary predictions
        probs = model(X_val_t).numpy()
    y_pred = (probs > 0.5).astype(int).flatten()
    
    # 4. Evaluate Fairness for multiple features
    sensitive_features = [
        'Gender', 'Married', 'Dependents', 
        'Education', 'Self_Employed', 'Credit_History', 
        'Property_Area'
    ]
    
    summary_results = []
    
    os.makedirs('responsible_ai/plots', exist_ok=True)
    indices_val = y_val.index
    
    print("\n" + "="*50)
    print("COMPREHENSIVE FAIRNESS EVALUATION REPORT")
    print("="*50)

    for feature in sensitive_features:
        if feature not in X_orig.columns:
            continue
            
        sensitive_data = X_orig.loc[indices_val, feature]
        metrics, di, dpd = calculate_fairness_metrics(y_val.values, y_pred, sensitive_data)
        
        print(f"\n--- Feature: {feature} ---")
        for group, m in metrics.items():
            print(f"  Group [{group}]: Selection Rate={m['selection_rate']:.3f}, Accuracy={m['accuracy']:.3f}")
        print(f"  Disparate Impact: {di:.3f}")
        print(f"  Demographic Parity Difference: {dpd:.3f}")
        
        summary_results.append({
            'Feature': feature,
            'Disparate Impact': di,
            'DP Difference': dpd
        })
        
        # Plot for each feature
        plt.figure(figsize=(10, 6))
        groups = [str(g) for g in metrics.keys()]
        rates = [metrics[g]['selection_rate'] for g in metrics.keys()]
        plt.bar(groups, rates, color=plt.cm.Paired(np.arange(len(groups))))
        plt.axhline(y=max(rates)*0.8, color='r', linestyle='--', label='80% Rule (DI Threshold)')
        plt.title(f"Loan Selection Rate by {feature}")
        plt.ylabel("Selection Rate")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.savefig(f'responsible_ai/plots/{feature.lower()}_fairness.png', bbox_inches='tight')
        plt.close()

    # 5. Global Fairness Summary Plot
    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        plt.figure(figsize=(12, 6))
        plt.bar(df_summary['Feature'], df_summary['Disparate Impact'], color='skyblue')
        plt.axhline(y=0.8, color='r', linestyle='--', label='Fairness Threshold (0.8)')
        plt.title("Disparate Impact Score across all Features")
        plt.ylabel("Disparate Impact Ratio")
        plt.xticks(rotation=45)
        plt.ylim(0, 1.2)
        plt.legend()
        plt.savefig('responsible_ai/plots/overall_fairness_summary.png', bbox_inches='tight')
        plt.close()

    print("\n" + "="*50)
    print("Fairness visualizations saved in: responsible_ai/plots/")
    print("="*50)

if __name__ == "__main__":
    run_fairness_check()
