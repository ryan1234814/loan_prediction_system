import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from loan_approval import preprocess_data, train_model

def calculate_extended_metrics(y_true, y_pred, sensitive_feature):
    # Performance by group
    groups = np.unique(sensitive_feature)
    metrics = {}
    
    for group in groups:
        mask = (sensitive_feature == group)
        if not np.any(mask):
            continue
            
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        metrics[group] = {
            'accuracy': accuracy_score(y_t, y_p),
            'precision': precision_score(y_t, y_p, zero_division=0),
            'recall': recall_score(y_t, y_p, zero_division=0),
            'f1': f1_score(y_t, y_p, zero_division=0),
            'selection_rate': np.mean(y_p),
            'true_positive_rate': recall_score(y_t, y_p, zero_division=0) if np.sum(y_t) > 0 else 0,
            'count': len(y_t)
        }
    
    # Calculate Fairness Differences (between the groups with min and max selection rates)
    if len(groups) >= 2:
        # Sort groups by selection rate to find the most and least favored groups
        sorted_groups = sorted(metrics.keys(), key=lambda g: metrics[g]['selection_rate'])
        g_min = sorted_groups[0]
        g_max = sorted_groups[-1]
        
        sr_min = metrics[g_min]['selection_rate']
        sr_max = metrics[g_max]['selection_rate']
        tpr_min = metrics[g_min]['true_positive_rate']
        tpr_max = metrics[g_max]['true_positive_rate']
        
        disparate_impact = sr_min / sr_max if sr_max > 0 else 1.0
        demographic_parity_diff = abs(sr_max - sr_min)
        equal_opp_diff = abs(tpr_max - tpr_min)
    else:
        disparate_impact, demographic_parity_diff, equal_opp_diff = 1.0, 0.0, 0.0
        
    return metrics, disparate_impact, demographic_parity_diff, equal_opp_diff

def run_sensitivity_analysis(model, X_val, feature_names, output_dir):
    """Simple feature sensitivity analysis using Partial Dependence Plots logic."""
    print("  Running Feature Sensitivity Analysis (PDP)...")
    model.eval()
    
    # Analyze all numeric features for sensitivity
    target_features = ['Credit_History', 'ApplicantIncome', 'LoanAmount', 'CoapplicantIncome', 'Loan_Amount_Term']
    
    for feat in target_features:
        if feat not in feature_names:
            continue
            
        idx = feature_names.index(feat)
        feat_min = X_val[:, idx].min()
        feat_max = X_val[:, idx].max()
        
        # Create 50 points across the range
        grid = np.linspace(feat_min, feat_max, 50)
        avg_preds = []
        
        for val in grid:
            X_temp = X_val.copy()
            X_temp[:, idx] = val
            X_temp_t = torch.tensor(X_temp, dtype=torch.float32)
            with torch.no_grad():
                preds = model(X_temp_t).numpy()
            avg_preds.append(np.mean(preds))
            
        plt.figure(figsize=(10, 6))
        plt.plot(grid, avg_preds, linewidth=3, color='#2c3e50')
        plt.title(f"Sensitivity Analysis: Model Output vs {feat}", fontsize=14)
        plt.xlabel(f"Scaled {feat}")
        plt.ylabel("Mean Predicted Probability")
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/sensitivity_{feat.lower()}.png', bbox_inches='tight')
        plt.close()

def run_bias_analysis():
    print("Initializing Responsible AI Analysis...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_path = os.path.join(base_dir, 'data', 'train_u6lujuX_CVtuZ9i.csv')
    test_path = os.path.join(base_dir, 'data', 'test_Y3wMUE5_7gLdaTN.csv')
    
    output_dir = 'responsible_ai/outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    X_orig, X_scaled, y = preprocess_data(train_path, test_path)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print("Training model for analysis...")
    model = train_model(X_train, y_train, X_val, y_val)
    model.eval()
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        probs = model(X_val_t).numpy()
    y_pred = (probs > 0.5).astype(int).flatten()
    
    # Include all features requested by the user (except Loan_ID as it's a unique identifier)
    sensitive_features = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
        'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ]
    
    print("\nCalculating Fairness Metrics for all features...")
    fairness_results = []
    indices_val = y_val.index
    
    for feat in sensitive_features:
        if feat not in X_orig.columns:
            print(f"  Skipping {feat} - not found in data.")
            continue
            
        sensitive_data = X_orig.loc[indices_val, feat]
        
        # For continuous features, bin into quartiles for fairness analysis
        is_continuous = False
        if feat in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']:
            is_continuous = True
            # Create discrete bins (Quartiles) for continuous data
            try:
                sensitive_data_binned = pd.qcut(sensitive_data, q=4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')
                analysis_data = sensitive_data_binned
            except ValueError:
                # Fallback if too few unique values
                analysis_data = sensitive_data
        else:
            analysis_data = sensitive_data
            
        metrics, di, dpd, eod = calculate_extended_metrics(y_val.values, y_pred, analysis_data)
        
        fairness_results.append({
            'Feature': feat,
            'Disparate Impact': di,
            'Demographic Parity Diff': dpd,
            'Equal Opportunity Diff': eod,
            'Is_Continuous': is_continuous
        })
        
        # Plotting selection rate by group
        plt.figure(figsize=(10, 6))
        groups = [str(g) for g in metrics.keys()]
        rates = [metrics[g]['selection_rate'] for g in metrics.keys()]
        
        # Use a consistent color palette
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(groups)))
        plt.bar(groups, rates, color=colors)
        plt.axhline(y=0.8 * max(rates) if rates else 0, color='r', linestyle='--', label='80% Fairness Threshold')
        
        plt.title(f"Loan Selection Rate by {feat} {'(Binned)' if is_continuous else ''}")
        plt.ylabel("Selection Rate (Approval Probability)")
        plt.xlabel(feat)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.savefig(f'{output_dir}/bias_{feat.lower()}.png', bbox_inches='tight')
        plt.close()

    # Save Fairness Table
    df_fairness = pd.DataFrame(fairness_results)
    df_fairness.to_csv(f'{output_dir}/fairness_metrics_summary.csv', index=False)
    print(f"Fairness metrics for all features saved to {output_dir}/fairness_metrics_summary.csv")
    
    # Run Sensitivity Analysis
    run_sensitivity_analysis(model, X_val, X_orig.columns.tolist(), output_dir)
    
    print("\n" + "="*50)
    print("RESPONSIBLE AI ANALYSIS COMPLETE")
    print(f"Full report for {len(fairness_results)} features generated.")
    print(f"Results saved in: {output_dir}")
    print("="*50)

if __name__ == "__main__":
    run_bias_analysis()
