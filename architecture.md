# Project Architecture

## 1. Data Pipeline
- **Input**: CSV data containing applicant details (Income, Credit History, etc.).
- **Preprocessing**: 
  - Imputation of missing values (mean for numerical, mode for categorical).
  - Categorical encoding into numerical formats.
  - Standard Scaling for model input consistency.

## 2. Model Architecture (PyTorch)
- **Type**: Feed-Forward Neural Network.
- **Hidden Layers**:
  - Linear (16 nodes) + ReLU
  - Linear (8 nodes) + ReLU
- **Output Layer**: Sigmoid for Probability Score (0 to 1).

## 3. Explainability (SHAP)
- **Method**: KernelExplainer (Model-agnostic).
- **Visualizations**:
  - **Summary Plot**: Global feature impact distribution.
  - **Bar Plot**: Ranked feature importance.
  - **Force Plot**: Local explanation for individual instances.

## 4. Responsible AI (Comprehensive Fairness Checks)
- **Script**: `responsible_ai/fairness_check.py`.
- **Scope**: Evaluates bias across all sensitive and categorical features:
  - **Gender**, **Married**, **Dependents**, **Education**, **Self_Employed**, **Credit_History**, and **Property_Area**.
- **Metrics**: 
  - **Disparate Impact Ratio**: Target > 0.8 (following the 80% rule).
  - **Demographic Parity Difference**: Measures selection rate gaps.
  - **Per-Group Metrics**: Accuracy and Selection Rate per specific subgroup.
- **Visualizations**:
  - Individual bar plots for each feature in `responsible_ai/plots/`.
  - Global `overall_fairness_summary.png` comparing all features.