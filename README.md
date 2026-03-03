# Loan Prediction System with Explainable AI & Fairness

A high-performance machine learning system built with **PyTorch** for loan approval prediction, integrated with **SHAP** for model explainability and a robust **Responsible AI** framework for fairness and bias assessment.

---

## 🚀 Key Features

- **PyTorch Deep Learning**: A feed-forward neural network designed for binary classification of loan eligibility.
- **Explainable AI (XAI)**: Integration with SHAP (SHapley Additive exPlanations) to provide both global and local interpretations of model decisions.
- **Fairness & Bias Auditing**: Comprehensive evaluation of model predictions across demographic subgroups (Gender, Education, Credit History, etc.) using industry-standard fairness metrics.
- **Automated Preprocessing**: Intelligent handling of missing values, categorical encoding, and feature scaling.

---

## 🏗️ Project Architecture

### 1. Data Pipeline
- **Input**: CSV-based loan application data.
- **Preprocessing**: 
  - Numerical values imputed using the mean.
  - Categorical values imputed using the mode.
  - Standard scaling for feature normalization.

### 2. Model Architecture
- **Framework**: PyTorch
- **Structure**: 3-layer Feed-Forward Neural Network (16 -> 8 -> 1 nodes).
- **Activation**: ReLU for hidden layers, Sigmoid for the output probability.

### 3. Explainability (SHAP)
- Uses `KernelExplainer` for model-agnostic explanations.
- **Summary Plots**: Visualizes global feature impact across the entire dataset.
- **Bar Plots**: Ranks features by their average impact on model decisions.
- **Force Plots**: Provides specific, instance-level explanations for individual loan applicants.

### 4. Responsible AI (Fairness)
Evaluates bias using metrics such as **Disparate Impact Ratio** and **Demographic Parity Difference**. It audits the following features:
- Gender, Marital Status, Dependents, Education, Self-Employment, Credit History, and Property Area.

---

## 📂 Directory Structure

```text
ai_project/
├── data/
│   ├── train_u6lujuX_CVtuZ9i.csv       # Training dataset
│   └── test_Y3wMUE5_7gLdaTN.csv        # Testing dataset
├── explanation_plots/                  # SHAP interpretation visuals
├── responsible_ai/
│   ├── fairness_check.py               # Main fairness auditing script
│   └── plots/                          # Visual bias analysis results
├── loan_approval.py                    # Main PyTorch model training & XAI script
├── requirements.txt                    # Python dependencies
├── architecture.md                     # Technical design document
└── README.md                           # Project overview
```

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- SHAP, Scikit-Learn, Pandas, Matplotlib

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ryan1234814/loan_prediction_system.git
   cd loan_prediction_system
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the System
1. **Train Model & Generate Explanations**:
   ```bash
   python loan_approval.py
   ```
   *This will train the PyTorch model and save SHAP plots in `explanation_plots/`.*

2. **Run Fairness Audit**:
   ```bash
   python responsible_ai/fairness_check.py
   ```
   *This will evaluate bias and save fairness visualizations in `responsible_ai/plots/`.*

---

## 📊 Visualizations

The system generates high-resolution plots to help stakeholders understand model behavior:
- **Global Feature Importance**: Identifies which factors (e.g., Credit History) drive the most approvals.
- **Individual Case Explanations**: Shows exactly why a specific loan was approved or denied.
- **Bias Assessment**: High-contrast charts highlighting any performance gaps between demographic groups.

---
*Developed as part of the Loan Prediction System research on Accurate and Responsible ML.*
