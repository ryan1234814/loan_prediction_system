# Project Directory Structure

```
ai_project/
├── data/
│   ├── train_u6lujuX_CVtuZ9i.csv       # Training dataset
│   └── test_Y3wMUE5_7gLdaTN.csv        # Testing dataset
├── explanation_plots/
│   ├── shap_summary_plot.png           # Global feature importance
│   ├── shap_bar_plot.png               # Ranked feature importance
│   └── shap_force_plot_instance_0.png  # Instance-specific explanation
├── responsible_ai/
│   ├── fairness_check.py               # Comprehensive fairness evaluation script
│   └── plots/
│       ├── overall_fairness_summary.png # Global summary of bias metrics
│       ├── gender_fairness.png          # Bias analysis by Gender
│       ├── education_fairness.png       # Bias analysis by Education
│       ├── married_fairness.png         # Bias analysis by Marital Status
│       ├── dependents_fairness.png      # Bias analysis by Dependents
│       ├── self_employed_fairness.png   # Bias analysis by Employment Type
│       ├── credit_history_fairness.png  # Bias analysis by Credit History
│       └── property_area_fairness.png   # Bias analysis by Property Area
├── loan_approval.py                    # Main model and SHAP script
├── requirements.txt                    # Python dependencies
├── architecture.md                     # Design documentation
└── structure.md                        # File structure documentation
```