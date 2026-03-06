# Project Documentation: Loan Prediction System

This document provides an overview of the key files in the project and their contributions to the overall loan prediction system.

## Core Backend & ML Logic

| File/Directory | Description | Contribution |
| :--- | :--- | :--- |
| `loan_approval.py` | The main script for training the loan prediction model, including data preprocessing and evaluation. | **High**: Defines the core ML logic and trains the model used for predictions. |
| `backend/app.py` | The Flask server that provides API endpoints for the frontend to interact with. | **High**: Acts as the bridge between the ML model and the user interface. |
| `backend/predictor.py` | A utility script that handles the loading of the trained model and performs inference on user data. | **Medium**: Encapsulates the prediction logic for the backend. |
| `model_artifacts/` | Stores the trained PyTorch model (`loan_model.pth`) and preprocessing objects (`preprocessing_artifacts.joblib`). | **High**: Contains the "brain" of the application; without these, predictions cannot be made. |

## Frontend (React)

| File/Directory | Description | Contribution |
| :--- | :--- | :--- |
| `frontend/src/App.jsx` | The main React component that builds the user interface, handles form inputs, and displays results. | **High**: Provides the entire user experience and visual interface. |
| `frontend/src/main.jsx` | The entry point for the React application. | **Low**: Necessary for bootstrapping the frontend. |
| `frontend/index.html` | The base HTML file for the web application. | **Low**: Standard entry point for the Vite-based frontend. |

## Responsible AI & Explainability

| File/Directory | Description | Contribution |
| :--- | :--- | :--- |
| `responsible_ai/fairness_check.py` | Analyzes fairness metrics (e.g., disparate impact) across different demographic groups. | **Medium**: Ensures the model is ethical and unbiased before deployment. |
| `responsible_ai/bias_analysis.py` | Specifically focuses on identifying biases in the dataset and model outputs. | **Medium**: Critical for audit and trust in the AI system. |
| `responsible_ai/outputs/` | Contains visual plots and CSV summaries of fairness and bias analysis. | **Medium**: Provides the evidence for the "Responsible AI" claims. |
| `explanation_plots/` | Stores SHAP and other explainability plots generated during model analysis. | **Medium**: Helps users understand *why* a particular loan was approved or rejected. |

## Miscellaneous

| File/Directory | Description | Contribution |
| :--- | :--- | :--- |
| `requirements.txt` | Lists all Python dependencies required to run the backend and training scripts. | **Medium**: Essential for setting up the environment. |
| `frontend/package.json` | Defines the JavaScript dependencies and scripts for the frontend. | **Medium**: Essential for building and running the React app. |
| `test_model.py` | A script for testing the model's performance on a held-out test set. | **Low**: Useful for validation but not part of the live system. |
| `README.md` | General overview of the project. | **Low**: Provides high-level documentation. |
