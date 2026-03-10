# AI Loan Prediction System: Technical Deep Dive

This project is an end-to-end Machine Learning application designed for **Responsible AI**. It doesn't just predict if a loan will be approved; it explains **why** using SHAP and audits the model for **fairness** across demographic groups.

---

## � File-by-File Explanation

### 1. Core Machine Learning & Training
*   **`loan_approval.py`**: The "Heart" of the model development.
    *   **Mechanism**: Uses **PyTorch** to define a 3-layer Neural Network.
    *   **Logic**: It handles the full training loop (preprocessing → training → evaluation).
    *   **SHAP Integration**: After training, it uses `shap.KernelExplainer` to generate global importance plots, saved in `/explanation_plots`.
*   **`model_artifacts/`**:
    *   `loan_model.pth`: The saved state of the trained PyTorch neural network.
    *   `preprocessing_artifacts.joblib`: Contains the **Scikit-learn** scalers, label encoders, and imputers used during training. These are critical for processing "live" data exactly like "training" data.

### 2. Backend API (Flask)
*   **`backend/app.py`**: The API wrapper.
    *   **Function**: Exposes two main endpoints: `/metadata` (to tell the frontend what inputs are needed) and `/predict` (to process form submissions).
    *   **CORS**: Uses `flask-cors` to allow communication with the React frontend.
*   **`backend/predictor.py`**: The Inference Logic.
    *   **Mechanism**: Loads the `.pth` and `.joblib` files. It converts raw JSON input from the web into processed Tensors for the model.
    *   **XAI Mechanism**: It runs a **local SHAP explanation** for every single prediction request, allowing users to see exactly which of their features (e.g., Credit History) helped or hurt their case.

### 3. Frontend Dashboard (React + Vite)
*   **`frontend/src/App.jsx`**: The interactive interface.
    *   **Technique**: Uses **React Hooks** (`useState`, `useEffect`) to manage form state and API calls.
    *   **Visualization**: Uses **Recharts** to render the horizontal bar chart showing SHAP values.
*   **`frontend/src/App.css`**: Premium styling.
    *   **Aesthetics**: Implements a **Glassmorphism** design system with vibrant gradients and smooth transitions.

### 4. Responsible AI & Auditing
*   **`responsible_ai/fairness_check.py`**: A bias-detecting auditing tool.
    *   **Technique**: Calculates metrics like **Disparate Impact Ratio** and **Statistical Parity**. It checks if the model is systemically favoring one group (e.g., Married vs. Single) over another.
*   **`responsible_ai/bias_analysis.py`**: Provides deeper statistical breakdowns and visualizes error rates across subgroups.

### 5. Utility & Test Files
*   **`sample_test_cases.txt`**: A curated list of 13 unique applicant profiles with their expected outcomes, used for system verification.
*   **`requirements.txt` / `package.json`**: Dependency manifests for Python and JavaScript respectively.

---

## 🧠 Key Concepts & Techniques Used

### 1. Neural Networks (PyTorch)
The model is a **Feed-Forward Neural Network**. Unlike simpler models (like Logistic Regression), it can capture non-linear relationships between variables like Income and Loan Amount Term.

### 2. SHAP (SHapley Additive exPlanations)
Based on game theory, SHAP assigns each feature an "importance" value for a specific prediction. 
- **Positive SHAP**: The feature pushed the probability **up** towards approval.
- **Negative SHAP**: The feature pushed the probability **down** towards rejection.

### 3. Model-Agnostic Preprocessing
To ensure the model never crashes, we use a robust pipeline:
- **Imputation**: Handles missing values (e.g., if a user leaves "Dependents" blank).
- **Label Encoding**: Converts text (e.g., "Graduate") into numbers the neural network can understand.
- **Standard Scaling**: Normalizes different scales (e.g., comparing an Income of 5000 with a Loan Term of 360).

### 4. Fairness Engineering
We focus on **Group Fairness**. The goal is to ensure the "privileged" group (e.g., those with a credit history) doesn't have an unfairly high advantage compared to the "unprivileged" group, while still maintaining high predictive accuracy.

---

## � Summary of the Data Flow
1.  **User** enters data into the **React Frontend**.
2.  **Frontend** sends a JSON POST request to the **Flask Backend**.
3.  **Predictor.py** transforms the data and feeds it to the **PyTorch Model**.
4.  **SHAP** calculates the impact of each input field.
5.  **Backend** returns the prediction value, probability, and SHAP data.
6.  **Frontend** renders the result and the interactive explanation chart.
