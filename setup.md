# Setup Guide: AI Loan Prediction System

This guide provides step-by-step instructions to get the AI Loan Prediction System running on your local machine. The project consists of a **Flask Backend** (AI Inference Server) and a **React Frontend** (User Dashboard).

---

## 📋 Prerequisites

Before starting, ensure you have the following installed:
- **Python 3.8 or higher**
- **Node.js (v14+) and npm**
- **Git** (optional, for cloning)

---

## 🛠️ Step 1: Backend Setup (Python & Flask)

The backend handles the machine learning inference, SHAP explanations, and model metadata.

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Activate the Virtual Environment:**
    The project uses a pre-configured virtual environment named `ai_venv` located in the root directory.
    ```bash
    # From the backend directory
    source ../ai_venv/bin/activate
    ```

3.  **Install Dependencies (if not already installed):**
    If the environment needs updating, run:
    ```bash
    pip install -r ../requirements.txt
    ```

4.  **Run the Flask Server:**
    ```bash
    python app.py
    ```
    - The server will start at `http://localhost:5001`.
    - **Note:** Ensure port 5001 is free. If not, the server will log a "Connection Refused" error in the frontend.

---

## 💻 Step 2: Frontend Setup (React & Vite)

The frontend provides the interactive UI for users to submit applications and view AI insights.

1.  **Open a new terminal window.**

2.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

3.  **Install JavaScript Dependencies:**
    ```bash
    npm install
    ```

4.  **Run the Development Server:**
    ```bash
    npm run dev
    ```
    - The frontend will typically start at `http://localhost:5173`.
    - Open your browser and navigate to the provided URL.

---

## 🧪 Step 3: Verifying the Connection

1.  Once both servers are running, the frontend will automatically attempt to fetch **Model Metadata** from the backend.
2.  If successful, the **Applicant Details** form will populate with fields like Gender, Applicant Income, etc.
3.  If you see an error "Failed to connect to backend server," verify that the Flask server is running on port 5001.

---

## 📝 Running Tests

To verify the model logic without the UI, you can use the sample test cases:

```bash
# Run a specific test script
python test_model.py
```

Refer to `sample_test_cases.txt` for input data you can copy-paste into the web form.
