from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import LoanPredictor
import os

app = Flask(__name__)
# Enable CORS for frontend, let's just keep it simple with * for now
CORS(app)

# Initialize predictor
predictor = LoanPredictor(artifacts_dir='../model_artifacts')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Expecting a list of records or a single record
        if isinstance(data, dict):
            input_data = [data]
        else:
            input_data = data
        
        results = predictor.predict(input_data)
        return jsonify(results)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/metadata', methods=['GET'])
def get_metadata():
    """
    Return cat_cols and num_cols so the frontend knows what inputs to provide
    """
    try:
        if predictor.artifacts is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get categorical class names for better form labels/select options
        cat_options = {}
        for col, le in predictor.artifacts['label_encoders'].items():
            cat_options[col] = le.classes_.tolist()
            
        metadata = {
            'cat_cols': predictor.artifacts['cat_cols'],
            'num_cols': predictor.artifacts['num_cols'],
            'cat_options': cat_options
        }
        return jsonify(metadata)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Running on port 5001 by default
    app.run(debug=True, port=5001)
