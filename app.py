import os
from flask import Flask, request, jsonify # type: ignore
import numpy as np
import pickle

app = Flask(__name__)

# Get the absolute path to the model file
model_file_path = os.path.abspath('tuned_random_forest_model.pkl')
print(f"Model file path: {model_file_path}")

# Check if the file exists before loading
if not os.path.exists(model_file_path):
    print("Model file not found!")
else:
    # Load the Tuned Random Forest Model
    with open(model_file_path, 'rb') as file:
        model = pickle.load(file)

@app.route('/')
def home():
    return "Welcome to the Bank Customer Churn Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        features = np.array([
            data['credit_score'],
            data['gender'],
            data['age'],
            data['tenure'],
            data['balance'],
            data['num_of_products'],
            data['has_cr_card'],
            data['is_active_member'],
            data['estimated_salary'],
            data['geography_Germany'],
            data['geography_Spain']
        ]).reshape(1, -1)
    except KeyError as e:
        return jsonify({"error": f"Missing feature in input data: {str(e)}"}), 400
    
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)[:, 1]
    
    return jsonify({
        'prediction': int(prediction[0]),
        'churn_probability': float(prediction_proba[0])
    })

if __name__ == '__main__':
    app.run(debug=True)
