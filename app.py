from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the Tuned Random Forest Model
with open('tuned_random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Welcome to the Bank Customer Churn Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    
    # Extract features in the same order as they were trained
    try:
        features = np.array([
            data['credit_score'],
            data['gender'],              # Ensure it's aligned with the training columns
            data['age'],
            data['tenure'],
            data['balance'],
            data['num_of_products'],
            data['has_cr_card'],
            data['is_active_member'],
            data['estimated_salary'],
            data['geography_Germany'],   # One-hot encoded Geography for Germany
            data['geography_Spain']      # One-hot encoded Geography for Spain
        ]).reshape(1, -1)
    except KeyError as e:
        return jsonify({"error": f"Missing feature in input data: {str(e)}"}), 400
    
    # Make a prediction using the loaded model
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)[:, 1]
    
    # Return the result as a JSON response
    return jsonify({
        'prediction': int(prediction[0]),  # 0 or 1
        'churn_probability': float(prediction_proba[0])
    })

if __name__ == '__main__':
    app.run(debug=True)
