from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the model and features we saved in the notebook
model = joblib.load('xgb_model.pkl')
features = joblib.load('feature_list.pkl')

@app.route('/')
def home():
    return "XGBoost Diagnosis API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Convert to DataFrame to ensure column order matches training
        input_df = pd.DataFrame([data])
        
        # Select only the features the model was trained on
        input_df = input_df[features]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        result = {
            "diagnosis": "Malignant" if prediction == 1 else "Benign",
            "confidence": f"{round(probability * 100, 2)}%"
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)