import os
import logging
import joblib
import numpy as np
from flask import Flask, request, render_template, jsonify

# Set up logging
logging.basicConfig(filename="flask_app.log",
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# Load model and transformers
try:
    model = joblib.load("models/model.pkl")
    column_transformer = joblib.load("models/column_transformer.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    logging.info("Model and transformers loaded successfully.")
    print("✅ Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model or transformers: {e}")
    print(f"Error loading model or transformers: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from form
        crop_type = request.form['crop']
        water_availability = request.form['water']

        # Prepare input for prediction
        user_input = np.array([[crop_type, water_availability]], dtype=object)
        print("User Input:", user_input)  # Debug print

        # Apply encoding
        transformed_input = column_transformer.transform(user_input)
        print("Transformed Input:", transformed_input)  # Debug print

        # Make prediction
        prediction = model.predict(transformed_input)
        print("Prediction:", prediction)  # Debug print

        # Decode prediction
        predicted_label = label_encoder.inverse_transform(prediction)
        print("Predicted Label:", predicted_label)  # Debug print

        # Return prediction
        return jsonify({'prediction': predicted_label[0]})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)