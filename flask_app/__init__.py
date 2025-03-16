import os
import logging
import joblib
import numpy as np
from flask import Flask, request, render_template, jsonify

# Set up logging
logging.basicConfig(
    filename="flask_app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

# Define paths
MODEL_PATH = "models/model.pkl"
TRANSFORMER_PATH = "models/column_transformer.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# Load model and transformers
model, column_transformer, label_encoder = None, None, None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        logging.error("Model file not found.")

    if os.path.exists(TRANSFORMER_PATH):
        column_transformer = joblib.load(TRANSFORMER_PATH)
    else:
        logging.error("Column transformer file not found.")

    if os.path.exists(ENCODER_PATH):
        label_encoder = joblib.load(ENCODER_PATH)
    else:
        logging.error("Label encoder file not found.")

    if model and column_transformer and label_encoder:
        logging.info("Model and transformers loaded successfully.")
    else:
        logging.error("Failed to load all required files.")
except Exception as e:
    logging.error(f"Error loading model or transformers: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not column_transformer or not label_encoder:
        return jsonify({'error': "Model or transformers not loaded."}), 500

    try:
        # Get user input
        crop_type = request.form.get('crop')
        water_availability = request.form.get('water')

        if not crop_type or not water_availability:
            return jsonify({'error': "Missing input values."}), 400

        # Convert input to numpy array
        user_input = np.array([[crop_type, water_availability]], dtype=object)

        # Apply encoding
        transformed_input = column_transformer.transform(user_input)

        # Make prediction
        prediction = model.predict(transformed_input)

        # Decode prediction
        predicted_label = label_encoder.inverse_transform(prediction)

        # Return response
        return jsonify({'prediction': predicted_label[0]})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': "Prediction failed. Check server logs."}), 500

if __name__ == '__main__':
    app.run(debug=True)
