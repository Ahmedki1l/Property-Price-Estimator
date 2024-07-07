from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from NeuralNetworkModel.model_predict import load_model, predict_with_confidence
from XGBoostModel.xgboost_model_predict import predict_and_evaluate

app = Flask(__name__)

# Load your trained model
model = load_model('Models/NeuralNetwork/final_model18.keras')

# Load the scaler
with open('scaler2.pkl', 'rb') as f:
    scaler = pickle.load(f)

def preprocess_inputs(input_data):
    # Convert input data (array) directly to scaled array
    scaled_features = scaler.transform(input_data)
    return scaled_features

@app.route('/predict/neuralnetwork', methods=['POST'])
def predict_neural_network():
    try:
        data = request.get_json(force=True)
        # Convert the input data to numpy array for prediction
        features = np.array(data['inputs']).reshape(1, -1)
        # Scale the features
        scaled_features = preprocess_inputs(features)
        # Ensure the features array shape aligns with what the model expects
        if scaled_features.shape[1] != 9:
            return jsonify({'error': 'Incorrect number of features provided. Expected 9, got {}'.format(scaled_features.shape[1])}), 400
        predictions_with_confidence = predict_with_confidence(model, scaled_features)
        # Return the predictions with confidence intervals as a JSON response
        print(predictions_with_confidence)
        return jsonify(predictions_with_confidence)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

xgboost_path = "./Models/XGBoost/xgboost_model26.pkl"

@app.route('/predict/xgboost', methods=['POST'])
def predict_xgboost():
    try:
        # Load XGBoost model each time to ensure it's initialized
        with open(xgboost_path, 'rb') as f:
            xgboost_model = pickle.load(f)

        data = request.get_json(force=True)
        features = np.array(data['inputs']).reshape(1, -1)
        scaled_features = preprocess_inputs(features)

        if scaled_features.shape[1] != 9:
            return jsonify({'error': 'Incorrect number of features provided. Expected 9, got {}'.format(scaled_features.shape[1])}), 400

        predictions_with_confidence = predict_and_evaluate(xgboost_model, scaled_features)
        return jsonify(predictions_with_confidence)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
