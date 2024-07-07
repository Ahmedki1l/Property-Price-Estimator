import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score

def load_model(model_path):
    """Loads a saved Keras model from the specified path."""
    return tf.keras.models.load_model(model_path)

def predict_with_confidence(model, X_input, assumed_r_squared=0.8614):
    """Makes predictions with a model and calculates a confidence range based on a fixed R-squared and assumed standard deviation."""

    predictions = model.predict(X_input).flatten()

    # Calculate adjusted lower and upper bounds for each prediction
    lower_bounds = predictions + (predictions * (1 - assumed_r_squared))
    upper_bounds = predictions + (predictions * (1 - assumed_r_squared))*2

    results = [{
        "prediction": int(pred),
        "lower_bound": int(lower),
        "upper_bound": int(upper)
    } for pred, lower, upper in zip(lower_bounds, predictions, upper_bounds)]

    return results
