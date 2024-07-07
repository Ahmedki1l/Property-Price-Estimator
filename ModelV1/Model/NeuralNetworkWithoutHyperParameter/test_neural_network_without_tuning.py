import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score

def load_model(model_path):
    """Loads a saved Keras model from the specified path."""
    return tf.keras.models.load_model(model_path)

def predict_and_evaluate_without_hyperparameter(model, X_test, y_test):
    """Makes predictions with a model, evaluates them, and calculates a confidence range adjusted by R-squared."""
    predictions = model.predict(X_test).flatten()
    r_squared = r2_score(y_test, predictions)

    # Calculate adjusted lower and upper bounds for each prediction
    lower_bounds = predictions - (predictions * (1 - r_squared))
    upper_bounds = predictions + (predictions * (1 - r_squared))

    print(f"R-squared: {r_squared:.2%}")
    for pred, lower, upper in zip(predictions, lower_bounds, upper_bounds):
        print(f"Prediction: {pred:.2f}, Range: ({lower:.2f}, {upper:.2f})")

    return predictions, r_squared, lower_bounds, upper_bounds