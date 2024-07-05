import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score

def load_model(model_path):
    """Loads a saved Keras model from the specified path."""
    return tf.keras.models.load_model(model_path)

def predict_and_evaluate(model, X_test, y_test, confidence=1.96):
    """Makes predictions with a model, evaluates them, and calculates a confidence range adjusted by R-squared."""
    predictions = model.predict(X_test).flatten()
    mse = tf.keras.losses.MeanSquaredError()(y_test, predictions).numpy()
    r_squared = r2_score(y_test, predictions)
    std_dev = np.sqrt(mse)

    # Adjusting confidence interval based on R-squared
    # Reducing the standard deviation by a factor related to R-squared
    adjusted_std_dev = std_dev * (1 - r_squared)  # Reduce std_dev proportionally to R-squared

    # Calculate adjusted lower and upper bounds for each prediction
    lower_bounds = predictions - (confidence * adjusted_std_dev)
    upper_bounds = predictions + (confidence * adjusted_std_dev)

    print(f"Test MSE: {mse:.2f}, R-squared: {r_squared:.2%}")
    for pred, lower, upper in zip(predictions, lower_bounds, upper_bounds):
        print(f"Prediction: {pred:.2f}, Range: ({lower:.2f}, {upper:.2f})")

    return predictions, r_squared, lower_bounds, upper_bounds
