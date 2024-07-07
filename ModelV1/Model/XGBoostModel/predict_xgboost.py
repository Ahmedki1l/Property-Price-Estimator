import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score

def xgboost_load_model(model_path):
    return joblib.load(model_path)

def predict_and_evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    r_squared = r2_score(y_test, predictions)

    # Calculating the prediction interval around the predictions
    lower_bounds = predictions - (predictions * (1 - r_squared))
    upper_bounds = predictions + (predictions * (1 - r_squared))

    print(f"XGBoost R^2 score on test set: {r_squared}")
    for pred, lower, upper in zip(predictions, lower_bounds, upper_bounds):
        print(f"Prediction: {pred:.2f}, Range: ({lower:.2f}, {upper:.2f})")

    return predictions, r_squared, lower_bounds, upper_bounds
