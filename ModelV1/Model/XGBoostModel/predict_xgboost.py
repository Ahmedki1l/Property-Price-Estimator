import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def xgboost_load_model(model_path):
    return joblib.load(model_path)

def predict_and_evaluate(model, X_test, y_test, base_confidence=1.96):
    predictions = model.predict(X_test)
    test_score = rmse(y_test, predictions)
    r_squared = r2_score(y_test, predictions)

    # Adjust confidence level based on R-squared (higher R-squared, more confidence, narrower range)
    # Example adjustment: decrease the confidence interval multiplier as R-squared increases
    adjusted_confidence = base_confidence * (1 - r_squared)

    # Estimating the standard error assuming normal distribution of errors
    error_margin = adjusted_confidence * (test_score / np.sqrt(len(y_test)))

    # Calculating the prediction interval around the predictions
    lower_bounds = predictions - error_margin
    upper_bounds = predictions + error_margin

    print(f"XGBoost score on test set: {test_score}")
    print(f"XGBoost R^2 score on test set: {r_squared}")
    for pred, lower, upper in zip(predictions, lower_bounds, upper_bounds):
        print(f"Prediction: {pred:.2f}, Range: ({lower:.2f}, {upper:.2f})")

    return predictions, test_score, r_squared, lower_bounds, upper_bounds
