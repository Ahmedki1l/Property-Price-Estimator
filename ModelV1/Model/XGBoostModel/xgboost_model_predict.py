import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def xgboost_load_model(model_path):
    return joblib.load(model_path)

def predict_and_evaluate(model, X_test,  assumed_r_squared=0.865):
    predictions = model.predict(X_test)

    # Calculating the prediction interval around the predictions
    lower_bounds = predictions - (predictions * (1 - assumed_r_squared))
    upper_bounds = predictions + (predictions * (1 - assumed_r_squared))
    for pred, lower, upper in zip(predictions, lower_bounds, upper_bounds):
        print(f"Prediction: {pred:.2f}, Range: ({lower:.2f}, {upper:.2f})")

    results = [{
        "prediction": int(pred),
        "lower_bound": int(lower),
        "upper_bound": int(upper)
    } for pred, lower, upper in zip(predictions, lower_bounds, upper_bounds)]

    return results
