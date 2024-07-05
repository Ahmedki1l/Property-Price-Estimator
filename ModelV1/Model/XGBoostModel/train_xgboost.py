import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def train_xgboost_model(X_train, y_train, model_path):
    regr = xgb.XGBRegressor(
        colsample_bytree=0.6,
        gamma=0.0,
        learning_rate=0.001,
        max_depth=3,
        min_child_weight=2,
        n_estimators=15000,
        reg_alpha=0.1,
        reg_lambda=0.8,
        subsample=1,
        seed=42,
        silent=True
    )

    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_train)
    print("XGBoost score on training set:", rmse(y_train, y_pred))
    print("XGBoost R^2 score on training set:", r2_score(y_train, y_pred))

    joblib.dump(regr, model_path)
    return regr, y_pred

def plot_model_performance(y_train, y_train_pred):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.scatter(y_train, y_train_pred, alpha=0.2)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.title('XGBoost Training Performance')
    plt.show()

