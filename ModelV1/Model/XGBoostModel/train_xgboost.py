import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def train_xgboost_model(X_train, y_train, X_val, y_val, model_path):
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
        verbosity=0,  # Use verbosity instead of silent
        eval_metric="rmse"
    )
    history = regr.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    # Make predictions on the training set
    y_train_pred = regr.predict(X_train)
    print("XGBoost score on training set:", rmse(y_train, y_train_pred))
    print("XGBoost R^2 score on training set:", r2_score(y_train, y_train_pred))

    # Make predictions on the validation set
    y_val_pred = regr.predict(X_val)
    print("XGBoost score on validation set:", rmse(y_val, y_val_pred))
    print("XGBoost R^2 score on validation set:", r2_score(y_val, y_val_pred))

    # Save the model to disk
    joblib.dump(regr, model_path)

    return regr, y_train_pred, y_val_pred, history

def plot_xgboost_model_performance(y_train, y_train_pred):

    plt.figure(figsize=(10, 5))
    plt.scatter(y_train, y_train_pred, alpha=0.2)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.title('XGBoost Training Performance')
    plt.show()

def plot_xgboost_training_history(y_train, y_train_pred, y_val=None, y_val_pred=None):

    plt.figure(figsize=(10, 5))
    plt.scatter(y_train, y_train_pred, alpha=0.5, label='Train', color='blue')
    if y_val is not None and y_val_pred is not None:
        plt.scatter(y_val, y_val_pred, alpha=0.5, label='Validation', color='red')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.title('XGBoost Model Performance')
    plt.legend()
    plt.show()