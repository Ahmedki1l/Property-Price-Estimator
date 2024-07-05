import numpy as np
import pandas as pd
from NeuralNetworkModel.data_preparation import load_and_preprocess_data
from NeuralNetworkModel.feature_selection import feature_selection, save_features, load_features, filter_features
from NeuralNetworkModel.train_model import train_and_save_model, plot_neurarl_netwok_training_history, plot_neural_network_performance
from NeuralNetworkModel.model_usage import load_model, predict_and_evaluate
from XGBoostModel.train_xgboost import train_xgboost_model, plot_xgboost_model_performance, plot_xgboost_training_history
from XGBoostModel.predict_xgboost import predict_and_evaluate as predict_and_evaluate_xgb, xgboost_load_model

def main():
    # Load and preprocess data
    X, y, feature_names = load_and_preprocess_data("./data/train.csv")

    # Split data
    from sklearn.model_selection import train_test_split
    X_train_and_validation, X_test, y_train_and_validation, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train_and_validation, y_train_and_validation, test_size=0.2, random_state=42)

    # Perform feature selection and save the selected features
    selected_indices, selected_features = feature_selection(X_train, y_train, feature_names, 12)
    features_path = 'selected_features17.json'
    save_features(selected_features, features_path)  # Assuming a JSON file for saving features

    # Load selected features for prediction
    features_for_prediction = load_features(features_path)

    # Filter datasets to use only selected features
    X_train_sfs = filter_features(X_train, feature_names, features_for_prediction)
    X_val_sfs = filter_features(X_val, feature_names, features_for_prediction)
    X_test_sfs = filter_features(X_test, feature_names, features_for_prediction)

    # Train and save the model, also retrieve training history
    neural_network_path = 'Models/NeuralNetwork/final_model15.keras'

    input_shape = X_train_sfs.shape[1]  # Number of features after feature selection
    model, history, train_predictions, val_predictions = train_and_save_model(X_train_sfs, y_train, X_val_sfs, y_val, input_shape, neural_network_path)
    print(y_train.describe())
    train_predictions_df = pd.DataFrame(train_predictions, columns=['Predictions'])
    print(train_predictions_df.describe())

    print(y_val.describe())
    val_predictions_df = pd.DataFrame(val_predictions, columns=['Predictions'])
    print(val_predictions_df.describe())

    # Optionally, plot the training history to check for overfitting/underfitting
    plot_neurarl_netwok_training_history(history)

    # plot_neural_network_performance(y_train, train_predictions)
    # plot_neural_network_performance(y_val, val_predictions)

    # Load the model and evaluate its performance // better is the model number 14 and feature selection number 16
    model = load_model(neural_network_path)
    predictions, r_squared, lower_bounds, upper_bounds = predict_and_evaluate(model, X_test_sfs, y_test)
    print(f"Predictions: {predictions}")
    print(f"R-squared: {r_squared:.2%}")

    # XGBoost model best model is 22 with feature selection number 16
    xgboost_path = "./Models/XGBoost/xgboost_model23.pkl"

    regr, y_train_pred, y_val_pred, xg_history = train_xgboost_model(X_train_sfs, y_train, X_val_sfs, y_val, xgboost_path)
    plot_xgboost_training_history(y_train, y_train_pred, y_val, y_val_pred)
    plot_xgboost_model_performance(y_train, y_train_pred)
    plot_xgboost_model_performance(y_val, y_val_pred)

    loaded_xgboost_model = xgboost_load_model(xgboost_path)

    xgb_predictions, xgboost_test_score, xgboost_r_squared, lower_bounds, upper_bounds = predict_and_evaluate_xgb(
        loaded_xgboost_model, X_test_sfs, y_test)

    print("XGBoost score on test set:", xgboost_test_score)
    print("XGBoost R^2 score on test set:", xgboost_r_squared)
    print("Predictions with dynamically adjusted confidence intervals:")
    # for pred, lower, upper in zip(xgb_predictions, lower_bounds, upper_bounds):
    #     print(f"Prediction: {pred:.2f}, Range: ({lower:.2f}, {upper:.2f})")

if __name__ == "__main__":
    main()
