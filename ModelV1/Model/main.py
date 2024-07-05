import numpy as np
from NeuralNetworkModel.data_preparation import load_and_preprocess_data
from NeuralNetworkModel.feature_selection import feature_selection, save_features, load_features, filter_features
from NeuralNetworkModel.train_model import train_and_save_model, plot_training_history
from NeuralNetworkModel.model_usage import load_model, predict_and_evaluate
from XGBoostModel.train_xgboost import train_xgboost_model, plot_model_performance
from XGBoostModel.predict_xgboost import predict_and_evaluate as predict_and_evaluate_xgb, xgboost_load_model

def main():
    # Load and preprocess data
    X, y, feature_names = load_and_preprocess_data("./data/train.csv")

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform feature selection and save the selected features
    selected_indices, selected_features = feature_selection(X_train, y_train, feature_names, 12)
    features_path = 'selected_features16.json'
    save_features(selected_features, features_path)  # Assuming a JSON file for saving features

    # Load selected features for prediction
    features_for_prediction = load_features(features_path)

    # Filter datasets to use only selected features
    X_train_sfs = filter_features(X_train, feature_names, features_for_prediction)
    X_test_sfs = filter_features(X_test, feature_names, features_for_prediction)

    # Train and save the model, also retrieve training history
    neural_network_path = 'Models/NeuralNetwork/final_model14.keras'

    # input_shape = X_train_sfs.shape[1]  # Number of features after feature selection
    # model, history = train_and_save_model(X_train_sfs, y_train, input_shape, neural_network_path)

    # Optionally, plot the training history to check for overfitting/underfitting
    # plot_training_history(history)

    # Load the model and evaluate its performance // better is the model number 14 and feature selection number 16
    model = load_model(neural_network_path)
    predictions, r_squared, lower_bounds, upper_bounds = predict_and_evaluate(model, X_test_sfs, y_test)
    print(f"Predictions: {predictions}")
    print(f"R-squared: {r_squared:.2%}")

    # XGBoost model best model is 22 with feature selection number 16
    xgboost_path = "./Models/XGBoost/xgboost_model22.pkl"

    # regr, y_train_pred = train_xgboost_model(X_train_sfs, y_train, xgboost_path)
    # plot_model_performance(y_train, y_train_pred)

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
