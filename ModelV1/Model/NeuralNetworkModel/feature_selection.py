import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def feature_selection(X_train, y_train, feature_names, num_features=12, correlation_threshold=0.8):
    print("Starting feature selection using RandomForest feature importance...")
    # Train the RandomForestRegressor to get feature importances
    forest = RandomForestRegressor(n_estimators=100, random_state=42)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_

    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    selected_indices = indices[:num_features]
    selected_features = [feature_names[i] for i in selected_indices]

    # Ensure proper DataFrame handling if X_train is a DataFrame
    if isinstance(X_train, pd.DataFrame):
        X_selected = X_train.iloc[:, selected_indices]
    else:  # Assuming X_train is a numpy array
        X_selected = X_train[:, selected_indices]

    df_selected = pd.DataFrame(X_selected, columns=selected_features)

    # Calculate correlation matrix
    corr_matrix = df_selected.corr().abs()

    # Filter out highly correlated features
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    reduced_feature_names = [feature for feature in selected_features if feature not in to_drop]
    reduced_feature_names

    # Extracting the final reduced features
    reduced_features_values = df_selected[reduced_feature_names].values

    print("Feature selection completed. Reduced features after removing highly correlated ones:", reduced_feature_names)

    # Plotting feature importances
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(reduced_feature_names)), importances[selected_indices][:len(reduced_feature_names)], color='b',
            align='center')
    plt.xticks(range(len(reduced_feature_names)), reduced_feature_names, rotation=90)
    plt.tight_layout()
    plt.show()

    return reduced_features_values, reduced_feature_names


def save_features(feature_names, filename='selected_features8.json'):
    with open(filename, 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")


def load_features(filename='selected_features8.json'):
    with open(filename, 'r') as f:
        features = [line.strip() for line in f.readlines()]
    return features


def filter_features(data, feature_names, selected_feature_names):
    # Convert feature_names to a list if it is a pandas Index
    if isinstance(feature_names, pd.Index):
        feature_names = feature_names.tolist()

    # Convert numpy array to DataFrame if not already a DataFrame
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=feature_names)

    # Determine indices of selected features
    feature_indices = [feature_names.index(feature) for feature in selected_feature_names if feature in feature_names]

    # Filter the features in the DataFrame
    filtered_data = data.iloc[:, feature_indices]
    return filtered_data

