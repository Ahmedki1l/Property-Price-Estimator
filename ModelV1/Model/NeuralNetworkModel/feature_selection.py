from sklearn.ensemble import RandomForestRegressor
import numpy as np

def feature_selection(X_train, y_train, feature_names, num_features=12):
    print("Starting feature selection using RandomForest feature importance...")
    forest = RandomForestRegressor(n_estimators=100, random_state=42)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    selected_indices = indices[:num_features]
    selected_features = [feature_names[i] for i in selected_indices]
    print("Feature selection completed. Selected features:", selected_features)
    return selected_indices, selected_features


def save_features(feature_names, filename='selected_features8.json'):
    with open(filename, 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")


def load_features(filename='selected_features8.json'):
    with open(filename, 'r') as f:
        features = [line.strip() for line in f.readlines()]
    return features


def filter_features(data, feature_names, selected_features):
    # Convert selected feature names back to indices.
    feature_indices = [feature_names.get_loc(feature) for feature in selected_features if feature in feature_names]
    # Use .iloc to select columns by indices.
    filtered_data = data.iloc[:, feature_indices]
    return filtered_data
