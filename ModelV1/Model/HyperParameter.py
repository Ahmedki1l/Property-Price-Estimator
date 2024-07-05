import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras_tuner import RandomSearch
from sklearn.ensemble import RandomForestRegressor

def load_and_preprocess_data(filepath):
    print("Loading data...")
    train = pd.read_csv(filepath)
    print("Initial data shape:", train.shape)
    print("Handling missing values...")
    train.dropna(subset=['SalePrice'], inplace=True)
    train.fillna(0, inplace=True)
    train = pd.get_dummies(train, drop_first=True)
    scaler = MinMaxScaler()
    features = train.drop('SalePrice', axis=1)
    features_scaled = scaler.fit_transform(features)
    target = train['SalePrice']
    return features_scaled, target, features.columns

X, y, feature_names = load_and_preprocess_data("./data/train.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection using RandomForest
print("Starting feature selection using RandomForest feature importance...")
forest = RandomForestRegressor(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
selected_features_indices = indices[:20]  # Select top 20 features
X_train_sfs = X_train[:, selected_features_indices]
X_test_sfs = X_test[:, selected_features_indices]
selected_features_names = [feature_names[i] for i in selected_features_indices]
print("Feature selection completed. Selected features:", selected_features_names)

def build_model(hp):
    model = Sequential([
        Dense(units=hp.Int('units_first', min_value=512, max_value=2048, step=128), activation='relu',
              input_shape=(X_train_sfs.shape[1],)),
        Dropout(hp.Float('dropout_first', min_value=0.1, max_value=0.3, step=0.05)),
        Dense(units=hp.Int('units_second', min_value=256, max_value=1024, step=128), activation='relu'),
        Dropout(hp.Float('dropout_second', min_value=0.1, max_value=0.3, step=0.05)),
        Dense(units=hp.Int('units_third', min_value=128, max_value=512, step=64), activation='relu'),
        Dropout(hp.Float('dropout_third', min_value=0.1, max_value=0.3, step=0.05)),
        Dense(units=hp.Int('units_fourth', min_value=64, max_value=256, step=32), activation='relu'),
        Dropout(hp.Float('dropout_fourth', min_value=0.1, max_value=0.3, step=0.05)),
        Dense(units=hp.Int('units_fifth', min_value=32, max_value=128, step=16), activation='relu'),
        Dropout(hp.Float('dropout_fifth', min_value=0.1, max_value=0.3, step=0.05)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')), loss='mse', metrics=['mean_squared_error'])
    return model

tuner = RandomSearch(build_model, objective='val_mean_squared_error', max_trials=10, executions_per_trial=3, directory='tuning_dir', project_name='keras_tuner_demo')
print("Starting hyperparameter search...")
tuner.search(X_train_sfs, y_train, epochs=100, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)
print("Training the model with the best hyperparameters...")
history = model.fit(X_train_sfs, y_train, epochs=100, validation_split=0.2, callbacks=[ReduceLROnPlateau()])
mse = model.evaluate(X_test_sfs, y_test, verbose=0)[0]
predictions = model.predict(X_test_sfs).flatten()
r_squared = r2_score(y_test, predictions)

print("Selected Features:", selected_features_names)
print(f"Test MSE: {mse:.2f}, R-squared: {r_squared:.2%}")

import matplotlib.pyplot as plt

# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Check for signs of overfitting
if min(history.history['val_loss']) < min(history.history['loss']):
    print("No significant overfitting: Validation loss is lower than training loss at some point.")
else:
    print("Potential overfitting detected: Validation loss is consistently higher than training loss.")
