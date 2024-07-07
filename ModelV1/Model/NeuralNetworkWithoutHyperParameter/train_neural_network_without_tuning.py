from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import r2_score

def build_model(input_shape):
    model = Sequential([
        Dense(units=256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(units=128, activation='relu'),
        Dropout(0.2),
        Dense(units=64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mean_squared_error'])
    return model

def train_and_save_model_without_hyperparameter(X_train, y_train, X_val, y_val, input_shape, directory):
    model = build_model(input_shape)
    print("Starting model training...")
    history = model.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10),
                                   ReduceLROnPlateau()])

    # Calculate R-squared for training and validation sets
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    r_squared_train = r2_score(y_train, train_predictions)
    r_squared_val = r2_score(y_val, val_predictions)

    print(f"Training R-squared: {r_squared_train:.3f}")
    print(f"Validation R-squared: {r_squared_val:.3f}")

    # Save the model
    model.save(directory)
    return model, history, train_predictions, val_predictions
