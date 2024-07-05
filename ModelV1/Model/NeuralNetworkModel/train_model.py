from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner import RandomSearch, HyperParameters

def build_model(hp, input_shape):
    model = Sequential([
        Dense(units=hp.Int('units_first', min_value=512, max_value=2048, step=128), activation='relu', input_shape=(input_shape,)),
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

def train_and_save_model(X_train, y_train, X_val, y_val, input_shape, directory):
    tuner = RandomSearch(
        lambda hp: build_model(hp, input_shape),
        objective='val_mean_squared_error',
        max_trials=10,
        executions_per_trial=3,
        project_name='keras_tuner_demo'
    )
    print("Starting hyperparameter search...")
    tuner.search(X_train, y_train, epochs=100,  validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    print("Training the model with the best hyperparameters...")
    history = model.fit(X_train, y_train, epochs=200,  validation_data=(X_val, y_val), callbacks=[ReduceLROnPlateau()])

    # Calculate R-squared for training and validation sets
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    r_squared_train = r2_score(y_train, train_predictions)
    r_squared_val = r2_score(y_val, val_predictions)

    print(f"Training R-squared: {r_squared_train:.3f}")
    print(f"Validation R-squared: {r_squared_val:.3f}")

    model.save(directory)
    return model, history, train_predictions, val_predictions

def plot_neurarl_netwok_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_neural_network_performance(y_train, y_train_pred):

    plt.figure(figsize=(10, 5))
    plt.scatter(y_train, y_train_pred, alpha=0.2)
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'k--', lw=4)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.title('Neural Network Training Performance')
    plt.show()
