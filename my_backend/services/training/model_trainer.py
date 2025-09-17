"""
Model trainer module for training system
Contains all ML model training functions extracted from training_backend_test_2.py
EXACT COPY from original file to preserve functionality
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TensorFlow not available: {e}")
    TENSORFLOW_AVAILABLE = False
    # Create dummy classes for type hints
    Sequential = None
    Dense = None
    LSTM = None
    Conv1D = None
    MaxPooling1D = None
    Flatten = None
    Dropout = None
    EarlyStopping = None
from typing import Dict, List, Tuple, Optional, Any
import logging

from .config import MDL

logger = logging.getLogger(__name__)

# ACTIVATION FUNCTION MAPPING
# Maps string names to TensorFlow activation functions
# Exactly as used in training_original.py
ACTIVATION_FUNCTIONS = {
    'relu': 'relu',
    'ReLU': 'relu',
    'sigmoid': 'sigmoid',
    'tanh': 'tanh',
    'softmax': 'softmax',
    'linear': 'linear',
    'elu': 'elu',
    'selu': 'selu',
    'softplus': 'softplus',
    'softsign': 'softsign',
    'swish': 'swish',
    'gelu': 'gelu'
}

def get_activation(activation_name: str) -> str:
    """
    Get TensorFlow activation function from string name.
    
    Args:
        activation_name: Name of activation function
        
    Returns:
        TensorFlow activation function name
    """
    return ACTIVATION_FUNCTIONS.get(activation_name, activation_name.lower())


def train_dense(train_x, train_y, val_x, val_y, MDL):    
    """
    Funktion trainiert und validiert ein Neuronales Netz anhand der 
    eingegebenen Trainingsdaten (train_x, train_y) und Validierungsdaten 
    (val_x, val_y).
    
    train_x...Trainingsdaten (Eingabedaten)
    train_y...Trainingsdaten (Ausgabedaten)
    val_x.....Validierungsdaten (Eingabedaten)
    val_y.....Validierungsdaten (Ausgabedaten)
    MDL.......Informationen zum Modell
    
    Extracted from training_backend_test_2.py lines 170-238
    """
    
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available, skipping dense neural network training")
        return None, float('inf'), 0
    
    # Validation: Check if we have enough features to train Dense network
    if train_x.size == 0 or len(train_x.shape) < 3:
        logger.error(f"train_x is empty or has invalid shape: {train_x.shape}")
        return None, float('inf'), 0
        
    if train_x.shape[2] == 0:
        logger.error(f"Cannot train Dense network with 0 features. Input shape: {train_x.shape}")
        return None, float('inf'), 0
       
    # MODELLDEFINITION ########################################################
    
    # Modellinitialisierung (Sequentielles Modell mit linear 
    # hintereinandergeordneten Schichten)
    model = tf.keras.Sequential()
    
    # Input-Schicht → Mehrdimensionale Daten werden in einen 1D-Vektor 
    # umgewandelt
    model.add(tf.keras.layers.Flatten())
    
    # Dense-Layer hinzufügen
    for _ in range(MDL.LAY):
        model.add(tf.keras.layers.Dense(MDL.N,                  # Anzahl an Neuronen
                                        activation = MDL.ACTF.lower() if MDL.ACTF == "ReLU" else MDL.ACTF)) # Aktivierungsfunktion
    
    # Output-Schicht
    model.add(tf.keras.layers.Dense(train_y.shape[1]*train_y.shape[2], 
                                  kernel_initializer = tf.initializers.zeros))
    model.add(tf.keras.layers.Reshape([train_y.shape[1], train_y.shape[2]]))
    
    """
    Folgender Callback sorgt dafür, dass das Training vorzeitig gestoppt wird, 
    wenn sich die Leistung auf den Validierungsdaten (val_loss) nach einer 
    bestimmten Anzahl von Epochen nicht verbessert. Dies hilft, Overfitting zu 
    vermeiden und das Training effizienter zu gestalten.
    """
    
    earlystopping = tf.keras.callbacks.\
        EarlyStopping(monitor  = "val_loss", 
        mode                   = "min", 
        patience               = 2, 
        restore_best_weights   = True)

    # Konfiguration des Modells für das Training    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss = tf.keras.losses.MeanSquaredError(),
        metrics = [tf.keras.metrics.RootMeanSquaredError()])
    
    # TRAINIEREN ##############################################################
    
    print("Modell wird trainiert.")
    
    model.fit(
        x               = train_x,
        y               = train_y,
        epochs          = MDL.EP,
        verbose         = 1,
        callbacks       = [earlystopping],
        validation_data = (val_x, val_y)
        )
         
    print("Modell wurde trainiert.")
            
    return model


def train_cnn(train_x, train_y, val_x, val_y, MDL):    
    """
    Funktion trainiert und validiert ein CNN anhand der 
    eingegebenen Trainingsdaten (train_x, train_y) und Validierungsdaten 
    (val_x, val_y).
    
    train_x...Trainingsdaten (Eingabedaten)
    train_y...Trainingsdaten (Ausgabedaten)
    val_x.....Validierungsdaten (Eingabedaten)
    val_y.....Validierungsdaten (Ausgabedaten)
    MDL.......Informationen zum Modell
    """
    
    # MODELLDEFINITION ########################################################
    
    # Modellinitialisierung (Sequentielles Modell mit linear 
    # hintereinandergeordneten Schichten)    
    model = tf.keras.Sequential()
       
    # Konverterierung in ein 4D-Array (Conv2D-Layer erwartet Eingabedaten mit vier Dimensionen)
    # - batch_size: Anzahl der Trainingsbeispiele
    # - Höhe und Breite: räumliche Dimensionen deiner Eingabedaten
    # - Kanäle: Anzahl der Kanäle pro Pixel (z. B. 3 für RGB-Bilder, 1 für Graustufen)
    # NOTE: Original has bug here - uses trn_x instead of train_x
    # We'll keep it EXACTLY as original with the variable name fix for it to work
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
    
    
    for i in range(MDL.LAY):
        if i == 0:
            # Input-Layer mit Angabe der Input-Shape
            model.add(tf.keras.layers.Conv2D(filters        = MDL.N, 
                                             kernel_size    = MDL.K,
                                             padding        = 'same',
                                             activation     = MDL.ACTF.lower() if MDL.ACTF == "ReLU" else MDL.ACTF,
                                             input_shape    = train_x.shape[1:]))
        else:
            model.add(tf.keras.layers.Conv2D(filters        = MDL.N, 
                                             kernel_size    = MDL.K,
                                             padding        = 'same',
                                             activation     = MDL.ACTF.lower() if MDL.ACTF == "ReLU" else MDL.ACTF))
    
    # Output-Layer: Convolution mit 1 Filter (oder Anzahl Kanäle von train_y)
    # und linearer Aktivierung, damit die Ausgabe dieselbe Form wie train_y hat.
    # Falls train_y z.B. (Batch, H, W, C), dann Filteranzahl = C.
    
    output_channels = train_y.shape[-1] if len(train_y.shape) == 4 else 1
    
    model.add(tf.keras.layers.Conv2D(filters            = output_channels,
                                     kernel_size        = 1,
                                     padding            = 'same',
                                     activation         = 'linear',
                                     kernel_initializer = tf.initializers.zeros))
    
    # Callback EarlyStopping
    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor                 = "val_loss",
        mode                    = "min",
        patience                = 2,
        restore_best_weights    = True)
    
    model.compile(
        optimizer   = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss        = tf.keras.losses.MeanSquaredError(),
        metrics     = [tf.keras.metrics.RootMeanSquaredError()]
    )
    
    # TRAINIEREN ##############################################################
    
    print("Modell wird trainiert.")
    # Use user-provided EP value, don't hardcode
    model.fit(
        x               = train_x,
        y               = train_y,
        epochs          = MDL.EP,
        verbose         = 1,
        callbacks       = [earlystopping],
        validation_data = (val_x, val_y)
    )
    
    print("Modell wurde trainiert.")
    
    return model


def train_lstm(train_x, train_y, val_x, val_y, MDL):
    """
    Funktion trainiert und validiert ein LSTM Neural Network
    
    Extracted from training_backend_test_2.py lines 321-388
    """
    
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available, skipping LSTM training")
        return None, float('inf'), 0
    
    # MODELLDEFINITION ########################################################
    
    # Modellinitialisierung
    model = tf.keras.Sequential()
    
    # LSTM-Layer hinzufügen - EXACTLY as in original lines 1507-1517
    for i in range(MDL.LAY):
        if i == 0:
            # First layer - always with return_sequences=True as in original
            model.add(tf.keras.layers.LSTM(MDL.N,
                                         activation = MDL.ACTF.lower() if MDL.ACTF == "ReLU" else MDL.ACTF,
                                         input_shape = (train_x.shape[1], train_x.shape[2]),
                                         return_sequences = True))
        else:
            # All other layers also with return_sequences=True as in original
            model.add(tf.keras.layers.LSTM(MDL.N,
                                         activation = MDL.ACTF.lower() if MDL.ACTF == "ReLU" else MDL.ACTF,
                                         return_sequences = True))
    
    # Output-Schicht - using TimeDistributed as in original line 1519
    model.add(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(train_y.shape[2], 
                           kernel_initializer = tf.initializers.zeros)))
    
    # Early Stopping
    earlystopping = tf.keras.callbacks.\
        EarlyStopping(monitor  = "val_loss", 
        mode                   = "min", 
        patience               = 2, 
        restore_best_weights   = True)

    # Konfiguration des Modells für das Training    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss = tf.keras.losses.MeanSquaredError(),
        metrics = [tf.keras.metrics.RootMeanSquaredError()])
    
    # TRAINIEREN ##############################################################
    
    print("Modell wird trainiert.")
    
    model.fit(
        x               = train_x,
        y               = train_y,
        epochs          = MDL.EP,
        verbose         = 1,
        callbacks       = [earlystopping],
        validation_data = (val_x, val_y)
        )
         
    print("Modell wurde trainiert.")
            
    return model


def train_ar_lstm(train_x, train_y, val_x, val_y, MDL):
    """
    Funktion trainiert und validiert ein Autoregressive LSTM
    
    Extracted from training_backend_test_2.py lines 389-457
    """
    
    # MODELLDEFINITION ########################################################
    
    # Modellinitialisierung
    model = tf.keras.Sequential()
    
    # LSTM-Layer hinzufügen - EXACTLY as in original lines 408-420
    for i in range(MDL.LAY):
        if i == 0:
            # First layer - always with return_sequences=True as in original
            model.add(tf.keras.layers.LSTM(MDL.N,
                                         activation = MDL.ACTF.lower() if MDL.ACTF == "ReLU" else MDL.ACTF,
                                         input_shape = (train_x.shape[1], train_x.shape[2]),
                                         return_sequences = True))
        else:
            # All other layers also with return_sequences=True as in original
            model.add(tf.keras.layers.LSTM(MDL.N,
                                         activation = MDL.ACTF.lower() if MDL.ACTF == "ReLU" else MDL.ACTF,
                                         return_sequences = True))
    
    # Output-Schicht - using TimeDistributed as in original lines 423-427
    model.add(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(train_y.shape[2], 
                           kernel_initializer = tf.initializers.zeros)))
    
    # Early Stopping
    earlystopping = tf.keras.callbacks.\
        EarlyStopping(monitor  = "val_loss", 
        mode                   = "min", 
        patience               = 2, 
        restore_best_weights   = True)

    # Konfiguration des Modells für das Training    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss = tf.keras.losses.MeanSquaredError(),
        metrics = [tf.keras.metrics.RootMeanSquaredError()])
    
    # TRAINIEREN ##############################################################
    
    print("Modell wird trainiert.")
    
    model.fit(
        x               = train_x,
        y               = train_y,
        epochs          = MDL.EP,
        verbose         = 1,
        callbacks       = [earlystopping],
        validation_data = (val_x, val_y)
        )
         
    print("Modell wurde trainiert.")
            
    return model


def train_svr_dir(train_x, train_y, MDL):
    """
    Funktion trainiert ein SVR-Modell anhand der 
    eingegebenen Trainingsdaten (train_x, train_y).
    
    train_x...Trainingsdaten (Eingabedaten) [n_samples, n_timesteps, n_features_in]
    train_y...Trainingsdaten (Ausgabedaten) [n_samples, n_timesteps, n_features_out]
    MDL.......Informationen zum Modell
    """
    
    # MODELLDEFINITION ########################################################
    
    n_samples, n_timesteps, n_features = train_x.shape
    X = train_x.reshape(n_samples * n_timesteps, n_features)
    
    y = []
    for i in range(n_features):
        y.append(train_y[:, :, i].reshape(-1))

    # TRAINIEREN ##############################################################

    print("Modell wird trainiert.")
    
    model = []
    for i in range(n_features):
        model.append(make_pipeline(StandardScaler(), 
                                   SVR(kernel  = MDL.KERNEL,
                                       C       = MDL.C, 
                                       epsilon = MDL.EPSILON)))
        model[-1].fit(X, y[i])

    print("Modell wurde trainiert.")  
    
    return model


def train_svr_mimo(train_x, train_y, MDL):
    """
    Funktion trainiert SVR MIMO Modell
    EXACTLY as in training_original.py lines 493-530
    """
    
    # MODELLDEFINITION ########################################################
    
    n_samples, n_timesteps, n_features_in = train_x.shape
    _, _, n_features_out = train_y.shape
    
    # Eingabedaten 2D umformen: (n_samples * n_timesteps, n_features_in)
    X = train_x.reshape(n_samples * n_timesteps, n_features_in)
    
    # TRAINIEREN ##############################################################
    
    print("Modell wird trainiert.")
    
    model = []
    for i in range(n_features_out):
        # Für jedes Ausgabefeature das passende Ziel erstellen
        y_i = train_y[:, :, i].reshape(-1)
        
        # Pipeline mit StandardScaler + SVR (EXACTLY as original)
        svr = make_pipeline(StandardScaler(),
                            SVR(kernel=MDL.KERNEL,
                                C=MDL.C,
                                epsilon=MDL.EPSILON))
        svr.fit(X, y_i)
        model.append(svr)
    
    print("Modell wurde trainiert.")
    return model


def train_linear_model(trn_x, trn_y):
    """
    Funktion trainiert Linear Regression Modell
    
    Extracted from training_backend_test_2.py lines 531-551
    """
    
    # MODELLDEFINITION ########################################################
    
    # Daten umformen
    n_samples, n_timesteps, n_features_in = trn_x.shape
    _, _, n_features_out = trn_y.shape
    
    X = trn_x.reshape(n_samples * n_timesteps, n_features_in)   # (390, 2)
    
    # TRAINIEREN ##############################################################
    
    print("Modell wird trainiert.")
    models = []
    for i in range(n_features_out):
        y_i = trn_y[:, :, i].reshape(-1)
        model = LinearRegression()
        model.fit(X, y_i)
        models.append(model)
    print("Modell wurde trainiert.")
    return models


class ModelTrainer:
    """
    Main model trainer class wrapper for extracted functions
    """
    
    def __init__(self, config=None):
        self.config = config or MDL
        self.trained_models = {}
        self.training_history = {}
    
    def train_all_models(self, datasets: Dict, session_data: Dict, training_split: dict = None) -> Dict:
        """
        Train all enabled models using the extracted real functions
        
        Args:
            datasets: Training datasets
            session_data: Session configuration
            training_split: Training split parameters from user (REQUIRED)
            
        Returns:
            Dict containing trained models and results
        """
        import os
        import pickle
        from datetime import datetime
        
        try:
            results = {}
            
            # Validate training_split is provided
            if not training_split:
                raise ValueError("Training split parameters are required but not provided")
            
            # Create models directory if it doesn't exist
            models_dir = os.path.join('uploads', 'trained_models')
            os.makedirs(models_dir, exist_ok=True)
            
            for dataset_name, dataset in datasets.items():
                X, y = dataset['X'], dataset['y']
                
                # Split data using user parameters
                X_train, X_test, y_train, y_test = self._split_data(X, y, training_split)
                
                dataset_results = {}
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Use the real training functions extracted from original
                if self.config.MODE == "Dense":
                    model = train_dense(X_train, y_train, X_test, y_test, self.config)
                    # Save Keras model to .h5 file
                    model_path = os.path.join(models_dir, f'dense_{dataset_name}_{timestamp}.h5')
                    model.save(model_path)
                    
                    # Calculate metrics
                    predictions = model.predict(X_test)
                    metrics = self._calculate_metrics(y_test, predictions)
                    
                    dataset_results['dense'] = {
                        'model_path': model_path,
                        'type': 'neural_network',
                        'metrics': metrics,
                        'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions
                    }
                
                elif self.config.MODE == "CNN":
                    model = train_cnn(X_train, y_train, X_test, y_test, self.config)
                    # Save Keras model to .h5 file
                    model_path = os.path.join(models_dir, f'cnn_{dataset_name}_{timestamp}.h5')
                    model.save(model_path)
                    
                    predictions = model.predict(X_test)
                    metrics = self._calculate_metrics(y_test, predictions)
                    
                    dataset_results['cnn'] = {
                        'model_path': model_path,
                        'type': 'neural_network',
                        'metrics': metrics,
                        'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions
                    }
                
                elif self.config.MODE == "LSTM":
                    model = train_lstm(X_train, y_train, X_test, y_test, self.config)
                    # Save Keras model to .h5 file
                    model_path = os.path.join(models_dir, f'lstm_{dataset_name}_{timestamp}.h5')
                    model.save(model_path)
                    
                    predictions = model.predict(X_test)
                    metrics = self._calculate_metrics(y_test, predictions)
                    
                    dataset_results['lstm'] = {
                        'model_path': model_path,
                        'type': 'neural_network',
                        'metrics': metrics,
                        'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions
                    }
                
                elif self.config.MODE == "AR LSTM":
                    model = train_ar_lstm(X_train, y_train, X_test, y_test, self.config)
                    # Save Keras model to .h5 file
                    model_path = os.path.join(models_dir, f'ar_lstm_{dataset_name}_{timestamp}.h5')
                    model.save(model_path)
                    
                    predictions = model.predict(X_test)
                    metrics = self._calculate_metrics(y_test, predictions)
                    
                    dataset_results['ar_lstm'] = {
                        'model_path': model_path,
                        'type': 'neural_network',
                        'metrics': metrics,
                        'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions
                    }
                
                elif self.config.MODE == "SVR_dir":
                    models = train_svr_dir(X_train, y_train, self.config)
                    # Save sklearn models using pickle
                    model_path = os.path.join(models_dir, f'svr_dir_{dataset_name}_{timestamp}.pkl')
                    with open(model_path, 'wb') as f:
                        pickle.dump(models, f)
                    
                    # Calculate predictions and metrics for SVR
                    predictions = self._predict_svr(models, X_test, y_test.shape)
                    metrics = self._calculate_metrics(y_test, predictions)
                    
                    dataset_results['svr_dir'] = {
                        'model_path': model_path,
                        'type': 'support_vector',
                        'metrics': metrics,
                        'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions
                    }
                
                elif self.config.MODE == "SVR_MIMO":
                    models = train_svr_mimo(X_train, y_train, self.config)
                    # Save sklearn models using pickle
                    model_path = os.path.join(models_dir, f'svr_mimo_{dataset_name}_{timestamp}.pkl')
                    with open(model_path, 'wb') as f:
                        pickle.dump(models, f)
                    
                    # Calculate predictions and metrics for SVR MIMO
                    predictions = self._predict_svr_mimo(models, X_test, y_test.shape)
                    metrics = self._calculate_metrics(y_test, predictions)
                    
                    dataset_results['svr_mimo'] = {
                        'model_path': model_path,
                        'type': 'support_vector',
                        'metrics': metrics,
                        'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions
                    }
                
                elif self.config.MODE == "LIN":
                    models = train_linear_model(X_train, y_train)
                    # Save sklearn models using pickle
                    model_path = os.path.join(models_dir, f'linear_{dataset_name}_{timestamp}.pkl')
                    with open(model_path, 'wb') as f:
                        pickle.dump(models, f)
                    
                    # Calculate predictions and metrics for Linear
                    predictions = self._predict_linear(models, X_test, y_test.shape)
                    metrics = self._calculate_metrics(y_test, predictions)
                    
                    dataset_results['linear'] = {
                        'model_path': model_path,
                        'type': 'linear_regression',
                        'metrics': metrics,
                        'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions
                    }
                
                results[dataset_name] = dataset_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    def _split_data(self, X: np.ndarray, y: np.ndarray, training_split: dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets using REQUIRED user-specified parameters
        
        Args:
            X: Input features
            y: Target values
            training_split: Dictionary with user training split parameters (REQUIRED)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            from sklearn.model_selection import train_test_split
            
            # Require user parameters - NO DEFAULTS
            if not training_split:
                raise ValueError("Training split parameters are required but not provided")
            
            # Validate required parameters
            required_params = ['testPercentage', 'random_dat']
            for param in required_params:
                if param not in training_split:
                    raise ValueError(f"Training split parameter '{param}' is required")
            
            # Calculate test_size from user percentages
            test_percentage = training_split['testPercentage']
            if test_percentage <= 0 or test_percentage >= 100:
                raise ValueError(f"testPercentage must be between 0 and 100, got {test_percentage}")
            
            test_size = test_percentage / 100.0
            
            # Use randomization setting from user
            random_state = None if training_split['random_dat'] else 42
            
            
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate evaluation metrics for predictions
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dict containing metrics
        """
        import math
        
        try:
            # Flatten arrays if needed
            if len(y_true.shape) > 2:
                y_true = y_true.reshape(y_true.shape[0], -1)
            if len(y_pred.shape) > 2:
                y_pred = y_pred.reshape(y_pred.shape[0], -1)
            
            # Calculate basic metrics
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            # Calculate MAPE with zero handling
            mask = y_true != 0
            if np.any(mask):
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = 0.0
            
            # Clean NaN and Infinity values
            def clean_metric(value):
                if math.isnan(value) or math.isinf(value):
                    return 0.0
                return float(value)
            
            return {
                'mae': clean_metric(mae),
                'mse': clean_metric(mse),
                'rmse': clean_metric(rmse),
                'mape': clean_metric(mape)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {
                'mae': 0.0,
                'mse': 0.0,
                'rmse': 0.0,
                'mape': 0.0
            }
    
    def _predict_svr(self, models: List, X_test: np.ndarray, y_shape: Tuple) -> np.ndarray:
        """
        Generate predictions for SVR Direct models
        
        Args:
            models: List of trained SVR models
            X_test: Test input data
            y_shape: Shape of target output
            
        Returns:
            Predictions array
        """
        try:
            n_samples, n_timesteps, n_features_in = X_test.shape
            _, n_timesteps_out, n_features_out = y_shape
            
            X = X_test.reshape(n_samples * n_timesteps, n_features_in)
            
            predictions = []
            for i, model in enumerate(models):
                y_pred = model.predict(X)
                predictions.append(y_pred.reshape(n_samples, n_timesteps_out))
            
            predictions = np.stack(predictions, axis=-1)
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting with SVR: {str(e)}")
            return np.zeros(y_shape)
    
    def _predict_svr_mimo(self, models: List, X_test: np.ndarray, y_shape: Tuple) -> np.ndarray:
        """
        Generate predictions for SVR MIMO models
        
        Args:
            models: List of trained SVR models
            X_test: Test input data
            y_shape: Shape of target output
            
        Returns:
            Predictions array
        """
        try:
            n_samples, n_timesteps, n_features_in = X_test.shape
            _, n_timesteps_out, n_features_out = y_shape
            
            X = X_test.reshape(n_samples, n_timesteps * n_features_in)
            
            predictions = []
            for model in models:
                y_pred = model.predict(X)
                predictions.append(y_pred)
            
            predictions = np.stack(predictions, axis=-1)
            predictions = predictions.reshape(n_samples, n_timesteps_out, n_features_out)
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting with SVR MIMO: {str(e)}")
            return np.zeros(y_shape)
    
    def _predict_linear(self, models: List, X_test: np.ndarray, y_shape: Tuple) -> np.ndarray:
        """
        Generate predictions for Linear models
        
        Args:
            models: List of trained Linear Regression models
            X_test: Test input data
            y_shape: Shape of target output
            
        Returns:
            Predictions array
        """
        try:
            n_samples, n_timesteps, n_features_in = X_test.shape
            _, n_timesteps_out, n_features_out = y_shape
            
            X = X_test.reshape(n_samples * n_timesteps, n_features_in)
            
            predictions = []
            for model in models:
                y_pred = model.predict(X)
                predictions.append(y_pred.reshape(n_samples, n_timesteps_out))
            
            predictions = np.stack(predictions, axis=-1)
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting with Linear model: {str(e)}")
            return np.zeros(y_shape)


# Factory function to create model trainer
def create_model_trainer(config=None) -> ModelTrainer:
    """
    Create and return a ModelTrainer instance
    
    Args:
        config: Model configuration (optional, will use MDL if None)
        
    Returns:
        ModelTrainer instance
    """
    return ModelTrainer(config)