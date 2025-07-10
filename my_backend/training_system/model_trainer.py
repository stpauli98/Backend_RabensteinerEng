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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from typing import Dict, List, Tuple, Optional, Any
import logging

from .config import MDL

logger = logging.getLogger(__name__)


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
                                        activation = MDL.ACTF)) # Aktivierungsfunktion
    
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
    Funktion trainiert und validiert ein Convolutional Neural Network
    
    Extracted from training_backend_test_2.py lines 239-320
    """
    
    # MODELLDEFINITION ########################################################
    
    # Modellinitialisierung
    model = tf.keras.Sequential()
    
    # Conv1D-Layer hinzufügen
    for i in range(MDL.LAY):
        if i == 0:
            # Erste Schicht benötigt input_shape
            model.add(tf.keras.layers.Conv1D(filters = MDL.N,
                                           kernel_size = MDL.K,
                                           activation = MDL.ACTF,
                                           input_shape = (train_x.shape[1], train_x.shape[2])))
        else:
            model.add(tf.keras.layers.Conv1D(filters = MDL.N,
                                           kernel_size = MDL.K,
                                           activation = MDL.ACTF))
    
    # Daten für Dense-Layer vorbereiten
    model.add(tf.keras.layers.Flatten())
    
    # Output-Schicht
    model.add(tf.keras.layers.Dense(train_y.shape[1]*train_y.shape[2], 
                                  kernel_initializer = tf.initializers.zeros))
    model.add(tf.keras.layers.Reshape([train_y.shape[1], train_y.shape[2]]))
    
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


def train_lstm(train_x, train_y, val_x, val_y, MDL):
    """
    Funktion trainiert und validiert ein LSTM Neural Network
    
    Extracted from training_backend_test_2.py lines 321-388
    """
    
    # MODELLDEFINITION ########################################################
    
    # Modellinitialisierung
    model = tf.keras.Sequential()
    
    # LSTM-Layer hinzufügen
    for i in range(MDL.LAY):
        if i == 0:
            # Erste Schicht benötigt input_shape
            if i == MDL.LAY - 1:  # Letzte Schicht
                model.add(tf.keras.layers.LSTM(MDL.N,
                                             activation = MDL.ACTF,
                                             input_shape = (train_x.shape[1], train_x.shape[2]),
                                             return_sequences = False))
            else:
                model.add(tf.keras.layers.LSTM(MDL.N,
                                             activation = MDL.ACTF,
                                             input_shape = (train_x.shape[1], train_x.shape[2]),
                                             return_sequences = True))
        else:
            if i == MDL.LAY - 1:  # Letzte Schicht
                model.add(tf.keras.layers.LSTM(MDL.N,
                                             activation = MDL.ACTF,
                                             return_sequences = False))
            else:
                model.add(tf.keras.layers.LSTM(MDL.N,
                                             activation = MDL.ACTF,
                                             return_sequences = True))
    
    # Output-Schicht
    model.add(tf.keras.layers.Dense(train_y.shape[1]*train_y.shape[2], 
                                  kernel_initializer = tf.initializers.zeros))
    model.add(tf.keras.layers.Reshape([train_y.shape[1], train_y.shape[2]]))
    
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
    
    # LSTM-Layer hinzufügen (ähnlich wie LSTM aber mit anderem Output)
    for i in range(MDL.LAY):
        if i == 0:
            # Erste Schicht benötigt input_shape
            if i == MDL.LAY - 1:  # Letzte Schicht
                model.add(tf.keras.layers.LSTM(MDL.N,
                                             activation = MDL.ACTF,
                                             input_shape = (train_x.shape[1], train_x.shape[2]),
                                             return_sequences = False))
            else:
                model.add(tf.keras.layers.LSTM(MDL.N,
                                             activation = MDL.ACTF,
                                             input_shape = (train_x.shape[1], train_x.shape[2]),
                                             return_sequences = True))
        else:
            if i == MDL.LAY - 1:  # Letzte Schicht
                model.add(tf.keras.layers.LSTM(MDL.N,
                                             activation = MDL.ACTF,
                                             return_sequences = False))
            else:
                model.add(tf.keras.layers.LSTM(MDL.N,
                                             activation = MDL.ACTF,
                                             return_sequences = True))
    
    # Output-Schicht für Autoregressive
    model.add(tf.keras.layers.Dense(train_y.shape[1]*train_y.shape[2], 
                                  kernel_initializer = tf.initializers.zeros))
    model.add(tf.keras.layers.Reshape([train_y.shape[1], train_y.shape[2]]))
    
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
    Funktion trainiert SVR Direktmodell
    
    Extracted from training_backend_test_2.py lines 458-492
    """
    
    # MODELLDEFINITION ########################################################
    
    # Daten umformen
    n_samples, n_timesteps, n_features_in = train_x.shape
    _, _, n_features_out = train_y.shape
    
    X = train_x.reshape(n_samples * n_timesteps, n_features_in)
    
    # TRAINIEREN ##############################################################
    
    print("Modell wird trainiert.")
    models = []
    for i in range(n_features_out):
        y_i = train_y[:, :, i].reshape(-1)
        
        # SVR Modell erstellen
        model = SVR(kernel = MDL.KERNEL, 
                   C = MDL.C, 
                   epsilon = MDL.EPSILON)
        model.fit(X, y_i)
        models.append(model)
    print("Modell wurde trainiert.")
    return models


def train_svr_mimo(train_x, train_y, MDL):
    """
    Funktion trainiert SVR MIMO Modell
    
    Extracted from training_backend_test_2.py lines 493-530
    """
    
    # MODELLDEFINITION ########################################################
    
    # Daten umformen für MIMO
    n_samples, n_timesteps, n_features_in = train_x.shape
    _, n_timesteps_out, n_features_out = train_y.shape
    
    X = train_x.reshape(n_samples, n_timesteps * n_features_in)
    Y = train_y.reshape(n_samples, n_timesteps_out * n_features_out)
    
    # TRAINIEREN ##############################################################
    
    print("Modell wird trainiert.")
    models = []
    for i in range(Y.shape[1]):  # Für jeden Output
        y_i = Y[:, i]
        
        # SVR Modell erstellen
        model = SVR(kernel = MDL.KERNEL, 
                   C = MDL.C, 
                   epsilon = MDL.EPSILON)
        model.fit(X, y_i)
        models.append(model)
    print("Modell wurde trainiert.")
    return models


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
    
    def train_all_models(self, datasets: Dict, session_data: Dict) -> Dict:
        """
        Train all enabled models using the extracted real functions
        
        Args:
            datasets: Training datasets
            session_data: Session configuration
            
        Returns:
            Dict containing trained models and results
        """
        try:
            results = {}
            
            for dataset_name, dataset in datasets.items():
                X, y = dataset['X'], dataset['y']
                
                # Split data
                X_train, X_test, y_train, y_test = self._split_data(X, y)
                
                dataset_results = {}
                
                # Use the real training functions extracted from original
                if self.config.MODE == "Dense":
                    dataset_results['dense'] = train_dense(X_train, y_train, X_test, y_test, self.config)
                
                elif self.config.MODE == "CNN":
                    dataset_results['cnn'] = train_cnn(X_train, y_train, X_test, y_test, self.config)
                
                elif self.config.MODE == "LSTM":
                    dataset_results['lstm'] = train_lstm(X_train, y_train, X_test, y_test, self.config)
                
                elif self.config.MODE == "AR LSTM":
                    dataset_results['ar_lstm'] = train_ar_lstm(X_train, y_train, X_test, y_test, self.config)
                
                elif self.config.MODE == "SVR_dir":
                    dataset_results['svr_dir'] = train_svr_dir(X_train, y_train, self.config)
                
                elif self.config.MODE == "SVR_MIMO":
                    dataset_results['svr_mimo'] = train_svr_mimo(X_train, y_train, self.config)
                
                elif self.config.MODE == "LIN":
                    dataset_results['linear'] = train_linear_model(X_train, y_train)
                
                results[dataset_name] = dataset_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            from sklearn.model_selection import train_test_split
            
            return train_test_split(X, y, test_size=0.2, random_state=42)
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise


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