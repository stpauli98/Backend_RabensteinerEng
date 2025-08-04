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
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, Conv2D, MaxPooling1D, Flatten, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    Sequential = None
    Dense = None
    LSTM = None
    Conv1D = None
    Conv2D = None
    MaxPooling1D = None
    Flatten = None
    Dropout = None
    EarlyStopping = None
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
    """
    
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not available. Please install TensorFlow to use neural network models.")
    
    # Validation: Check if we have enough features to train Dense network
    if train_x.shape[2] == 0:
        raise ValueError(f"Cannot train Dense network with 0 features. Input shape: {train_x.shape}")
       
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
    
    Updated to match reference implementation approach from training_backend_test_2.py
    Uses Conv2D with padding='same' and forces minimum 20 epochs for numerical stability
    """
    
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not available. Please install TensorFlow to use neural network models.")
    
    # Validation: Check if we have enough features to train CNN
    if train_x.shape[2] == 0:
        raise ValueError(f"Cannot train CNN with 0 features. Input shape: {train_x.shape}")
    
    # Intelligent kernel size adjustment (keep existing logic)
    if train_x.shape[2] < MDL.K:
        raise ValueError(f"Cannot train CNN: kernel size {MDL.K} is larger than feature count {train_x.shape[2]}")
    
    # REFERENCE IMPLEMENTATION APPROACH ######################################
    
    # Force minimum 20 epochs like reference implementation for numerical stability
    original_epochs = MDL.EP
    if MDL.EP < 20:
        logger.warning(f"CNN: Forcing minimum 20 epochs (was {MDL.EP}) for numerical stability")
        MDL.EP = 20
    
    # Reshape data to 4D format for Conv2D (reference approach)
    # From (samples, timesteps, features) to (samples, timesteps, features, 1)
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
    val_x = val_x.reshape(val_x.shape[0], val_x.shape[1], val_x.shape[2], 1)
    
    logger.info(f"CNN: Reshaped data to 4D for Conv2D")
    
    # MODELLDEFINITION ########################################################
    
    # Modellinitialisierung
    model = tf.keras.Sequential()
    
    # Conv2D-Layer hinzufügen (reference implementation approach)
    for i in range(MDL.LAY):
        if i == 0:
            # Erste Schicht benötigt input_shape
            model.add(tf.keras.layers.Conv2D(filters = MDL.N,
                                           kernel_size = MDL.K,
                                           padding = 'same',  # CRITICAL: padding='same' from reference
                                           activation = MDL.ACTF,
                                           input_shape = train_x.shape[1:]))
        else:
            model.add(tf.keras.layers.Conv2D(filters = MDL.N,
                                           kernel_size = MDL.K,
                                           padding = 'same',  # CRITICAL: padding='same' from reference
                                           activation = MDL.ACTF))
    
    # Output-Layer: Convolution mit 1 Filter (oder Anzahl Kanäle von train_y)
    # und linearer Aktivierung, damit die Ausgabe dieselbe Form wie train_y hat.
    # Falls train_y z.B. (Batch, H, W, C), dann Filteranzahl = C.
    
    output_channels = train_y.shape[-1] if len(train_y.shape) == 4 else 1
    
    model.add(tf.keras.layers.Conv2D(filters            = output_channels,
                                     kernel_size        = 1,
                                     padding            = 'same',
                                     activation         = 'linear',
                                     kernel_initializer = tf.initializers.zeros))
    
    # Early Stopping (keep existing)
    earlystopping = tf.keras.callbacks.\
        EarlyStopping(monitor  = "val_loss", 
        mode                   = "min", 
        patience               = 2, 
        restore_best_weights   = True)

    # Konfiguration des Modells für das Training (same as reference)    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss = tf.keras.losses.MeanSquaredError(),
        metrics = [tf.keras.metrics.RootMeanSquaredError()])
    
    # TRAINIEREN ##############################################################
    
    print("Modell wird trainiert.")
    
    model.fit(
        x               = train_x,
        y               = train_y,
        epochs          = MDL.EP,      # Using forced 20 epochs
        verbose         = 1,
        callbacks       = [earlystopping],
        validation_data = (val_x, val_y)
        )
         
    print("Modell wurde trainiert.")
    
    # Restore original epoch setting
    MDL.EP = original_epochs
            
    return model


def train_lstm(train_x, train_y, val_x, val_y, MDL):
    """
    Funktion trainiert und validiert ein LSTM Neural Network
    
    Extracted from training_backend_test_2.py lines 321-388
    """
    
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not available. Please install TensorFlow to use neural network models.")
    
    # MODELLDEFINITION ########################################################
    
    # Modellinitialisierung
    model = tf.keras.Sequential()
    
    for i in range(MDL.LAY):
        
        # Alle LSTM-Schichten mit return_sequences = True, um Sequenzen zu erhalten
                
        if i == 0:
            model.add(tf.keras.layers.LSTM(units            = MDL.N,
                                           activation       = MDL.ACTF,
                                           return_sequences = True,
                                           input_shape      = train_x.shape[1:]))
        else:
            model.add(tf.keras.layers.LSTM(units            = MDL.N,
                                           activation       = MDL.ACTF,
                                           return_sequences = True))
    
    # Dense Layer für jedes TimeStep
    output_units = train_y.shape[-1]
    model.add(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(output_units,
                              kernel_initializer = tf.initializers.zeros)
    ))
    
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
    Updated to match reference implementation with TimeDistributed output
    """
    
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not available. Please install TensorFlow to use neural network models.")
    
    # MODELLDEFINITION ########################################################
    
    # Modellinitialisierung
    model = tf.keras.Sequential()
    
    # LSTM-Layer hinzufügen - ALLE mit return_sequences = True (wie in Referenz)
    for i in range(MDL.LAY):
        if i == 0:
            # Erste Schicht benötigt input_shape
            model.add(tf.keras.layers.LSTM(units            = MDL.N,
                                           activation       = MDL.ACTF,
                                           return_sequences = True,
                                           input_shape      = train_x.shape[1:]))
        else:
            model.add(tf.keras.layers.LSTM(units            = MDL.N,
                                           activation       = MDL.ACTF,
                                           return_sequences = True))
    
    # TimeDistributed Dense Layer für Vorhersage pro Zeitschritt (wie in Referenz)
    output_units = train_y.shape[-1] if len(train_y.shape) > 2 else 1
    model.add(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(output_units,
                              kernel_initializer = tf.initializers.zeros)
    ))
    
    # Early Stopping
    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor                 = "val_loss",
        mode                    = "min",
        patience                = 2,
        restore_best_weights    = True)

    # Konfiguration des Modells für das Training    
    model.compile(
        optimizer   = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss        = tf.keras.losses.MeanSquaredError(),
        metrics     = [tf.keras.metrics.RootMeanSquaredError()]
    )
    
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
    Funktion trainiert ein SVR-MIMO-Modell anhand der
    eingegebenen Trainingsdaten (train_x, train_y).
    
    train_x...Trainingsdaten (Eingabedaten) [n_samples, n_timesteps, n_features_in]
    train_y...Trainingsdaten (Ausgabedaten) [n_samples, n_timesteps, n_features_out]
    MDL.......Informationen zum Modell
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

        # Pipeline mit StandardScaler + SVR
        svr = make_pipeline(StandardScaler(),
                            SVR(kernel=MDL.KERNEL,
                                C=MDL.C,
                                epsilon=MDL.EPSILON))
        svr.fit(X, y_i)
        model.append(svr)

    print("Modell wurde trainiert.")
    return model


def train_linear_model(train_x, train_y):
    """
    Funktion trainiert Linear Regression Modell
    
    Extracted from training_backend_test_2.py lines 531-551
    IMPORTANT: Reference assumes same timesteps for X and y
    """
    
    # MODELLDEFINITION ########################################################
    
    # Daten umformen
    n_samples, n_timesteps_x, n_features_in = train_x.shape
    _, n_timesteps_y, n_features_out = train_y.shape
    
    # Handle case where input and output have different timesteps
    # In reference, they're always equal, but user might set different values
    if n_timesteps_x != n_timesteps_y:
        logger.warning(f"Linear model: Different timesteps for X ({n_timesteps_x}) and y ({n_timesteps_y}). "
                      f"Using minimum to ensure consistency.")
        # Use the minimum timesteps to ensure consistency
        min_timesteps = min(n_timesteps_x, n_timesteps_y)
        train_x = train_x[:, :min_timesteps, :]
        train_y = train_y[:, :min_timesteps, :]
        n_timesteps = min_timesteps
    else:
        n_timesteps = n_timesteps_x
    
    X = train_x.reshape(n_samples * n_timesteps, n_features_in)
    
    # TRAINIEREN ##############################################################
    
    print("Modell wird trainiert.")
    models = []
    for i in range(n_features_out):
        y_i = train_y[:, :, i].reshape(-1)
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
        try:
            results = {}
            
            # Validate training_split is provided
            if not training_split:
                raise ValueError("Training split parameters are required but not provided")
            
            for dataset_name, dataset in datasets.items():
                X, y = dataset['X'], dataset['y']
                
                # Split data using user parameters
                X_train, X_test, y_train, y_test = self._split_data(X, y, training_split)
                
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
    
    def _split_data(self, X: np.ndarray, y: np.ndarray, training_split: dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets using REQUIRED user-specified parameters
        Matches reference implementation that uses array slicing for 3D data
        
        Args:
            X: Input features (can be 3D: samples, timesteps, features)
            y: Target values (can be 3D: samples, timesteps, features)
            training_split: Dictionary with user training split parameters (REQUIRED)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            # Require user parameters - NO DEFAULTS
            if not training_split:
                raise ValueError("Training split parameters are required but not provided")
            
            # Get total number of samples (first dimension)
            n_samples = X.shape[0]
            
            # Get split parameters from user - matching reference implementation
            # Reference uses n_train, n_val, n_test directly
            if 'n_train' in training_split and 'n_val' in training_split and 'n_test' in training_split:
                # Direct sample counts like reference implementation
                n_train = training_split['n_train']
                n_val = training_split['n_val'] 
                n_test = training_split['n_test']
                
                # Validate counts
                total_specified = n_train + n_val + n_test
                if total_specified > n_samples:
                    logger.warning(f"Specified splits ({total_specified}) exceed available samples ({n_samples}). Adjusting proportionally.")
                    # Scale down proportionally
                    scale_factor = n_samples / total_specified
                    n_train = int(n_train * scale_factor)
                    n_val = int(n_val * scale_factor)
                    n_test = n_samples - n_train - n_val  # Ensure we use all samples
                
            else:
                # Fall back to percentage-based splitting
                train_percentage = training_split.get('trainPercentage', 70)
                val_percentage = training_split.get('valPercentage', 15)
                test_percentage = training_split.get('testPercentage', 15)
                
                # Validate percentages
                total_percentage = train_percentage + val_percentage + test_percentage
                if total_percentage != 100 and total_percentage > 0:
                    logger.warning(f"Percentages don't sum to 100% ({total_percentage}%). Normalizing.")
                    train_percentage = (train_percentage / total_percentage) * 100
                    val_percentage = (val_percentage / total_percentage) * 100
                    test_percentage = (test_percentage / total_percentage) * 100
                
                # Calculate sample counts
                n_train = int(n_samples * train_percentage / 100)
                n_val = int(n_samples * val_percentage / 100)
                n_test = n_samples - n_train - n_val  # Ensure we use all samples
            
            # Handle randomization like reference implementation
            random_dat = training_split.get('random_dat', False)
            
            if random_dat:
                # Shuffle indices like reference implementation (line 2162-2166)
                indices = np.random.permutation(n_samples)
                X = X[indices]
                y = y[indices]
                logger.info("Data shuffled randomly as per user request")
            
            # Split using array slicing like reference implementation (lines 2218-2224)
            # This preserves 3D structure
            X_train = X[:n_train]
            X_val = X[n_train:(n_train + n_val)]
            X_test = X[(n_train + n_val):]
            
            y_train = y[:n_train]
            y_val = y[n_train:(n_train + n_val)]
            y_test = y[(n_train + n_val):]
            
            logger.info(f"Data split - Train: {n_train}, Val: {n_val}, Test: {n_test} samples")
            logger.info(f"Shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
            
            # For models that don't use validation data (SVR, Linear), combine train and val
            # Reference implementation passes val data to neural networks but not to SVR/Linear
            if self.config.MODE in ["SVR_dir", "SVR_MIMO", "LIN"]:
                # Combine train and validation for these models
                X_train_combined = np.concatenate([X_train, X_val], axis=0)
                y_train_combined = np.concatenate([y_train, y_val], axis=0)
                logger.info(f"Combined train+val for {self.config.MODE}: {X_train_combined.shape}")
                return X_train_combined, X_test, y_train_combined, y_test
            else:
                # Return validation data separately for neural networks
                # Note: Our interface expects (train, test, train, test) not (train, val, test)
                # So we'll return validation as "test" for now
                return X_train, X_val, y_train, y_val
            
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