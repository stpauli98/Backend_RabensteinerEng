"""
Configuration module for training system
Contains all configuration classes and constants extracted from training_backend_test_2.py
"""

import datetime
import calendar


class MTS:
    """
    Multivariate Time Series configuration class
    Extracted from training_backend_test_2.py around lines 619-692
    """
    def __init__(self):
        # TODO: Extract actual MTS class definition from training_backend_test_2.py
        # This is a placeholder structure based on typical ML configurations
        
        # Time series parameters
        self.time_steps_in = 24      # Input time steps
        self.time_steps_out = 1      # Output time steps
        self.time_step_size = 1      # Size of each time step
        self.offset = 0              # Time offset
        
        # Data processing parameters
        self.interpolation = True
        self.outlier_removal = True
        self.scaling = True
        
        # Model parameters
        self.epochs = 100
        self.batch_size = 32
        self.validation_split = 0.2
        
        # Feature engineering
        self.use_time_features = True
        self.use_holidays = True
        self.timezone = "UTC"
        
        # Categories and features
        self.jahr = True      # Year
        self.monat = True     # Month
        self.woche = True     # Week
        self.feiertag = True  # Holiday
        
        # Data paths (will be dynamic)
        self.input_files = []
        self.output_files = []


class MDL:
    """
    Model configuration class
    Extracted from training_backend_test_2.py around lines 2046-2141
    """
    def __init__(self):
        # TODO: Extract actual MDL class definition from training_backend_test_2.py
        # This is a placeholder structure
        
        # Model types to train
        self.models = {
            'dense': True,
            'cnn': True,
            'lstm': True,
            'ar_lstm': True,
            'svr_dir': True,
            'svr_mimo': True,
            'linear': True
        }
        
        # Dense neural network parameters
        self.dense_layers = [64, 32, 16]
        self.dense_activation = 'relu'
        self.dense_dropout = 0.2
        
        # CNN parameters
        self.cnn_filters = [32, 64]
        self.cnn_kernel_size = 3
        self.cnn_pool_size = 2
        
        # LSTM parameters
        self.lstm_units = [50, 50]
        self.lstm_return_sequences = True
        self.lstm_dropout = 0.2
        
        # SVR parameters
        self.svr_kernel = 'rbf'
        self.svr_C = 1.0
        self.svr_epsilon = 0.1
        
        # Training parameters
        self.early_stopping = True
        self.patience = 10
        self.min_delta = 0.001
        
        # Evaluation parameters
        self.test_size = 0.2
        self.random_state = 42


# Holiday configuration dictionary
# TODO: Extract actual HOL dictionary from training_backend_test_2.py
HOL = {
    'AT': {  # Austria holidays
        'neue_jahr': (1, 1),
        'heilige_drei_koenige': (1, 6),
        'staatsfeiertag': (5, 1),
        'maria_himmelfahrt': (8, 15),
        'nationalfeiertag': (10, 26),
        'allerheiligen': (11, 1),
        'maria_empfaengnis': (12, 8),
        'weihnachten': (12, 25),
        'stefanitag': (12, 26),
        # Easter-dependent holidays will be calculated dynamically
    }
}


# Global constants
SUPPORTED_TIMEZONES = ['UTC', 'Europe/Vienna', 'Europe/Berlin']
SUPPORTED_LANGUAGES = ['de', 'en']
DEFAULT_TIMEZONE = 'UTC'
DEFAULT_LANGUAGE = 'de'

# Data processing constants
MIN_DATA_POINTS = 100
MAX_DATA_POINTS = 1000000
OUTLIER_THRESHOLD = 3.0
INTERPOLATION_METHOD = 'linear'

# Model training constants
MAX_EPOCHS = 1000
MIN_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_VALIDATION_SPLIT = 0.2

# Evaluation metrics
EVALUATION_METRICS = [
    'mae',          # Mean Absolute Error
    'mape',         # Mean Absolute Percentage Error
    'mse',          # Mean Squared Error
    'rmse',         # Root Mean Squared Error
    'nrmse',        # Normalized Root Mean Squared Error
    'wape',         # Weighted Absolute Percentage Error
    'smape',        # Symmetric Mean Absolute Percentage Error
    'mase'          # Mean Absolute Scaled Error
]

# Visualization settings
PLOT_SETTINGS = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn',
    'color_palette': 'husl',
    'font_size': 12
}