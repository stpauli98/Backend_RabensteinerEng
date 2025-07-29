"""
Parameter conversion utilities for training system
Converts frontend parameters to reference MDL format (Phase 3.1)
Ensures exact synchronization with training_backend_test_2.py structure
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReferenceMDLStructure:
    """
    Reference MDL structure exactly matching training_backend_test_2.py
    This represents the exact nested class structure expected by the reference implementation
    """
    
    @dataclass
    class DENSE:
        """Dense Neural Network configuration - EXACT structure from reference"""
        L1_N: int = 64      # Layer 1 neurons
        L1_A: str = "relu"  # Layer 1 activation 
        L2_N: int = 32      # Layer 2 neurons
        L2_A: str = "relu"  # Layer 2 activation
        L3_N: int = 16      # Layer 3 neurons
        L3_A: str = "relu"  # Layer 3 activation
        EP: int = 100       # Epochs
        BS: int = 32        # Batch size
        VAL_S: float = 0.2  # Validation split
        OPT: str = "adam"   # Optimizer
        LOSS: str = "mse"   # Loss function
        LR: float = 0.001   # Learning rate
        METRICS: List[str] = None  # Metrics to track
        
        def __post_init__(self):
            if self.METRICS is None:
                self.METRICS = ["mae"]
    
    @dataclass  
    class CNN:
        """CNN configuration - EXACT structure from reference"""
        L1_F: int = 32      # Layer 1 filters
        L1_K: int = 3       # Layer 1 kernel size
        L1_A: str = "relu"  # Layer 1 activation
        L1_P: int = 2       # Layer 1 pool size
        L2_F: int = 64      # Layer 2 filters
        L2_K: int = 3       # Layer 2 kernel size
        L2_A: str = "relu"  # Layer 2 activation
        L2_P: int = 2       # Layer 2 pool size
        L3_N: int = 50      # Dense layer neurons
        L3_A: str = "relu"  # Dense layer activation
        EP: int = 100       # Epochs
        BS: int = 32        # Batch size
        VAL_S: float = 0.2  # Validation split
        OPT: str = "adam"   # Optimizer
        LOSS: str = "mse"   # Loss function
        LR: float = 0.001   # Learning rate
        METRICS: List[str] = None  # Metrics to track
        
        def __post_init__(self):
            if self.METRICS is None:
                self.METRICS = ["mae"]
    
    @dataclass
    class LSTM:
        """LSTM configuration - EXACT structure from reference"""
        L1_N: int = 50      # Layer 1 LSTM units
        L1_D: float = 0.2   # Layer 1 dropout
        L1_RS: bool = True  # Layer 1 return sequences
        L2_N: int = 50      # Layer 2 LSTM units  
        L2_D: float = 0.2   # Layer 2 dropout
        L2_RS: bool = False # Layer 2 return sequences
        L3_N: int = 25      # Dense layer neurons
        L3_A: str = "relu"  # Dense layer activation
        EP: int = 100       # Epochs
        BS: int = 32        # Batch size
        VAL_S: float = 0.2  # Validation split
        OPT: str = "adam"   # Optimizer
        LOSS: str = "mse"   # Loss function
        LR: float = 0.001   # Learning rate
        METRICS: List[str] = None  # Metrics to track
        
        def __post_init__(self):
            if self.METRICS is None:
                self.METRICS = ["mae"]
    
    @dataclass
    class AR_LSTM:
        """Autoregressive LSTM configuration - EXACT structure from reference"""
        L1_N: int = 50      # LSTM units
        L1_D: float = 0.2   # LSTM dropout
        L2_N: int = 25      # Dense layer neurons
        L2_A: str = "relu"  # Dense layer activation
        EP: int = 100       # Epochs
        BS: int = 32        # Batch size
        VAL_S: float = 0.2  # Validation split
        OPT: str = "adam"   # Optimizer
        LOSS: str = "mse"   # Loss function
        LR: float = 0.001   # Learning rate
        METRICS: List[str] = None  # Metrics to track
        
        def __post_init__(self):
            if self.METRICS is None:
                self.METRICS = ["mae"]
    
    @dataclass
    class SVR:
        """SVR configuration - EXACT structure from reference"""
        KERNEL: str = "rbf"     # Kernel type
        C: float = 1.0          # Regularization parameter
        GAMMA: str = "scale"    # Kernel coefficient
        EPSILON: float = 0.1    # Epsilon parameter
        DEGREE: int = 3         # Polynomial degree (for poly kernel)
        COEF0: float = 0.0      # Independent term (for poly/sigmoid)
        SHRINKING: bool = True  # Whether to use shrinking heuristic
        TOL: float = 0.001      # Tolerance for stopping criterion
        CACHE_SIZE: int = 200   # Cache size in MB
        MAX_ITER: int = -1      # Maximum iterations (-1 for no limit)
    
    @dataclass
    class LINEAR:
        """Linear Regression configuration - EXACT structure from reference"""
        FIT_INTERCEPT: bool = True   # Whether to fit intercept
        NORMALIZE: bool = False      # Whether to normalize features (deprecated)
        COPY_X: bool = True          # Whether to copy X
        N_JOBS: Optional[int] = None # Number of jobs for computation
        POSITIVE: bool = False       # Force coefficients to be positive


class FrontendParameterConverter:
    """
    Converts frontend flat parameters to reference MDL nested structure
    Handles conversion from TrainingApiService.ts format to training_backend_test_2.py format
    """
    
    def __init__(self):
        """Initialize parameter converter with validation and mapping rules"""
        self.logger = logging.getLogger(__name__ + ".FrontendParameterConverter")
        
        # Frontend activation function mapping to reference format
        self.activation_mapping = {
            'ReLU': 'relu',
            'Sigmoid': 'sigmoid', 
            'Tanh': 'tanh',
            'Linear': 'linear',
            'ELU': 'elu',
            'LeakyReLU': 'leaky_relu',
            'Swish': 'swish'
        }
        
        # SVR kernel mapping
        self.kernel_mapping = {
            'RBF': 'rbf',
            'Linear': 'linear',
            'Polynomial': 'poly',
            'Sigmoid': 'sigmoid'
        }
        
        # Optimizer mapping
        self.optimizer_mapping = {
            'Adam': 'adam',
            'SGD': 'sgd',
            'RMSprop': 'rmsprop',
            'Adagrad': 'adagrad',
            'Adadelta': 'adadelta'
        }
    
    def convert_frontend_to_mdl_format(self, frontend_params: Dict) -> Dict:
        """
        Convert frontend parameters to reference MDL format
        
        Args:
            frontend_params: Frontend parameters from TrainingApiService.ts
            Expected structure:
            {
                "dense_layers": [64, 32, 16],
                "dense_activation": "relu", 
                "dense_epochs": 100,
                "dense_batch_size": 32,
                "dense_optimizer": "adam",
                "dense_learning_rate": 0.001,
                "cnn_filters": [32, 64],
                "cnn_kernel_size": [3, 3],
                "cnn_activation": "relu",
                "cnn_epochs": 100,
                "lstm_units": [50, 50],
                "lstm_dropout": 0.2,
                "lstm_epochs": 100,
                "svr_kernel": "rbf",
                "svr_C": 1.0,
                "svr_epsilon": 0.1,
                "linear_fit_intercept": true
            }
            
        Returns:
            Dict with MDL structure exactly matching reference implementation
        """
        try:
            self.logger.info(f"Converting frontend parameters: {list(frontend_params.keys())}")
            
            mdl_config = {}
            
            # Convert Dense parameters
            if self._has_dense_params(frontend_params):
                mdl_config['DENSE'] = self._convert_dense_params(frontend_params)
                self.logger.info("Converted Dense Neural Network parameters")
            
            # Convert CNN parameters  
            if self._has_cnn_params(frontend_params):
                mdl_config['CNN'] = self._convert_cnn_params(frontend_params)
                self.logger.info("Converted CNN parameters")
            
            # Convert LSTM parameters
            if self._has_lstm_params(frontend_params):
                mdl_config['LSTM'] = self._convert_lstm_params(frontend_params)
                self.logger.info("Converted LSTM parameters")
            
            # Convert AR-LSTM parameters (if different from LSTM)
            if self._has_ar_lstm_params(frontend_params):
                mdl_config['AR_LSTM'] = self._convert_ar_lstm_params(frontend_params)
                self.logger.info("Converted AR-LSTM parameters")
            
            # Convert SVR parameters
            if self._has_svr_params(frontend_params):
                mdl_config['SVR'] = self._convert_svr_params(frontend_params)
                self.logger.info("Converted SVR parameters")
            
            # Convert Linear parameters
            if self._has_linear_params(frontend_params):
                mdl_config['LINEAR'] = self._convert_linear_params(frontend_params)
                self.logger.info("Converted Linear Regression parameters")
            
            if not mdl_config:
                self.logger.warning("No model parameters detected in frontend data")
                return {}
            
            self.logger.info(f"Successfully converted {len(mdl_config)} model configurations to MDL format")
            return mdl_config
            
        except Exception as e:
            self.logger.error(f"Error converting frontend parameters to MDL format: {str(e)}")
            raise
    
    def _has_dense_params(self, params: Dict) -> bool:
        """Check if Dense parameters are present"""
        dense_keys = ['dense_layers', 'dense_neurons', 'dense_activation', 'dense_epochs']
        return any(key in params for key in dense_keys)
    
    def _has_cnn_params(self, params: Dict) -> bool:
        """Check if CNN parameters are present"""
        cnn_keys = ['cnn_filters', 'cnn_kernel_size', 'cnn_activation', 'cnn_epochs']
        return any(key in params for key in cnn_keys)
    
    def _has_lstm_params(self, params: Dict) -> bool:
        """Check if LSTM parameters are present"""
        lstm_keys = ['lstm_units', 'lstm_dropout', 'lstm_epochs', 'lstm_layers']
        return any(key in params for key in lstm_keys)
    
    def _has_ar_lstm_params(self, params: Dict) -> bool:
        """Check if AR-LSTM parameters are present"""
        ar_lstm_keys = ['ar_lstm_units', 'ar_lstm_dropout', 'ar_lstm_epochs']
        return any(key in params for key in ar_lstm_keys)
    
    def _has_svr_params(self, params: Dict) -> bool:
        """Check if SVR parameters are present"""
        svr_keys = ['svr_kernel', 'svr_C', 'svr_epsilon', 'svr_gamma']
        return any(key in params for key in svr_keys)
    
    def _has_linear_params(self, params: Dict) -> bool:
        """Check if Linear parameters are present"""
        linear_keys = ['linear_fit_intercept', 'linear_normalize', 'linear_positive']
        return any(key in params for key in linear_keys)
    
    def _convert_dense_params(self, params: Dict) -> Dict:
        """Convert Dense Neural Network parameters to reference format"""
        try:
            dense_config = {}
            
            # Extract layer configuration
            layers = params.get('dense_layers', [64, 32, 16])
            if isinstance(layers, list) and len(layers) >= 1:
                # Map to reference structure (L1_N, L2_N, L3_N)
                dense_config['L1_N'] = int(layers[0]) if len(layers) > 0 else 64
                dense_config['L2_N'] = int(layers[1]) if len(layers) > 1 else 32
                dense_config['L3_N'] = int(layers[2]) if len(layers) > 2 else 16
            else:
                # Single neuron count - distribute across layers
                neurons = int(params.get('dense_neurons', 64))
                dense_config['L1_N'] = neurons
                dense_config['L2_N'] = max(16, neurons // 2)
                dense_config['L3_N'] = max(8, neurons // 4)
            
            # Activation functions (same for all layers in reference)
            activation = params.get('dense_activation', 'relu')
            mapped_activation = self.activation_mapping.get(activation, activation.lower())
            dense_config['L1_A'] = mapped_activation
            dense_config['L2_A'] = mapped_activation  
            dense_config['L3_A'] = mapped_activation
            
            # Training parameters
            dense_config['EP'] = int(params.get('dense_epochs', 100))
            dense_config['BS'] = int(params.get('dense_batch_size', 32))
            dense_config['VAL_S'] = float(params.get('dense_validation_split', 0.2))
            
            # Optimizer and learning parameters
            optimizer = params.get('dense_optimizer', 'adam')
            dense_config['OPT'] = self.optimizer_mapping.get(optimizer, optimizer.lower())
            dense_config['LOSS'] = params.get('dense_loss', 'mse').lower()
            dense_config['LR'] = float(params.get('dense_learning_rate', 0.001))
            
            # Metrics
            metrics = params.get('dense_metrics', ['mae'])
            dense_config['METRICS'] = metrics if isinstance(metrics, list) else ['mae']
            
            self.logger.info(f"Dense config: L1_N={dense_config['L1_N']}, L2_N={dense_config['L2_N']}, L3_N={dense_config['L3_N']}, EP={dense_config['EP']}")
            return dense_config
            
        except Exception as e:
            self.logger.error(f"Error converting Dense parameters: {str(e)}")
            raise
    
    def _convert_cnn_params(self, params: Dict) -> Dict:
        """Convert CNN parameters to reference format"""
        try:
            cnn_config = {}
            
            # Extract filter configuration
            filters = params.get('cnn_filters', [32, 64])
            if isinstance(filters, list) and len(filters) >= 1:
                cnn_config['L1_F'] = int(filters[0]) if len(filters) > 0 else 32
                cnn_config['L2_F'] = int(filters[1]) if len(filters) > 1 else 64
            else:
                # Single filter count
                filter_count = int(params.get('cnn_filter_count', 32))
                cnn_config['L1_F'] = filter_count
                cnn_config['L2_F'] = filter_count * 2
            
            # Kernel sizes
            kernel_size = params.get('cnn_kernel_size', [3, 3])
            if isinstance(kernel_size, list) and len(kernel_size) >= 2:
                cnn_config['L1_K'] = int(kernel_size[0])
                cnn_config['L2_K'] = int(kernel_size[1])
            else:
                # Single kernel size for both layers
                k_size = int(params.get('cnn_kernel', 3))
                cnn_config['L1_K'] = k_size
                cnn_config['L2_K'] = k_size
            
            # Pool sizes
            pool_size = params.get('cnn_pool_size', [2, 2])
            if isinstance(pool_size, list) and len(pool_size) >= 2:
                cnn_config['L1_P'] = int(pool_size[0])
                cnn_config['L2_P'] = int(pool_size[1])
            else:
                # Single pool size for both layers
                p_size = int(params.get('cnn_pool', 2))
                cnn_config['L1_P'] = p_size
                cnn_config['L2_P'] = p_size
            
            # Activation functions
            activation = params.get('cnn_activation', 'relu')
            mapped_activation = self.activation_mapping.get(activation, activation.lower())
            cnn_config['L1_A'] = mapped_activation
            cnn_config['L2_A'] = mapped_activation
            
            # Dense layer configuration (after CNN layers)
            cnn_config['L3_N'] = int(params.get('cnn_dense_neurons', 50))
            cnn_config['L3_A'] = mapped_activation
            
            # Training parameters
            cnn_config['EP'] = int(params.get('cnn_epochs', 100))
            cnn_config['BS'] = int(params.get('cnn_batch_size', 32))
            cnn_config['VAL_S'] = float(params.get('cnn_validation_split', 0.2))
            
            # Optimizer and learning parameters
            optimizer = params.get('cnn_optimizer', 'adam')
            cnn_config['OPT'] = self.optimizer_mapping.get(optimizer, optimizer.lower())
            cnn_config['LOSS'] = params.get('cnn_loss', 'mse').lower()
            cnn_config['LR'] = float(params.get('cnn_learning_rate', 0.001))
            
            # Metrics
            metrics = params.get('cnn_metrics', ['mae'])
            cnn_config['METRICS'] = metrics if isinstance(metrics, list) else ['mae']
            
            self.logger.info(f"CNN config: L1_F={cnn_config['L1_F']}, L2_F={cnn_config['L2_F']}, L1_K={cnn_config['L1_K']}, L2_K={cnn_config['L2_K']}")
            return cnn_config
            
        except Exception as e:
            self.logger.error(f"Error converting CNN parameters: {str(e)}")
            raise
    
    def _convert_lstm_params(self, params: Dict) -> Dict:
        """Convert LSTM parameters to reference format"""
        try:
            lstm_config = {}
            
            # Extract LSTM units configuration
            units = params.get('lstm_units', [50, 50])
            if isinstance(units, list) and len(units) >= 1:
                lstm_config['L1_N'] = int(units[0]) if len(units) > 0 else 50
                lstm_config['L2_N'] = int(units[1]) if len(units) > 1 else 50
            else:
                # Single unit count for both layers
                unit_count = int(params.get('lstm_unit_count', 50))
                lstm_config['L1_N'] = unit_count
                lstm_config['L2_N'] = unit_count
            
            # Dropout configuration
            dropout = params.get('lstm_dropout', 0.2)
            if isinstance(dropout, list) and len(dropout) >= 2:
                lstm_config['L1_D'] = float(dropout[0])
                lstm_config['L2_D'] = float(dropout[1])
            else:
                # Same dropout for both layers
                dropout_rate = float(dropout)
                lstm_config['L1_D'] = dropout_rate
                lstm_config['L2_D'] = dropout_rate
            
            # Return sequences configuration
            lstm_config['L1_RS'] = bool(params.get('lstm_return_sequences_l1', True))
            lstm_config['L2_RS'] = bool(params.get('lstm_return_sequences_l2', False))
            
            # Dense layer configuration (after LSTM layers)
            lstm_config['L3_N'] = int(params.get('lstm_dense_neurons', 25))
            
            # Dense layer activation
            activation = params.get('lstm_dense_activation', 'relu')
            mapped_activation = self.activation_mapping.get(activation, activation.lower())
            lstm_config['L3_A'] = mapped_activation
            
            # Training parameters
            lstm_config['EP'] = int(params.get('lstm_epochs', 100))
            lstm_config['BS'] = int(params.get('lstm_batch_size', 32))
            lstm_config['VAL_S'] = float(params.get('lstm_validation_split', 0.2))
            
            # Optimizer and learning parameters
            optimizer = params.get('lstm_optimizer', 'adam')
            lstm_config['OPT'] = self.optimizer_mapping.get(optimizer, optimizer.lower())
            lstm_config['LOSS'] = params.get('lstm_loss', 'mse').lower()
            lstm_config['LR'] = float(params.get('lstm_learning_rate', 0.001))
            
            # Metrics
            metrics = params.get('lstm_metrics', ['mae'])
            lstm_config['METRICS'] = metrics if isinstance(metrics, list) else ['mae']
            
            self.logger.info(f"LSTM config: L1_N={lstm_config['L1_N']}, L2_N={lstm_config['L2_N']}, L1_D={lstm_config['L1_D']}, L2_D={lstm_config['L2_D']}")
            return lstm_config
            
        except Exception as e:
            self.logger.error(f"Error converting LSTM parameters: {str(e)}")
            raise
    
    def _convert_ar_lstm_params(self, params: Dict) -> Dict:
        """Convert AR-LSTM parameters to reference format"""
        try:
            ar_lstm_config = {}
            
            # LSTM units (single layer for AR-LSTM)
            ar_lstm_config['L1_N'] = int(params.get('ar_lstm_units', 50))
            ar_lstm_config['L1_D'] = float(params.get('ar_lstm_dropout', 0.2))
            
            # Dense layer configuration
            ar_lstm_config['L2_N'] = int(params.get('ar_lstm_dense_neurons', 25))
            
            # Dense layer activation
            activation = params.get('ar_lstm_dense_activation', 'relu')
            mapped_activation = self.activation_mapping.get(activation, activation.lower())
            ar_lstm_config['L2_A'] = mapped_activation
            
            # Training parameters
            ar_lstm_config['EP'] = int(params.get('ar_lstm_epochs', 100))
            ar_lstm_config['BS'] = int(params.get('ar_lstm_batch_size', 32))
            ar_lstm_config['VAL_S'] = float(params.get('ar_lstm_validation_split', 0.2))
            
            # Optimizer and learning parameters
            optimizer = params.get('ar_lstm_optimizer', 'adam')
            ar_lstm_config['OPT'] = self.optimizer_mapping.get(optimizer, optimizer.lower())
            ar_lstm_config['LOSS'] = params.get('ar_lstm_loss', 'mse').lower()
            ar_lstm_config['LR'] = float(params.get('ar_lstm_learning_rate', 0.001))
            
            # Metrics
            metrics = params.get('ar_lstm_metrics', ['mae'])
            ar_lstm_config['METRICS'] = metrics if isinstance(metrics, list) else ['mae']
            
            self.logger.info(f"AR-LSTM config: L1_N={ar_lstm_config['L1_N']}, L1_D={ar_lstm_config['L1_D']}, L2_N={ar_lstm_config['L2_N']}")
            return ar_lstm_config
            
        except Exception as e:
            self.logger.error(f"Error converting AR-LSTM parameters: {str(e)}")
            raise
    
    def _convert_svr_params(self, params: Dict) -> Dict:
        """Convert SVR parameters to reference format"""
        try:
            svr_config = {}
            
            # Kernel configuration
            kernel = params.get('svr_kernel', 'rbf')
            svr_config['KERNEL'] = self.kernel_mapping.get(kernel, kernel.lower())
            
            # Regularization parameter
            svr_config['C'] = float(params.get('svr_C', 1.0))
            
            # Gamma parameter
            gamma = params.get('svr_gamma', 'scale')
            if isinstance(gamma, str):
                svr_config['GAMMA'] = gamma  # 'scale' or 'auto'
            else:
                svr_config['GAMMA'] = float(gamma)  # Numeric value
            
            # Epsilon parameter
            svr_config['EPSILON'] = float(params.get('svr_epsilon', 0.1))
            
            # Polynomial degree (for poly kernel)
            svr_config['DEGREE'] = int(params.get('svr_degree', 3))
            
            # Independent term (for poly/sigmoid kernels)
            svr_config['COEF0'] = float(params.get('svr_coef0', 0.0))
            
            # Additional parameters
            svr_config['SHRINKING'] = bool(params.get('svr_shrinking', True))
            svr_config['TOL'] = float(params.get('svr_tolerance', 0.001))
            svr_config['CACHE_SIZE'] = int(params.get('svr_cache_size', 200))
            svr_config['MAX_ITER'] = int(params.get('svr_max_iter', -1))
            
            self.logger.info(f"SVR config: KERNEL={svr_config['KERNEL']}, C={svr_config['C']}, EPSILON={svr_config['EPSILON']}")
            return svr_config
            
        except Exception as e:
            self.logger.error(f"Error converting SVR parameters: {str(e)}")
            raise
    
    def _convert_linear_params(self, params: Dict) -> Dict:
        """Convert Linear Regression parameters to reference format"""
        try:
            linear_config = {}
            
            # Intercept fitting
            linear_config['FIT_INTERCEPT'] = bool(params.get('linear_fit_intercept', True))
            
            # Normalization (deprecated in newer sklearn versions)
            linear_config['NORMALIZE'] = bool(params.get('linear_normalize', False))
            
            # Copy X parameter
            linear_config['COPY_X'] = bool(params.get('linear_copy_x', True))
            
            # Number of jobs for parallel computation
            n_jobs = params.get('linear_n_jobs', None)
            linear_config['N_JOBS'] = int(n_jobs) if n_jobs is not None else None
            
            # Positive coefficients constraint
            linear_config['POSITIVE'] = bool(params.get('linear_positive', False))
            
            self.logger.info(f"Linear config: FIT_INTERCEPT={linear_config['FIT_INTERCEPT']}, POSITIVE={linear_config['POSITIVE']}")
            return linear_config
            
        except Exception as e:
            self.logger.error(f"Error converting Linear parameters: {str(e)}")
            raise


class ReferenceMDLValidator:
    """
    Validates converted MDL parameters against reference implementation requirements
    Ensures all parameters are within acceptable ranges and match reference constraints
    """
    
    def __init__(self):
        """Initialize validator with reference constraints"""
        self.logger = logging.getLogger(__name__ + ".ReferenceMDLValidator")
        
        # Reference validation rules
        self.validation_rules = {
            'DENSE': {
                'L1_N': {'min': 1, 'max': 1024, 'type': int},
                'L2_N': {'min': 1, 'max': 1024, 'type': int},
                'L3_N': {'min': 1, 'max': 1024, 'type': int},
                'L1_A': {'values': ['relu', 'sigmoid', 'tanh', 'linear', 'elu'], 'type': str},
                'L2_A': {'values': ['relu', 'sigmoid', 'tanh', 'linear', 'elu'], 'type': str},
                'L3_A': {'values': ['relu', 'sigmoid', 'tanh', 'linear', 'elu'], 'type': str},
                'EP': {'min': 1, 'max': 10000, 'type': int},
                'BS': {'min': 1, 'max': 1024, 'type': int},
                'VAL_S': {'min': 0.0, 'max': 0.9, 'type': float},
                'LR': {'min': 0.0001, 'max': 1.0, 'type': float},
                'OPT': {'values': ['adam', 'sgd', 'rmsprop', 'adagrad'], 'type': str},
                'LOSS': {'values': ['mse', 'mae', 'huber'], 'type': str}
            },
            'CNN': {
                'L1_F': {'min': 1, 'max': 512, 'type': int},
                'L2_F': {'min': 1, 'max': 512, 'type': int},
                'L1_K': {'min': 1, 'max': 15, 'type': int},
                'L2_K': {'min': 1, 'max': 15, 'type': int},
                'L1_P': {'min': 1, 'max': 10, 'type': int},
                'L2_P': {'min': 1, 'max': 10, 'type': int},
                'L3_N': {'min': 1, 'max': 1024, 'type': int},
                'EP': {'min': 1, 'max': 10000, 'type': int},
                'BS': {'min': 1, 'max': 1024, 'type': int}
            },
            'LSTM': {
                'L1_N': {'min': 1, 'max': 512, 'type': int},
                'L2_N': {'min': 1, 'max': 512, 'type': int},
                'L1_D': {'min': 0.0, 'max': 0.9, 'type': float},
                'L2_D': {'min': 0.0, 'max': 0.9, 'type': float},
                'L3_N': {'min': 1, 'max': 512, 'type': int},
                'EP': {'min': 1, 'max': 10000, 'type': int},
                'BS': {'min': 1, 'max': 1024, 'type': int}
            },
            'SVR': {
                'KERNEL': {'values': ['rbf', 'linear', 'poly', 'sigmoid'], 'type': str},
                'C': {'min': 0.001, 'max': 1000.0, 'type': float},
                'EPSILON': {'min': 0.0, 'max': 10.0, 'type': float},
                'DEGREE': {'min': 1, 'max': 10, 'type': int},
                'CACHE_SIZE': {'min': 50, 'max': 2000, 'type': int}
            },
            'LINEAR': {
                'FIT_INTERCEPT': {'type': bool},
                'NORMALIZE': {'type': bool},
                'COPY_X': {'type': bool},
                'POSITIVE': {'type': bool}
            }
        }
    
    def validate_mdl_config(self, mdl_config: Dict) -> Dict:
        """
        Validate MDL configuration against reference constraints
        
        Args:
            mdl_config: MDL configuration to validate
            
        Returns:
            Dict with validation results and corrected parameters
        """
        try:
            validation_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'corrected_config': mdl_config.copy()
            }
            
            for model_type, model_config in mdl_config.items():
                if model_type in self.validation_rules:
                    model_validation = self._validate_model_config(
                        model_type, model_config, self.validation_rules[model_type]
                    )
                    
                    validation_result['errors'].extend(model_validation['errors'])
                    validation_result['warnings'].extend(model_validation['warnings'])
                    
                    if model_validation['corrected_config']:
                        validation_result['corrected_config'][model_type] = model_validation['corrected_config']
                    
                    if not model_validation['is_valid']:
                        validation_result['is_valid'] = False
                else:
                    validation_result['warnings'].append(f"Unknown model type: {model_type}")
            
            self.logger.info(f"MDL validation completed: valid={validation_result['is_valid']}, "
                           f"errors={len(validation_result['errors'])}, warnings={len(validation_result['warnings'])}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating MDL configuration: {str(e)}")
            raise
    
    def _validate_model_config(self, model_type: str, config: Dict, rules: Dict) -> Dict:
        """Validate individual model configuration"""
        try:
            result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'corrected_config': config.copy()
            }
            
            for param_name, param_value in config.items():
                if param_name in rules:
                    rule = rules[param_name]
                    
                    # Type validation
                    if 'type' in rule and not isinstance(param_value, rule['type']):
                        if rule['type'] == int and isinstance(param_value, float):
                            # Convert float to int if possible
                            result['corrected_config'][param_name] = int(param_value)
                            result['warnings'].append(f"{model_type}.{param_name}: converted float to int")
                        elif rule['type'] == float and isinstance(param_value, int):
                            # Convert int to float
                            result['corrected_config'][param_name] = float(param_value)
                        else:
                            result['errors'].append(f"{model_type}.{param_name}: expected {rule['type'].__name__}, got {type(param_value).__name__}")
                            result['is_valid'] = False
                            continue
                    
                    # Range validation
                    if 'min' in rule and param_value < rule['min']:
                        result['warnings'].append(f"{model_type}.{param_name}: value {param_value} below minimum {rule['min']}")
                        result['corrected_config'][param_name] = rule['min']
                    
                    if 'max' in rule and param_value > rule['max']:
                        result['warnings'].append(f"{model_type}.{param_name}: value {param_value} above maximum {rule['max']}")
                        result['corrected_config'][param_name] = rule['max']
                    
                    # Value validation
                    if 'values' in rule and param_value not in rule['values']:
                        result['errors'].append(f"{model_type}.{param_name}: invalid value '{param_value}', must be one of {rule['values']}")
                        result['is_valid'] = False
                else:
                    result['warnings'].append(f"{model_type}.{param_name}: unknown parameter")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error validating {model_type} config: {str(e)}")
            raise


# Factory functions for easy access
def create_parameter_converter() -> FrontendParameterConverter:
    """Create and return a FrontendParameterConverter instance"""
    return FrontendParameterConverter()


def create_mdl_validator() -> ReferenceMDLValidator:
    """Create and return a ReferenceMDLValidator instance"""
    return ReferenceMDLValidator()


def convert_frontend_parameters_to_mdl(frontend_params: Dict) -> Tuple[Dict, Dict]:
    """
    Convenience function to convert and validate frontend parameters
    
    Args:
        frontend_params: Frontend parameters
        
    Returns:
        Tuple of (mdl_config, validation_result)
    """
    try:
        # Convert parameters
        converter = create_parameter_converter()
        mdl_config = converter.convert_frontend_to_mdl_format(frontend_params)
        
        # Validate converted parameters
        validator = create_mdl_validator()
        validation_result = validator.validate_mdl_config(mdl_config)
        
        return validation_result['corrected_config'], validation_result
        
    except Exception as e:
        logger.error(f"Error in parameter conversion pipeline: {str(e)}")
        raise


# Example usage and testing
if __name__ == "__main__":
    # Example frontend parameters
    example_frontend_params = {
        "dense_layers": [128, 64, 32],
        "dense_activation": "ReLU",
        "dense_epochs": 150,
        "dense_batch_size": 64,
        "dense_optimizer": "Adam",
        "dense_learning_rate": 0.001,
        "cnn_filters": [64, 128],
        "cnn_kernel_size": [5, 3],
        "cnn_activation": "ReLU",
        "cnn_epochs": 100,
        "lstm_units": [100, 50],
        "lstm_dropout": 0.3,
        "lstm_epochs": 200,
        "svr_kernel": "RBF",
        "svr_C": 2.0,
        "svr_epsilon": 0.05,
        "linear_fit_intercept": True
    }
    
    # Convert and validate
    try:
        mdl_config, validation_result = convert_frontend_parameters_to_mdl(example_frontend_params)
        
        print("Conversion successful!")
        print(f"Models configured: {list(mdl_config.keys())}")
        print(f"Validation errors: {len(validation_result['errors'])}")
        print(f"Validation warnings: {len(validation_result['warnings'])}")
        
        if validation_result['errors']:
            print("Errors:", validation_result['errors'])
        if validation_result['warnings']:
            print("Warnings:", validation_result['warnings'])
            
    except Exception as e:
        print(f"Conversion failed: {str(e)}")