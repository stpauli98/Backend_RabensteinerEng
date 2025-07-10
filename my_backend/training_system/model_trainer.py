"""
Model trainer module for training system
Contains all ML model training functions extracted from training_backend_test_2.py
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


class ModelTrainer:
    """
    Main model trainer class
    Contains all model training functions extracted from training_backend_test_2.py
    """
    
    def __init__(self, config: MDL):
        self.config = config
        self.trained_models = {}
        self.training_history = {}
    
    def train_all_models(self, datasets: Dict, session_data: Dict) -> Dict:
        """
        Train all enabled models
        
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
                
                # Train each enabled model
                if self.config.models.get('dense', False):
                    dataset_results['dense'] = self.train_dense(X_train, y_train, X_test, y_test)
                
                if self.config.models.get('cnn', False):
                    dataset_results['cnn'] = self.train_cnn(X_train, y_train, X_test, y_test)
                
                if self.config.models.get('lstm', False):
                    dataset_results['lstm'] = self.train_lstm(X_train, y_train, X_test, y_test)
                
                if self.config.models.get('ar_lstm', False):
                    dataset_results['ar_lstm'] = self.train_ar_lstm(X_train, y_train, X_test, y_test)
                
                if self.config.models.get('svr_dir', False):
                    dataset_results['svr_dir'] = self.train_svr_dir(X_train, y_train, X_test, y_test)
                
                if self.config.models.get('svr_mimo', False):
                    dataset_results['svr_mimo'] = self.train_svr_mimo(X_train, y_train, X_test, y_test)
                
                if self.config.models.get('linear', False):
                    dataset_results['linear'] = self.train_linear_model(X_train, y_train, X_test, y_test)
                
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
            
            return train_test_split(X, y, test_size=self.config.test_size, 
                                  random_state=self.config.random_state)
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def train_dense(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train Dense Neural Network model
        Extracted from training_backend_test_2.py around lines 170-220
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict containing model and results
        """
        try:
            # TODO: Extract actual dense model architecture from training_backend_test_2.py
            # This is placeholder implementation
            
            model = Sequential()
            
            # Input layer
            model.add(Dense(self.config.dense_layers[0], 
                          activation=self.config.dense_activation,
                          input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Flatten())
            
            # Hidden layers
            for units in self.config.dense_layers[1:]:
                model.add(Dense(units, activation=self.config.dense_activation))
                model.add(Dropout(self.config.dense_dropout))
            
            # Output layer
            model.add(Dense(y_train.shape[-1] if len(y_train.shape) > 1 else 1))
            
            # Compile model
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Train model
            callbacks = []
            if self.config.early_stopping:
                callbacks.append(EarlyStopping(patience=self.config.patience, 
                                             min_delta=self.config.min_delta))
            
            history = model.fit(X_train, y_train,
                              epochs=100,
                              batch_size=32,
                              validation_split=0.2,
                              callbacks=callbacks,
                              verbose=0)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            
            result = {
                'model': model,
                'history': history.history,
                'predictions': y_pred,
                'metrics': metrics
            }
            
            self.trained_models['dense'] = model
            self.training_history['dense'] = history.history
            
            return result
            
        except Exception as e:
            logger.error(f"Error training dense model: {str(e)}")
            raise
    
    def train_cnn(self, X_train: np.ndarray, y_train: np.ndarray, 
                  X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train CNN model
        Extracted from training_backend_test_2.py around lines 221-280
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict containing model and results
        """
        try:
            # TODO: Extract actual CNN architecture from training_backend_test_2.py
            
            model = Sequential()
            
            # CNN layers
            for i, filters in enumerate(self.config.cnn_filters):
                if i == 0:
                    model.add(Conv1D(filters=filters, 
                                   kernel_size=self.config.cnn_kernel_size,
                                   activation='relu',
                                   input_shape=(X_train.shape[1], X_train.shape[2])))
                else:
                    model.add(Conv1D(filters=filters, 
                                   kernel_size=self.config.cnn_kernel_size,
                                   activation='relu'))
                
                model.add(MaxPooling1D(pool_size=self.config.cnn_pool_size))
            
            # Flatten and dense layers
            model.add(Flatten())
            model.add(Dense(50, activation='relu'))
            model.add(Dense(y_train.shape[-1] if len(y_train.shape) > 1 else 1))
            
            # Compile and train
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            callbacks = []
            if self.config.early_stopping:
                callbacks.append(EarlyStopping(patience=self.config.patience))
            
            history = model.fit(X_train, y_train,
                              epochs=100,
                              batch_size=32,
                              validation_split=0.2,
                              callbacks=callbacks,
                              verbose=0)
            
            y_pred = model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)
            
            result = {
                'model': model,
                'history': history.history,
                'predictions': y_pred,
                'metrics': metrics
            }
            
            self.trained_models['cnn'] = model
            self.training_history['cnn'] = history.history
            
            return result
            
        except Exception as e:
            logger.error(f"Error training CNN model: {str(e)}")
            raise
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train LSTM model
        Extracted from training_backend_test_2.py around lines 281-340
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict containing model and results
        """
        try:
            # TODO: Extract actual LSTM architecture from training_backend_test_2.py
            
            model = Sequential()
            
            # LSTM layers
            for i, units in enumerate(self.config.lstm_units):
                return_sequences = (i < len(self.config.lstm_units) - 1) or self.config.lstm_return_sequences
                
                if i == 0:
                    model.add(LSTM(units, 
                                 return_sequences=return_sequences,
                                 input_shape=(X_train.shape[1], X_train.shape[2])))
                else:
                    model.add(LSTM(units, return_sequences=return_sequences))
                
                model.add(Dropout(self.config.lstm_dropout))
            
            # Output layer
            model.add(Dense(y_train.shape[-1] if len(y_train.shape) > 1 else 1))
            
            # Compile and train
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            callbacks = []
            if self.config.early_stopping:
                callbacks.append(EarlyStopping(patience=self.config.patience))
            
            history = model.fit(X_train, y_train,
                              epochs=100,
                              batch_size=32,
                              validation_split=0.2,
                              callbacks=callbacks,
                              verbose=0)
            
            y_pred = model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)
            
            result = {
                'model': model,
                'history': history.history,
                'predictions': y_pred,
                'metrics': metrics
            }
            
            self.trained_models['lstm'] = model
            self.training_history['lstm'] = history.history
            
            return result
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            raise
    
    def train_ar_lstm(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train Autoregressive LSTM model
        Extracted from training_backend_test_2.py around lines 341-400
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict containing model and results
        """
        try:
            # TODO: Extract actual AR-LSTM architecture from training_backend_test_2.py
            # This is placeholder implementation
            
            model = Sequential()
            
            # AR-LSTM layers
            model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(y_train.shape[-1] if len(y_train.shape) > 1 else 1))
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            callbacks = []
            if self.config.early_stopping:
                callbacks.append(EarlyStopping(patience=self.config.patience))
            
            history = model.fit(X_train, y_train,
                              epochs=100,
                              batch_size=32,
                              validation_split=0.2,
                              callbacks=callbacks,
                              verbose=0)
            
            y_pred = model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)
            
            result = {
                'model': model,
                'history': history.history,
                'predictions': y_pred,
                'metrics': metrics
            }
            
            self.trained_models['ar_lstm'] = model
            self.training_history['ar_lstm'] = history.history
            
            return result
            
        except Exception as e:
            logger.error(f"Error training AR-LSTM model: {str(e)}")
            raise
    
    def train_svr_dir(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train SVR Direct model
        Extracted from training_backend_test_2.py around lines 401-460
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict containing model and results
        """
        try:
            # TODO: Extract actual SVR Direct implementation from training_backend_test_2.py
            
            # Flatten input for SVR
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            y_train_flat = y_train.reshape(y_train.shape[0], -1)
            
            # Create and train SVR model
            model = make_pipeline(StandardScaler(), 
                                SVR(kernel=self.config.svr_kernel,
                                    C=self.config.svr_C,
                                    epsilon=self.config.svr_epsilon))
            
            model.fit(X_train_flat, y_train_flat.ravel())
            
            # Predict
            y_pred = model.predict(X_test_flat)
            y_pred = y_pred.reshape(-1, 1)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            
            result = {
                'model': model,
                'predictions': y_pred,
                'metrics': metrics
            }
            
            self.trained_models['svr_dir'] = model
            
            return result
            
        except Exception as e:
            logger.error(f"Error training SVR Direct model: {str(e)}")
            raise
    
    def train_svr_mimo(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train SVR MIMO model
        Extracted from training_backend_test_2.py around lines 461-520
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict containing model and results
        """
        try:
            # TODO: Extract actual SVR MIMO implementation from training_backend_test_2.py
            
            # Flatten input for SVR
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            # Train multiple SVR models for MIMO
            models = []
            predictions = []
            
            for i in range(y_train.shape[-1]):
                model = make_pipeline(StandardScaler(), 
                                    SVR(kernel=self.config.svr_kernel,
                                        C=self.config.svr_C,
                                        epsilon=self.config.svr_epsilon))
                
                model.fit(X_train_flat, y_train[:, i])
                pred = model.predict(X_test_flat)
                
                models.append(model)
                predictions.append(pred)
            
            y_pred = np.column_stack(predictions)
            metrics = self._calculate_metrics(y_test, y_pred)
            
            result = {
                'models': models,
                'predictions': y_pred,
                'metrics': metrics
            }
            
            self.trained_models['svr_mimo'] = models
            
            return result
            
        except Exception as e:
            logger.error(f"Error training SVR MIMO model: {str(e)}")
            raise
    
    def train_linear_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train Linear Regression model
        Extracted from training_backend_test_2.py around lines 521-553
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict containing model and results
        """
        try:
            # TODO: Extract actual linear model implementation from training_backend_test_2.py
            
            # Flatten input for linear regression
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            y_train_flat = y_train.reshape(y_train.shape[0], -1)
            
            # Create and train linear model
            model = LinearRegression()
            model.fit(X_train_flat, y_train_flat)
            
            # Predict
            y_pred = model.predict(X_test_flat)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            
            result = {
                'model': model,
                'predictions': y_pred,
                'metrics': metrics
            }
            
            self.trained_models['linear'] = model
            
            return result
            
        except Exception as e:
            logger.error(f"Error training linear model: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dict containing metrics
        """
        try:
            # Flatten arrays for metric calculation
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
            
            metrics = {
                'mae': float(mean_absolute_error(y_true_flat, y_pred_flat)),
                'mse': float(mean_squared_error(y_true_flat, y_pred_flat)),
                'rmse': float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))),
                'mape': float(mean_absolute_percentage_error(y_true_flat, y_pred_flat))
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise


# Factory function to create model trainer
def create_model_trainer(config: MDL) -> ModelTrainer:
    """
    Create and return a ModelTrainer instance
    
    Args:
        config: MDL configuration object
        
    Returns:
        ModelTrainer instance
    """
    return ModelTrainer(config)