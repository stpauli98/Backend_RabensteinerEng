"""
Results generator module for training system
Handles formatting and preparing results for frontend
Contains evaluation metrics extracted from training_backend_test_2.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def wape(y_true, y_pred):
    """
    FUNKTION ZUR BERECHNUNG DES GEWICHTETEN ABSOLUTEN PROZENTUALEN FEHLERS
    
    Extracted from training_backend_test_2.py lines 555-566
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        WAPE value as percentage (0.0 if NaN)
    """
    
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))
    
    if denominator == 0:
        return 0.0

    return (numerator/denominator)*100


def smape(y_true, y_pred):
    """
    FUNKTION ZUR BERECHNUNG DES SYMMETRISCHEN MITTLEREN ABSOLUTEN PROZENTUALEN
    FEHLERS
    
    Extracted from training_backend_test_2.py lines 570-585
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        SMAPE value as percentage
    """

    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)    

    n = len(y_true)
    smape_values = []

    for yt, yp in zip(y_true, y_pred):
        denominator = (abs(yt)+abs(yp))/2
        if denominator == 0:
            smape_values.append(0)
        else:
            smape_values.append(abs(yp-yt)/denominator)

    return sum(smape_values)/n*100


def mase(y_true, y_pred, m = 1):
    """
    FUNKTION ZUR BERECHNUNG DES MITTLEREN ABSOLUTEN FEHLERS
    
    Extracted from training_backend_test_2.py lines 588-608
    
    Args:
        y_true: True values
        y_pred: Predicted values
        m: Seasonality parameter (default 1)
        
    Returns:
        MASE value (0.0 if error)
    """
    try:
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)      
        
        n = len(y_true)
        
        mae_forecast = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n

        if n <= m:
            return 0.0
            
        naive_errors = [abs(y_true[t] - y_true[t - m]) for t in range(m, n)]
        mae_naive = sum(naive_errors) / len(naive_errors)

        if mae_naive == 0:
            return 0.0

        return mae_forecast/mae_naive
        
    except Exception:
        return 0.0


class ResultsGenerator:
    """
    Handles generation and formatting of training results
    Contains evaluation metrics extracted from training_backend_test_2.py
    """
    
    def __init__(self):
        self.results = {}
        self.evaluation_dataframes = {}
    
    def generate_results(self, training_results: Dict, session_data: Dict) -> Dict:
        """
        Generate formatted results for frontend
        
        Args:
            training_results: Results from model training
            session_data: Session configuration data
            
        Returns:
            Dict containing formatted results
        """
        try:
            evaluation_results = self._generate_evaluation_metrics(training_results)
            
            evaluation_dataframes = self._generate_evaluation_dataframes(training_results, session_data)
            
            model_comparison = self._generate_model_comparison(training_results)
            
            training_metadata = self._generate_training_metadata(training_results, session_data)
            
            results = {
                'evaluation_metrics': evaluation_results,
                'evaluation_dataframes': evaluation_dataframes,
                'model_comparison': model_comparison,
                'training_metadata': training_metadata,
                'status': 'completed',
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            self.results = results
            return results
            
        except Exception as e:
            logger.error(f"Error generating results: {str(e)}")
            raise
    
    def _generate_evaluation_metrics(self, training_results: Dict) -> Dict:
        """
        Generate evaluation metrics for all models
        
        Args:
            training_results: Results from model training
            
        Returns:
            Dict containing evaluation metrics
        """
        try:
            evaluation_metrics = {}
            
            for dataset_name, dataset_results in training_results.items():
                dataset_metrics = {}
                
                for model_name, model_result in dataset_results.items():
                    if 'metrics' in model_result:
                        base_metrics = model_result['metrics']
                        
                        if 'predictions' in model_result:
                            y_pred = model_result['predictions']
                            additional_metrics = {
                                'nrmse': 0.0,
                                'wape': 0.0,
                                'smape': 0.0,
                                'mase': 0.0
                            }
                            
                            all_metrics = {**base_metrics, **additional_metrics}
                            dataset_metrics[model_name] = all_metrics
                        else:
                            dataset_metrics[model_name] = base_metrics
                
                evaluation_metrics[dataset_name] = dataset_metrics
            
            return evaluation_metrics
            
        except Exception as e:
            logger.error(f"Error generating evaluation metrics: {str(e)}")
            raise
    
    def _generate_evaluation_dataframes(self, training_results: Dict, session_data: Dict) -> Dict:
        """
        Generate evaluation DataFrames (df_eval, df_eval_ts)
        Similar to what's created in training_backend_test_2.py
        
        Args:
            training_results: Results from model training
            session_data: Session configuration data
            
        Returns:
            Dict containing evaluation DataFrames
        """
        try:
            evaluation_dataframes = {}
            
            for dataset_name, dataset_results in training_results.items():
                df_eval = self._create_evaluation_dataframe(dataset_results)
                
                df_eval_ts = self._create_timeseries_evaluation_dataframe(dataset_results, session_data)
                
                evaluation_dataframes[dataset_name] = {
                    'df_eval': df_eval.to_dict('records') if df_eval is not None else [],
                    'df_eval_ts': df_eval_ts.to_dict('records') if df_eval_ts is not None else []
                }
            
            return evaluation_dataframes
            
        except Exception as e:
            logger.error(f"Error generating evaluation DataFrames: {str(e)}")
            raise
    
    def _create_evaluation_dataframe(self, dataset_results: Dict) -> pd.DataFrame:
        """
        Create evaluation DataFrame from model results
        
        Args:
            dataset_results: Results for a specific dataset
            
        Returns:
            Evaluation DataFrame
        """
        try:
            eval_data = []
            
            for model_name, model_result in dataset_results.items():
                if 'metrics' in model_result:
                    metrics = model_result['metrics']
                    
                    eval_row = {
                        'model': model_name,
                        'mae': metrics.get('mae', 0.0),
                        'mse': metrics.get('mse', 0.0),
                        'rmse': metrics.get('rmse', 0.0),
                        'mape': metrics.get('mape', 0.0),
                        'nrmse': metrics.get('nrmse', 0.0),
                        'wape': metrics.get('wape', 0.0),
                        'smape': metrics.get('smape', 0.0),
                        'mase': metrics.get('mase', 0.0)
                    }
                    
                    eval_data.append(eval_row)
            
            if eval_data:
                return pd.DataFrame(eval_data)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error creating evaluation DataFrame: {str(e)}")
            raise
    
    def _create_timeseries_evaluation_dataframe(self, dataset_results: Dict, session_data: Dict) -> pd.DataFrame:
        """
        Create time series evaluation DataFrame
        
        Args:
            dataset_results: Results for a specific dataset
            session_data: Session configuration data
            
        Returns:
            Time series evaluation DataFrame
        """
        try:
            
            ts_eval_data = []
            
            for model_name, model_result in dataset_results.items():
                if 'predictions' in model_result:
                    predictions = model_result['predictions']
                    
                    for i, pred in enumerate(predictions):
                        ts_eval_row = {
                            'model': model_name,
                            'timestamp': pd.Timestamp.now() + pd.Timedelta(hours=i),
                            'prediction': float(pred.flatten()[0]) if hasattr(pred, 'flatten') else float(pred),
                            'actual': 0.0,
                            'error': 0.0,
                            'abs_error': 0.0,
                            'percentage_error': 0.0
                        }
                        
                        ts_eval_data.append(ts_eval_row)
            
            if ts_eval_data:
                return pd.DataFrame(ts_eval_data)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error creating time series evaluation DataFrame: {str(e)}")
            raise
    
    def _generate_model_comparison(self, training_results: Dict) -> Dict:
        """
        Generate model comparison summary
        
        Args:
            training_results: Results from model training
            
        Returns:
            Dict containing model comparison
        """
        try:
            comparison = {}
            
            for dataset_name, dataset_results in training_results.items():
                model_metrics = {}
                
                for model_name, model_result in dataset_results.items():
                    if 'metrics' in model_result:
                        model_metrics[model_name] = model_result['metrics']
                
                best_models = {}
                
                if model_metrics:
                    all_metrics = set()
                    for metrics in model_metrics.values():
                        all_metrics.update(metrics.keys())
                    
                    for metric in all_metrics:
                        metric_values = {}
                        for model_name, metrics in model_metrics.items():
                            if metric in metrics:
                                metric_values[model_name] = metrics[metric]
                        
                        if metric_values:
                            best_model = min(metric_values, key=metric_values.get)
                            best_models[metric] = {
                                'model': best_model,
                                'value': metric_values[best_model]
                            }
                
                comparison[dataset_name] = {
                    'model_metrics': model_metrics,
                    'best_models': best_models
                }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error generating model comparison: {str(e)}")
            raise
    
    def _generate_training_metadata(self, training_results: Dict, session_data: Dict) -> Dict:
        """
        Generate training metadata
        
        Args:
            training_results: Results from model training
            session_data: Session configuration data
            
        Returns:
            Dict containing training metadata
        """
        try:
            metadata = {
                'total_models_trained': 0,
                'successful_models': 0,
                'failed_models': 0,
                'datasets_processed': len(training_results),
                'training_configuration': session_data,
                'model_details': {}
            }
            
            for dataset_name, dataset_results in training_results.items():
                dataset_metadata = {
                    'models_trained': len(dataset_results),
                    'model_names': list(dataset_results.keys()),
                    'training_history': {}
                }
                
                metadata['total_models_trained'] += len(dataset_results)
                
                for model_name, model_result in dataset_results.items():
                    if 'metrics' in model_result:
                        metadata['successful_models'] += 1
                        
                        if 'history' in model_result:
                            dataset_metadata['training_history'][model_name] = model_result['history']
                    else:
                        metadata['failed_models'] += 1
                
                metadata['model_details'][dataset_name] = dataset_metadata
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating training metadata: {str(e)}")
            raise
    
    def wape(self, y_true, y_pred):
        """
        FUNKTION ZUR BERECHNUNG DES GEWICHTETEN ABSOLUTEN PROZENTUALEN FEHLERS
        
        Extracted from training_backend_test_2.py lines 555-566
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            WAPE value as percentage (0.0 if NaN)
        """
        
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)
        
        numerator = np.sum(np.abs(y_true - y_pred))
        denominator = np.sum(np.abs(y_true))
        
        if denominator == 0:
            return 0.0

        return (numerator/denominator)*100
    
    def smape(self, y_true, y_pred):
        """
        FUNKTION ZUR BERECHNUNG DES SYMMETRISCHEN MITTLEREN ABSOLUTEN PROZENTUALEN
        FEHLERS
        
        Extracted from training_backend_test_2.py lines 570-585
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            SMAPE value as percentage
        """

        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)    

        n = len(y_true)
        smape_values = []

        for yt, yp in zip(y_true, y_pred):
            denominator = (abs(yt)+abs(yp))/2
            if denominator == 0:
                smape_values.append(0)
            else:
                smape_values.append(abs(yp-yt)/denominator)

        return sum(smape_values)/n*100
    
    def mase(self, y_true, y_pred, m = 1):
        """
        FUNKTION ZUR BERECHNUNG DES MITTLEREN ABSOLUTEN FEHLERS
        
        Extracted from training_backend_test_2.py lines 588-608
        
        Args:
            y_true: True values
            y_pred: Predicted values
            m: Seasonality parameter (default 1)
            
        Returns:
            MASE value (0.0 if error)
        """
        try:
            y_true = np.array(y_true, dtype=np.float64)
            y_pred = np.array(y_pred, dtype=np.float64)      
            
            n = len(y_true)
            
            mae_forecast = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n

            if n <= m:
                return 0.0
                
            naive_errors = [abs(y_true[t] - y_true[t - m]) for t in range(m, n)]
            mae_naive = sum(naive_errors) / len(naive_errors)

            if mae_naive == 0:
                return 0.0

            return mae_forecast/mae_naive
            
        except Exception:
            return 0.0
    
    def _clean_json_values(self, obj):
        """Clean NaN and Infinity values from nested dictionary/list for JSON serialization"""
        import math
        
        if isinstance(obj, dict):
            return {k: self._clean_json_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_json_values(item) for item in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        else:
            return obj
    
    def save_results_to_database(self, session_id: str, supabase_client) -> bool:
        """
        Save results to database
        
        Args:
            session_id: Session identifier
            supabase_client: Supabase client instance
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.results:
                logger.warning("No results to save")
                return False
            
            from utils.database import create_or_get_session_uuid
            
            uuid_session_id = create_or_get_session_uuid(session_id)
            if not uuid_session_id:
                logger.error(f"Could not get UUID for session {session_id}")
                return False
            
            cleaned_results = self._clean_json_values(self.results)
            
            result_data = {
                'session_id': uuid_session_id,
                'results': cleaned_results,
                'created_at': pd.Timestamp.now().isoformat(),
                'status': 'completed'
            }
            
            response = supabase_client.table('training_results').insert(result_data).execute()
            
            if response.data:
                return True
            else:
                logger.error(f"Failed to save results to database for session {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving results to database: {str(e)}")
            return False


def create_results_generator() -> ResultsGenerator:
    """
    Create and return a ResultsGenerator instance
    
    Returns:
        ResultsGenerator instance
    """
    return ResultsGenerator()
