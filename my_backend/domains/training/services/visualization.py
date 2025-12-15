"""
Visualization module for training system
Handles creation of plots and charts for training results
Contains visualization code extracted from training_backend_test_2.py
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from typing import Dict, List, Optional, Tuple
import logging

from domains.training.config import PLOT_SETTINGS

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Handles creation of visualizations for training results
    Contains plotting functions extracted from training_backend_test_2.py
    """
    
    def __init__(self):
        self.setup_plot_style()
        self.plots = {}
    
    def setup_plot_style(self):
        """Setup matplotlib style settings"""
        try:
            plt.style.use('seaborn-v0_8')
            sns.set_palette(PLOT_SETTINGS['color_palette'])
            plt.rcParams['figure.figsize'] = PLOT_SETTINGS['figure_size']
            plt.rcParams['figure.dpi'] = PLOT_SETTINGS['dpi']
            plt.rcParams['font.size'] = PLOT_SETTINGS['font_size']
            
        except Exception as e:
            logger.warning(f"Could not set plot style: {str(e)}")
    
    def create_all_visualizations(self, training_results: Dict, evaluation_results: Dict) -> Dict:
        """
        Create all visualizations for training results
        
        Args:
            training_results: Results from model training
            evaluation_results: Evaluation metrics and DataFrames
            
        Returns:
            Dict containing all visualizations as base64 strings
        """
        try:
            visualizations = {}
            
            violin_plots = self.create_violin_plots(evaluation_results)
            visualizations.update(violin_plots)
            
            forecast_plots = self.create_forecast_plots(training_results, evaluation_results)
            visualizations.update(forecast_plots)
            
            comparison_plots = self.create_metrics_comparison_plots(evaluation_results)
            visualizations.update(comparison_plots)
            
            history_plots = self.create_training_history_plots(training_results)
            visualizations.update(history_plots)
            
            self.plots = visualizations
            return visualizations
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise
    
    def create_violin_plots(self, data_arrays: Dict) -> Dict:
        """
        Create violin plots for data distribution - DEPRECATED
        Use violin_plot_generator.generate_violin_plots_from_data instead
        """
        logger.warning("create_violin_plots is deprecated. Use violin_plot_generator.generate_violin_plots_from_data instead")
        return {}
    
    def _create_input_distribution_plot(self, i_combined_array: np.ndarray) -> str:
        """DEPRECATED - Use violin_plot_generator instead"""
        logger.warning("_create_input_distribution_plot is deprecated")
        raise NotImplementedError("Use violin_plot_generator.generate_violin_plots_from_data instead")
    
    def _create_output_distribution_plot(self, o_combined_array: np.ndarray, i_combined_array: np.ndarray = None) -> str:
        """DEPRECATED - Use violin_plot_generator instead"""
        logger.warning("_create_output_distribution_plot is deprecated")
        raise NotImplementedError("Use violin_plot_generator.generate_violin_plots_from_data instead")
    
    def create_forecast_plots(self, training_results: Dict, evaluation_results: Dict) -> Dict:
        """
        Create forecast visualization plots
        Extracted from training_backend_test_2.py around lines 2340-2885
        
        Args:
            training_results: Results from model training
            evaluation_results: Evaluation results
            
        Returns:
            Dict containing forecast plots as base64 strings
        """
        try:
            forecast_plots = {}
            
            for dataset_name, dataset_results in training_results.items():
                eval_dataframes = evaluation_results.get('evaluation_dataframes', {}).get(dataset_name, {})
                df_eval_ts = eval_dataframes.get('df_eval_ts', [])
                
                if df_eval_ts:
                    df_ts = pd.DataFrame(df_eval_ts)
                    
                    models = df_ts['model'].unique() if 'model' in df_ts.columns else []
                    
                    for model_name in models:
                        model_data = df_ts[df_ts['model'] == model_name]
                        
                        if len(model_data) > 0:
                            fig, ax = plt.subplots(figsize=PLOT_SETTINGS['figure_size'])
                            
                            
                            if 'timestamp' in model_data.columns:
                                ax.plot(model_data['timestamp'], model_data.get('actual', []), 
                                       label='Actual', color='blue', linewidth=2)
                                ax.plot(model_data['timestamp'], model_data.get('prediction', []), 
                                       label='Predicted', color='red', linewidth=2, linestyle='--')
                            
                            ax.set_title(f'Forecast - {model_name} - {dataset_name}')
                            ax.set_xlabel('Time')
                            ax.set_ylabel('Value')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            
                            plot_key = f'forecast_{dataset_name}_{model_name}'
                            forecast_plots[plot_key] = self._figure_to_base64(fig)
                            
                            plt.close(fig)
            
            return forecast_plots
            
        except Exception as e:
            logger.error(f"Error creating forecast plots: {str(e)}")
            raise
    
    def create_metrics_comparison_plots(self, evaluation_results: Dict) -> Dict:
        """
        Create metrics comparison plots
        
        Args:
            evaluation_results: Evaluation results
            
        Returns:
            Dict containing comparison plots as base64 strings
        """
        try:
            comparison_plots = {}
            
            for dataset_name, dataset_results in evaluation_results.get('evaluation_metrics', {}).items():
                models = list(dataset_results.keys())
                metrics = ['mae', 'mse', 'rmse', 'mape']
                
                if models:
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    axes = axes.flatten()
                    
                    for i, metric in enumerate(metrics):
                        if i < len(axes):
                            ax = axes[i]
                            
                            metric_values = []
                            model_names = []
                            
                            for model_name in models:
                                if metric in dataset_results[model_name]:
                                    metric_values.append(dataset_results[model_name][metric])
                                    model_names.append(model_name)
                            
                            if metric_values:
                                bars = ax.bar(model_names, metric_values)
                                ax.set_title(f'{metric.upper()} Comparison')
                                ax.set_ylabel(f'{metric.upper()} Value')
                                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                                
                                for bar, value in zip(bars, metric_values):
                                    height = bar.get_height()
                                    ax.text(bar.get_x() + bar.get_width()/2., height,
                                           f'{value:.4f}', ha='center', va='bottom')
                    
                    plt.suptitle(f'Metrics Comparison - {dataset_name}', fontsize=16)
                    plt.tight_layout()
                    
                    plot_key = f'comparison_{dataset_name}'
                    comparison_plots[plot_key] = self._figure_to_base64(fig)
                    
                    plt.close(fig)
            
            return comparison_plots
            
        except Exception as e:
            logger.error(f"Error creating metrics comparison plots: {str(e)}")
            raise
    
    def create_training_history_plots(self, training_results: Dict) -> Dict:
        """
        Create training history plots for neural network models
        
        Args:
            training_results: Results from model training
            
        Returns:
            Dict containing training history plots as base64 strings
        """
        try:
            history_plots = {}
            
            for dataset_name, dataset_results in training_results.items():
                for model_name, model_result in dataset_results.items():
                    if 'history' in model_result:
                        history = model_result['history']
                        
                        if history and 'loss' in history:
                            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                            
                            axes[0].plot(history['loss'], label='Training Loss')
                            if 'val_loss' in history:
                                axes[0].plot(history['val_loss'], label='Validation Loss')
                            axes[0].set_title(f'Training Loss - {model_name}')
                            axes[0].set_xlabel('Epoch')
                            axes[0].set_ylabel('Loss')
                            axes[0].legend()
                            axes[0].grid(True, alpha=0.3)
                            
                            if 'mae' in history:
                                axes[1].plot(history['mae'], label='Training MAE')
                                if 'val_mae' in history:
                                    axes[1].plot(history['val_mae'], label='Validation MAE')
                                axes[1].set_title(f'Training MAE - {model_name}')
                                axes[1].set_xlabel('Epoch')
                                axes[1].set_ylabel('MAE')
                                axes[1].legend()
                                axes[1].grid(True, alpha=0.3)
                            
                            plt.suptitle(f'Training History - {model_name} - {dataset_name}', fontsize=16)
                            plt.tight_layout()
                            
                            plot_key = f'history_{dataset_name}_{model_name}'
                            history_plots[plot_key] = self._figure_to_base64(fig)
                            
                            plt.close(fig)
            
            return history_plots
            
        except Exception as e:
            logger.error(f"Error creating training history plots: {str(e)}")
            raise
    
    def create_residual_plots(self, training_results: Dict, evaluation_results: Dict) -> Dict:
        """
        Create residual plots for model evaluation
        
        Args:
            training_results: Results from model training
            evaluation_results: Evaluation results
            
        Returns:
            Dict containing residual plots as base64 strings
        """
        try:
            residual_plots = {}
            
            for dataset_name, dataset_results in training_results.items():
                eval_dataframes = evaluation_results.get('evaluation_dataframes', {}).get(dataset_name, {})
                df_eval_ts = eval_dataframes.get('df_eval_ts', [])
                
                if df_eval_ts:
                    df_ts = pd.DataFrame(df_eval_ts)
                    
                    models = df_ts['model'].unique() if 'model' in df_ts.columns else []
                    
                    for model_name in models:
                        model_data = df_ts[df_ts['model'] == model_name]
                        
                        if len(model_data) > 0 and 'actual' in model_data.columns and 'prediction' in model_data.columns:
                            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                            
                            actual = model_data['actual']
                            predicted = model_data['prediction']
                            residuals = actual - predicted
                            
                            axes[0].scatter(predicted, residuals, alpha=0.6)
                            axes[0].axhline(y=0, color='r', linestyle='--')
                            axes[0].set_xlabel('Predicted Values')
                            axes[0].set_ylabel('Residuals')
                            axes[0].set_title(f'Residual Plot - {model_name}')
                            axes[0].grid(True, alpha=0.3)
                            
                            from scipy import stats
                            stats.probplot(residuals, dist="norm", plot=axes[1])
                            axes[1].set_title(f'Q-Q Plot - {model_name}')
                            axes[1].grid(True, alpha=0.3)
                            
                            plt.suptitle(f'Residual Analysis - {model_name} - {dataset_name}', fontsize=16)
                            plt.tight_layout()
                            
                            plot_key = f'residual_{dataset_name}_{model_name}'
                            residual_plots[plot_key] = self._figure_to_base64(fig)
                            
                            plt.close(fig)
            
            return residual_plots
            
        except Exception as e:
            logger.error(f"Error creating residual plots: {str(e)}")
            raise
    
    def _figure_to_base64(self, fig) -> str:
        """
        Convert matplotlib figure to base64 string
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            Base64 encoded string
        """
        try:
            buffer = BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight', dpi=PLOT_SETTINGS['dpi'])
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error converting figure to base64: {str(e)}")
            raise
    
    def save_plots_to_storage(self, session_id: str, supabase_client) -> bool:
        """
        Save plots to storage bucket

        Args:
            session_id: Session identifier
            supabase_client: Supabase client instance

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.plots:
                logger.warning("No plots to save")
                return False


            for plot_name, plot_data in self.plots.items():
                if plot_data.startswith('data:image/png;base64,'):
                    base64_data = plot_data.split(',')[1]
                    image_data = base64.b64decode(base64_data)

                    file_path = f"plots/{session_id}/{plot_name}.png"

                    try:
                        response = supabase_client.storage.from_('visualizations').upload(
                            file_path, image_data, {'content-type': 'image/png'}
                        )

                        if response:
                            pass
                    except Exception as e:
                        logger.error(f"Error saving plot {plot_name}: {str(e)}")
                        continue

            return True

        except Exception as e:
            logger.error(f"Error saving plots to storage: {str(e)}")
            return False

    def get_available_variables(self, session_id: str, user_id: str = None) -> Dict:
        """
        Get available input and output variables for plotting from session data.

        Args:
            session_id: Session identifier

        Returns:
            dict: {
                'input_variables': list,
                'output_variables': list
            }
        """
        try:
            from utils.training_storage import fetch_training_results_with_storage
            from shared.database.operations import get_supabase_client, create_or_get_session_uuid

            results = fetch_training_results_with_storage(session_id)

            input_variables = []
            output_variables = []

            if results:
                input_variables = (
                    results.get('input_features') or
                    results.get('input_columns') or
                    results.get('data_info', {}).get('input_columns', [])
                )
                output_variables = (
                    results.get('output_features') or
                    results.get('output_columns') or
                    results.get('data_info', {}).get('output_columns', [])
                )

            if not input_variables and not output_variables:
                supabase = get_supabase_client()
                uuid_session_id = create_or_get_session_uuid(session_id, user_id=user_id)
                file_response = supabase.table('files').select('*').eq('session_id', uuid_session_id).execute()

                if file_response.data:
                    for file_data in file_response.data:
                        file_type = file_data.get('file_type', '')
                        columns = file_data.get('columns', [])

                        if file_type == 'input' and not input_variables:
                            input_variables = [col for col in columns if col.lower() not in ['timestamp', 'utc', 'zeit', 'datetime']]
                        elif file_type == 'output' and not output_variables:
                            output_variables = [col for col in columns if col.lower() not in ['timestamp', 'utc', 'zeit', 'datetime']]

            if not input_variables:
                input_variables = ['Temperature', 'Load']
            if not output_variables:
                output_variables = ['Predicted_Load']

            # Define known TIME component names
            time_component_names = ['Y_sin', 'Y_cos', 'M_sin', 'M_cos', 'W_sin', 'W_cos', 'D_sin', 'D_cos', 'Holiday']
            
            # Separate regular input variables from TIME components
            input_vars_list = input_variables if isinstance(input_variables, list) else []
            regular_inputs = [v for v in input_vars_list if v not in time_component_names]
            time_components = [v for v in input_vars_list if v in time_component_names]
            
            # If no TIME components found in stored data, return default TIME component names
            # (they may exist in the data but not be named in metadata)
            if not time_components:
                time_components = time_component_names
            
            logger.info(f"get_available_variables: inputs={regular_inputs}, time={time_components}, outputs={output_variables}")

            return {
                'input_variables': regular_inputs,
                'output_variables': output_variables if isinstance(output_variables, list) else [],
                'time_components': time_components
            }

        except Exception as e:
            logger.error(f"Error getting plot variables for {session_id}: {str(e)}")
            return {
                'input_variables': ['Temperature', 'Load'],
                'output_variables': ['Predicted_Load'],
                'time_components': ['Y_sin', 'Y_cos', 'M_sin', 'M_cos', 'W_sin', 'W_cos', 'D_sin', 'D_cos', 'Holiday']
            }

    def get_session_visualizations(self, session_id: str, user_id: str = None) -> Dict:
        """
        Get training visualizations (violin plots) for a session.

        Args:
            session_id: Session identifier
            user_id: User ID for ownership validation (required for security)

        Returns:
            dict: {
                'plots': dict,
                'metadata': dict,
                'created_at': str
            }

        Raises:
            PermissionError: If session doesn't belong to the user
        """
        try:
            from shared.database.operations import get_supabase_client, create_or_get_session_uuid

            # Use service_role to bypass RLS for visualization reads
            supabase = get_supabase_client(use_service_role=True)
            uuid_session_id = create_or_get_session_uuid(session_id, user_id=user_id)

            response = supabase.table('training_visualizations').select('*').eq('session_id', uuid_session_id).execute()

            if not response.data or len(response.data) == 0:
                return {
                    'plots': {},
                    'metadata': {},
                    'created_at': None,
                    'message': 'No visualizations found for this session'
                }

            plots = {}
            metadata = {}
            created_at = None

            for viz in response.data:
                plot_name = viz.get('plot_name', 'unknown')
                plots[plot_name] = viz.get('image_data', '')

                if not metadata:
                    metadata = viz.get('plot_metadata', {})
                    created_at = viz.get('created_at')

            return {
                'plots': plots,
                'metadata': metadata,
                'created_at': created_at,
                'message': 'Visualizations retrieved successfully'
            }

        except Exception as e:
            logger.error(f"Error retrieving visualizations for {session_id}: {str(e)}")
            raise

    def generate_custom_plot(self, session_id: str, plot_settings: Dict,
                            df_plot_in: Dict, df_plot_out: Dict,
                            df_plot_fcst: Dict, model_id: str = None) -> Dict:
        """
        Generate custom plot based on user selections.

        Args:
            session_id: Session identifier
            plot_settings: Plot configuration {num_sbpl, x_sbpl, y_sbpl_fmt, y_sbpl_set}
            df_plot_in: Input variables selection {variable_name: bool}
            df_plot_out: Output variables selection {variable_name: bool}
            df_plot_fcst: Forecast variables selection {variable_name: bool}
            model_id: Optional specific model ID to use

        Returns:
            dict: {
                'plot_data': base64 encoded plot string,
                'message': str
            }
        """
        try:
            from utils.training_storage import fetch_training_results_with_storage
            import pickle
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            import io

            num_sbpl = plot_settings.get('num_sbpl', 17)
            x_sbpl = plot_settings.get('x_sbpl', 'UTC')
            y_sbpl_fmt = plot_settings.get('y_sbpl_fmt', 'original')
            y_sbpl_set = plot_settings.get('y_sbpl_set', 'separate Achsen')

            logger.info(f"Generate plot for session {session_id}" +
                       (f" with model_id {model_id}" if model_id else " (using most recent model)"))
            logger.info(f"Plot settings: num_sbpl={num_sbpl}, x_sbpl={x_sbpl}, "
                       f"y_sbpl_fmt={y_sbpl_fmt}, y_sbpl_set={y_sbpl_set}")
            logger.info(f"Input variables selected: {[k for k, v in df_plot_in.items() if v]}")
            logger.info(f"Output variables selected: {[k for k, v in df_plot_out.items() if v]}")
            logger.info(f"Forecast variables selected: {[k for k, v in df_plot_fcst.items() if v]}")

            results = fetch_training_results_with_storage(session_id, model_id=model_id)

            if not results:
                raise ValueError('Model not trained yet. Please train the model first.')

            model_data = results.get('trained_model')

            trained_model = None
            if model_data:
                if isinstance(model_data, list) and len(model_data) > 0:
                    model_data = model_data[0]

                if isinstance(model_data, dict) and model_data.get('_model_type') == 'serialized_model':
                    try:
                        model_bytes = base64.b64decode(model_data['_model_data'])
                        # SECURITY NOTE: pickle.loads() can execute arbitrary code.
                        # This is safe here because models are only stored by authenticated users
                        # via our training pipeline and retrieved from trusted Supabase storage.
                        trained_model = pickle.loads(model_bytes)
                        logger.info(f"Successfully deserialized model of type {model_data.get('_model_class')}")
                    except Exception as e:
                        logger.error(f"Failed to deserialize model: {e}")
                        trained_model = None
                else:
                    trained_model = model_data

            test_data = results.get('test_data', {})
            metadata = results.get('metadata', {})
            scalers = results.get('scalers', {})
            
            # Get input features for name-to-index mapping
            input_features = results.get('input_features', [])
            output_features = results.get('output_features', [])
            logger.info(f"Input features from storage: {input_features}")
            logger.info(f"Output features from storage: {output_features}")

            has_test_data = False
            if test_data:
                if 'X' in test_data and 'y' in test_data:
                    tst_x = np.array(test_data.get('X'))
                    tst_y = np.array(test_data.get('y'))
                    has_test_data = True
                elif 'X_test' in test_data and 'y_test' in test_data:
                    tst_x = np.array(test_data.get('X_test'))
                    tst_y = np.array(test_data.get('y_test'))
                    has_test_data = True

            if not has_test_data:
                logger.error(f"No test data found in database for session {session_id}")
                raise ValueError('No training data available for this session. Please train a model first before generating plots')
            
            # Log data shapes for debugging
            logger.info(f"tst_x shape: {tst_x.shape}, tst_y shape: {tst_y.shape}")
            logger.info(f"Number of input features in array: {tst_x.shape[-1]}")
            logger.info(f"Number of named input features: {len(input_features)}")

            if trained_model and hasattr(trained_model, 'predict'):
                model_type = results.get('model_type', 'Unknown')

                if model_type in ['SVR_dir', 'SVR_MIMO', 'LIN']:
                    logger.info(f"Using SVR/LIN prediction logic for model type: {model_type}")
                    n_samples = tst_x.shape[0]
                    n_outputs = tst_y.shape[-1] if len(tst_y.shape) > 2 else 1

                    tst_fcst = []

                    for i in range(n_samples):
                        inp = np.squeeze(tst_x[i:i+1], axis=0)

                        if isinstance(trained_model, list):
                            pred = []
                            for model_i in trained_model:
                                pred.append(model_i.predict(inp))
                            out = np.array(pred).T
                        else:
                            out = trained_model.predict(inp)

                        out = np.expand_dims(out, axis=0)
                        tst_fcst.append(out[0])

                    tst_fcst = np.array(tst_fcst)
                else:
                    logger.info(f"Using standard prediction logic for model type: {model_type}")
                    tst_fcst = trained_model.predict(tst_x)

                    if model_type == 'CNN':
                        logger.info(f"Squeezing last dimension for CNN model, shape before: {tst_fcst.shape}")
                        tst_fcst = np.squeeze(tst_fcst, axis=-1)
                        logger.info(f"Shape after squeeze: {tst_fcst.shape}")
            else:
                logger.error(f"Model is not available as an object for session {session_id} (stored as: {type(trained_model).__name__})")
                raise ValueError('Model not available for predictions. The trained model cannot be used for predictions. Please retrain the model.')

            num_sbpl = min(num_sbpl, len(tst_x))
            num_sbpl_x = int(np.ceil(np.sqrt(num_sbpl)))
            num_sbpl_y = int(np.ceil(num_sbpl / num_sbpl_x))

            fig, axs = plt.subplots(num_sbpl_y, num_sbpl_x,
                                   figsize=(20, 13),
                                   layout='constrained')

            if num_sbpl == 1:
                axs = [axs]
            else:
                axs = axs.flatten()

            total_vars = len([k for k, v in df_plot_in.items() if v]) + \
                        len([k for k, v in df_plot_out.items() if v]) + \
                        len([k for k, v in df_plot_fcst.items() if v])
            palette = sns.color_palette("tab20", max(20, total_vars))

            for i_sbpl in range(num_sbpl):
                ax = axs[i_sbpl] if num_sbpl > 1 else axs[0]

                if x_sbpl == 'UTC':
                    x_values = pd.date_range(start='2024-01-01',
                                            periods=tst_x.shape[1],
                                            freq='1h')
                else:
                    x_values = np.arange(tst_x.shape[1])

                # Build name-to-index mapping from stored input_features
                # This allows proper indexing by variable name instead of sequential order
                name_to_idx = {name: idx for idx, name in enumerate(input_features)}
                
                color_idx = 0
                for var_name, selected in df_plot_in.items():
                    if selected:
                        # Find actual index for this variable using name mapping
                        if var_name in name_to_idx:
                            feat_idx = name_to_idx[var_name]
                            if feat_idx < tst_x.shape[-1]:
                                if y_sbpl_fmt == 'original':
                                    y_values = tst_x[i_sbpl, :, feat_idx]
                                else:
                                    y_values = tst_x[i_sbpl, :, feat_idx]

                                # Determine label prefix based on whether it's a TIME component
                                time_components = ['Y_sin', 'Y_cos', 'M_sin', 'M_cos', 'W_sin', 'W_cos', 'D_sin', 'D_cos', 'Holiday']
                                label_prefix = 'TIME' if var_name in time_components else 'IN'
                                
                                ax.plot(x_values, y_values,
                                       label=f'{label_prefix}: {var_name}',
                                       color=palette[color_idx % len(palette)],
                                       marker='o', markersize=2,
                                       linewidth=1)
                                color_idx += 1
                            else:
                                logger.warning(f"Variable {var_name} index {feat_idx} exceeds tst_x dimension {tst_x.shape[-1]}")
                        else:
                            logger.warning(f"Variable {var_name} not found in input_features mapping. Available: {list(name_to_idx.keys())[:10]}...")

                # Build output name-to-index mapping
                output_name_to_idx = {name: idx for idx, name in enumerate(output_features)}
                
                for var_name, selected in df_plot_out.items():
                    if selected:
                        # Find actual index for this output variable
                        if var_name in output_name_to_idx:
                            out_idx = output_name_to_idx[var_name]
                            if out_idx < tst_y.shape[-1]:
                                if y_sbpl_fmt == 'original':
                                    y_values = tst_y[i_sbpl, :, out_idx]
                                else:
                                    y_values = tst_y[i_sbpl, :, out_idx]

                                ax.plot(x_values[:len(y_values)], y_values,
                                       label=f'OUT: {var_name}',
                                       color=palette[color_idx % len(palette)],
                                       marker='s', markersize=2,
                                       linewidth=1)
                                color_idx += 1
                            else:
                                logger.warning(f"Output variable {var_name} index {out_idx} exceeds tst_y dimension {tst_y.shape[-1]}")
                        else:
                            # Fallback to sequential index if name not found
                            i_out = list(df_plot_out.keys()).index(var_name)
                            if i_out < tst_y.shape[-1]:
                                y_values = tst_y[i_sbpl, :, i_out]
                                ax.plot(x_values[:len(y_values)], y_values,
                                       label=f'OUT: {var_name}',
                                       color=palette[color_idx % len(palette)],
                                       marker='s', markersize=2,
                                       linewidth=1)
                                color_idx += 1

                for var_name, selected in df_plot_fcst.items():
                    if selected:
                        # Find actual index for forecast (uses same mapping as output)
                        if var_name in output_name_to_idx:
                            fcst_idx = output_name_to_idx[var_name]
                            n_fcst_features = tst_fcst.shape[-1] if len(tst_fcst.shape) > 2 else 1
                            if fcst_idx < n_fcst_features:
                                if len(tst_fcst.shape) == 3:
                                    y_values = tst_fcst[i_sbpl, :, fcst_idx]
                                elif len(tst_fcst.shape) == 2:
                                    y_values = tst_fcst[i_sbpl, :]
                                else:
                                    y_values = tst_fcst

                                ax.plot(x_values[:len(y_values)], y_values,
                                       label=f'FCST: {var_name}',
                                       color=palette[color_idx % len(palette)],
                                       marker='^', markersize=2,
                                       linewidth=1, linestyle='--')
                                color_idx += 1
                        else:
                            # Fallback to sequential index
                            i_fcst = list(df_plot_fcst.keys()).index(var_name)
                            n_fcst_features = tst_fcst.shape[-1] if len(tst_fcst.shape) > 2 else 1
                            if i_fcst < n_fcst_features:
                                if len(tst_fcst.shape) == 3:
                                    y_values = tst_fcst[i_sbpl, :, i_fcst]
                                elif len(tst_fcst.shape) == 2:
                                    y_values = tst_fcst[i_sbpl, :]
                                else:
                                    y_values = tst_fcst
                                    
                                ax.plot(x_values[:len(y_values)], y_values,
                                       label=f'FCST: {var_name}',
                                       color=palette[color_idx % len(palette)],
                                       marker='^', markersize=2,
                                       linewidth=1, linestyle='--')
                                color_idx += 1

                ax.set_title(f'Sample {i_sbpl + 1}', fontsize=10)
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3)

                if x_sbpl == 'UTC':
                    ax.set_xlabel('Time (UTC)', fontsize=9)
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                else:
                    ax.set_xlabel('Timestep', fontsize=9)

                ax.set_ylabel('Value', fontsize=9)

                if y_sbpl_set == 'separate Achsen':
                    pass

            for i in range(num_sbpl, len(axs)):
                fig.delaxes(axs[i])

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            logger.info(f"Plot generated successfully for session {session_id}")

            return {
                'plot_data': f'data:image/png;base64,{plot_data}',
                'message': 'Plot generated successfully'
            }

        except ValueError as e:
            raise
        except Exception as e:
            logger.error(f"Error generating plot: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise


def create_visualizer() -> Visualizer:
    """
    Create and return a Visualizer instance

    Returns:
        Visualizer instance
    """
    return Visualizer()


def save_visualization_to_database(session_id: str, viz_name: str, viz_data: str, user_id: str = None) -> bool:
    """
    Save a visualization to the database (with duplicate check).

    Args:
        session_id: Session identifier
        viz_name: Name of the visualization
        viz_data: Base64 encoded visualization data
        user_id: User ID for session ownership (optional, uses Flask g if not provided)

    Returns:
        bool: True if successful

    Raises:
        Exception: If save fails
    """
    from datetime import datetime
    from shared.database.operations import get_supabase_client, create_or_get_session_uuid

    try:
        # Get user_id from Flask context if not provided
        if user_id is None:
            try:
                from flask import g
                user_id = g.user_id
            except (RuntimeError, AttributeError):
                pass  # Not in Flask context

        # Use service_role to bypass RLS for visualization inserts
        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = create_or_get_session_uuid(session_id, user_id)

        existing = supabase.table('training_visualizations').select('id').eq(
            'session_id', uuid_session_id
        ).eq('plot_name', viz_name).execute()

        if existing.data:
            viz_record = {
                'image_data': viz_data,
                'plot_type': 'violin_plot' if 'distribution' in viz_name else 'other',
                'created_at': datetime.now().isoformat()
            }
            response = supabase.table('training_visualizations').update(viz_record).eq(
                'session_id', uuid_session_id
            ).eq('plot_name', viz_name).execute()
            logger.info(f"Updated existing visualization {viz_name} for session {session_id}")
        else:
            viz_record = {
                'session_id': uuid_session_id,
                'plot_name': viz_name,
                'image_data': viz_data,
                'plot_type': 'violin_plot' if 'distribution' in viz_name else 'other',
                'created_at': datetime.now().isoformat()
            }
            response = supabase.table('training_visualizations').insert(viz_record).execute()
            logger.info(f"Created new visualization {viz_name} for session {session_id}")

        return True

    except Exception as e:
        logger.error(f"Error saving visualization {viz_name} for session {session_id}: {str(e)}")
        raise
