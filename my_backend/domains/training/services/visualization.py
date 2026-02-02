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
from domains.training.constants import (
    TIME_COMPONENT_NAMES,
    TIME_NAME_DISPLAY_MAP,
    TIMESTAMP_COLUMN_NAMES,
    get_active_time_components
)

logger = logging.getLogger(__name__)


def get_file_color_map(session_id: str) -> Dict[str, tuple]:
    """
    Get color mapping for all files in a session.

    This function provides consistent color assignments for files across
    violin plots and evaluation diagrams. Each file gets a unique color
    based on its color_index in the database.

    Color indexing logic:
    - Input files: indices 0, 1, 2, ...
    - Output files: input_count, input_count+1, ...
    - TIME components: input_count + output_count

    Args:
        session_id: Session identifier (string or UUID)

    Returns:
        Dict mapping bezeichnung to RGB color tuple.
        Example: {'load_grid': (0.12, 0.47, 0.71), 'temperature': (1.0, 0.5, 0.05)}
    """
    from shared.database.operations import get_supabase_client, create_or_get_session_uuid

    try:
        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = create_or_get_session_uuid(session_id, user_id=None)

        response = supabase.table('files').select(
            'bezeichnung', 'color_index'
        ).eq('session_id', uuid_session_id).execute()

        if not response.data:
            return {}

        # Determine palette size based on max color_index
        max_index = max((f.get('color_index', 0) for f in response.data), default=0)
        palette = sns.color_palette("tab20", max(max_index + 1, 20))

        return {
            f['bezeichnung']: palette[f.get('color_index', 0)]
            for f in response.data
        }

    except Exception as e:
        logger.error(f"Error getting file color map for session {session_id}: {e}")
        return {}


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

        DYNAMIC: All values are fetched from database, no hardcoded fallbacks.

        Args:
            session_id: Session identifier
            user_id: User ID for session lookup

        Returns:
            dict: {
                'input_variables': list (from files/training_results),
                'output_variables': list (from files/training_results),
                'time_components': list (from time_info table)
            }
        """
        try:
            from utils.training_storage import fetch_training_results_with_storage
            from shared.database.operations import get_supabase_client, create_or_get_session_uuid

            supabase = get_supabase_client()
            uuid_session_id = create_or_get_session_uuid(session_id, user_id=user_id)

            input_variables = []
            output_variables = []
            time_components = []

            # 1. Try to get from training_results first (most reliable after training)
            results = fetch_training_results_with_storage(session_id)

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

            # 2. Fallback: Get from files table (before training is complete)
            if not input_variables or not output_variables:
                file_response = supabase.table('files').select('*').eq('session_id', uuid_session_id).execute()

                if file_response.data:
                    for file_data in file_response.data:
                        file_type = file_data.get('type', '') or file_data.get('file_type', '')
                        datentyp = file_data.get('datentyp', '')
                        columns = file_data.get('columns', [])
                        bezeichnung = file_data.get('bezeichnung', '')

                        # Check both file_type and datentyp for compatibility
                        is_input = file_type == 'input' or datentyp in ['input', 'Eingabedaten']
                        is_output = file_type == 'output' or datentyp in ['output', 'Ausgabedaten']

                        # Use columns if available, otherwise use bezeichnung as the feature name
                        feature_names = columns if columns else [bezeichnung] if bezeichnung else []

                        if is_input:
                            feature_list = [
                                col for col in feature_names
                                if col.lower() not in TIMESTAMP_COLUMN_NAMES
                            ]
                            if feature_list:
                                input_variables.extend(feature_list)
                        elif is_output:
                            feature_list = [
                                col for col in feature_names
                                if col.lower() not in TIMESTAMP_COLUMN_NAMES
                            ]
                            if feature_list:
                                output_variables.extend(feature_list)

            # 3. Get TIME components from time_info table (DYNAMIC based on user configuration)
            time_info_response = supabase.table('time_info').select('*').eq('session_id', uuid_session_id).execute()

            if time_info_response.data:
                time_info = time_info_response.data[0]
                time_components = get_active_time_components(time_info)

            # 4. Separate regular inputs from TIME components (if TIME components are in input_variables)
            input_vars_list = input_variables if isinstance(input_variables, list) else []
            regular_inputs = [v for v in input_vars_list if v not in TIME_COMPONENT_NAMES]

            # Also check if TIME components are stored in input_variables
            stored_time_components = [v for v in input_vars_list if v in TIME_COMPONENT_NAMES]
            if stored_time_components and not time_components:
                time_components = stored_time_components

            return {
                'input_variables': regular_inputs,
                'output_variables': output_variables if isinstance(output_variables, list) else [],
                'time_components': time_components
            }

        except Exception as e:
            logger.error(f"Error getting plot variables for {session_id}: {str(e)}")
            # Return empty arrays on error - no hardcoded fallbacks
            return {
                'input_variables': [],
                'output_variables': [],
                'time_components': [],
                'error': str(e)
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

                # Try both column names for backward compatibility
                image_data = viz.get('image_data', '') or viz.get('plot_data_base64', '')

                # Get type from metadata.type (stored in JSONB column)
                viz_metadata = viz.get('metadata', {})
                if isinstance(viz_metadata, dict) and viz_metadata.get('type'):
                    plot_type = viz_metadata.get('type')
                else:
                    # Fallback for legacy plots: infer type from plot_name
                    if 'output' in plot_name.lower():
                        plot_type = 'output'
                    elif 'time' in plot_name.lower():
                        plot_type = 'time'
                    else:
                        plot_type = 'input'

                # Return new format with data and type
                plots[plot_name] = {
                    'data': image_data,
                    'type': plot_type
                }

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

        MATCHED TO ORIGINAL training.py (lines 2401-3309):
        - Random subplot selection
        - tst_inf dictionary structure with DataFrames
        - "separate Achsen" implementation with twinx()
        - Color palette indexing: input at i_feat, output at n_input + i_feat
        - Reference UTC marker (vertical dashed line)
        - FCST with marker='x', linestyle='--'
        - Subplot title with UTC timestamp

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
            import math
            import random
            import datetime
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

            results = fetch_training_results_with_storage(session_id, model_id=model_id)

            if not results:
                raise ValueError('Model not trained yet. Please train the model first.')

            model_data = results.get('trained_model')

            trained_model = None
            if model_data:
                if isinstance(model_data, list) and len(model_data) > 0:
                    model_data = model_data[0]

                # Deserialize model - supports both old JSON format and new pickle format
                from utils.serialization_helpers import deserialize_model_or_scaler
                trained_model = deserialize_model_or_scaler(model_data)
                if trained_model is None:
                    logger.error(f"Failed to deserialize trained model")

            test_data = results.get('test_data', {})
            metadata = results.get('metadata', {})
            scalers = results.get('scalers', {})

            # ===================================================================
            # GET FEATURE NAMES - PRIORITIZE STORED FEATURES OVER DATABASE
            # ===================================================================
            # The stored training results contain the actual features that were
            # used during training. The database might have different configuration
            # (e.g., TIME components added after training).
            from shared.database.operations import get_supabase_client, create_or_get_session_uuid

            supabase = get_supabase_client()
            uuid_session_id = create_or_get_session_uuid(session_id)

            # First try to get features from stored results (most reliable)
            stored_input_features = results.get('input_features', [])
            stored_output_features = results.get('output_features', [])

            # Get file names from files table (for fallback and reference)
            file_response = supabase.table('files').select('bezeichnung, type, color_index').eq('session_id', uuid_session_id).order('color_index').execute()

            input_file_names = []
            output_file_names = []

            if file_response.data:
                for f in file_response.data:
                    bezeichnung = f.get('bezeichnung', '')
                    file_type = f.get('type', '')
                    if file_type == 'input' and bezeichnung:
                        input_file_names.append(bezeichnung)
                    elif file_type == 'output' and bezeichnung:
                        output_file_names.append(bezeichnung)

            # Get TIME components from time_info table
            time_info_response = supabase.table('time_info').select('*').eq('session_id', uuid_session_id).execute()
            time_components_from_db = []
            if time_info_response.data:
                time_info = time_info_response.data[0]
                time_components_from_db = get_active_time_components(time_info)

            # Decide which features to use:
            # 1. If stored features are available and non-empty, use them
            # 2. Otherwise, build from database
            if stored_input_features:
                input_features = stored_input_features
                # Separate into files and time components for plotting keys
                input_file_names_actual = [f for f in input_features if f not in TIME_COMPONENT_NAMES]
                time_components = [f for f in input_features if f in TIME_COMPONENT_NAMES]
            else:
                # Build from database
                input_features = input_file_names + time_components_from_db
                input_file_names_actual = input_file_names
                time_components = time_components_from_db

            if stored_output_features:
                output_features = stored_output_features
            else:
                output_features = output_file_names

            # Update input_file_names with actual values for plotting
            input_file_names = input_file_names_actual

            # ===================================================================
            # LOAD TEST DATA - BOTH SCALED AND ORIGINAL (MATCHES ORIGINAL)
            # ===================================================================
            has_test_data = False
            tst_x_orig = None
            tst_y_orig = None

            if test_data:
                # Load scaled data
                if 'X' in test_data and 'y' in test_data:
                    tst_x = np.array(test_data.get('X'))
                    tst_y = np.array(test_data.get('y'))
                    has_test_data = True
                elif 'X_test' in test_data and 'y_test' in test_data:
                    tst_x = np.array(test_data.get('X_test'))
                    tst_y = np.array(test_data.get('y_test'))
                    has_test_data = True

                # Load original (unscaled) data - MATCHES ORIGINAL training.py
                if 'X_orig' in test_data:
                    tst_x_orig = np.array(test_data.get('X_orig'))
                else:
                    tst_x_orig = tst_x  # Fallback to scaled

                if 'y_orig' in test_data:
                    tst_y_orig = np.array(test_data.get('y_orig'))
                else:
                    tst_y_orig = tst_y  # Fallback to scaled

            if not has_test_data:
                logger.error(f"No test data found in database for session {session_id}")
                raise ValueError('No training data available for this session. Please train a model first before generating plots')

            # Get utc_ref_log from metadata - MATCHES ORIGINAL (line 2468)
            utc_ref_log = metadata.get('utc_ref_log', [])
            n_tst = tst_x.shape[0]

            # Convert string timestamps to datetime if needed
            utc_ref_log_tst = []
            if utc_ref_log:
                utc_ref_log_tst = utc_ref_log[-n_tst:] if len(utc_ref_log) >= n_tst else utc_ref_log
                # Convert strings to datetime objects
                for i, ts in enumerate(utc_ref_log_tst):
                    if isinstance(ts, str):
                        try:
                            utc_ref_log_tst[i] = pd.to_datetime(ts)
                        except Exception:
                            utc_ref_log_tst[i] = None

            # ===================================================================
            # VALIDATE AND ADJUST FEATURES TO MATCH ACTUAL DATA DIMENSIONS
            # ===================================================================
            actual_input_features = tst_x.shape[-1]
            actual_output_features = tst_y.shape[-1]

            # Validate input features match actual data dimensions
            if len(input_features) != actual_input_features:
                if len(input_features) > actual_input_features:
                    # Truncate to match array (TIME components might not be in stored data)
                    logger.warning(f"More feature names ({len(input_features)}) than array dimensions ({actual_input_features}). "
                                  f"Truncating to match stored data.")
                    input_features = input_features[:actual_input_features]
                    # Re-separate files and time components
                    input_file_names = [f for f in input_features if f not in TIME_COMPONENT_NAMES]
                    time_components = [f for f in input_features if f in TIME_COMPONENT_NAMES]
                else:
                    # Array has more features than we have names - ERROR
                    raise ValueError(
                        f"Input feature mismatch: Array has {actual_input_features} features but only "
                        f"{len(input_features)} feature names found ({input_features}). "
                        f"This indicates corrupted or incomplete training data."
                    )

            # Validate output features match actual data dimensions
            if len(output_features) != actual_output_features:
                if len(output_features) > actual_output_features:
                    logger.warning(f"More output names ({len(output_features)}) than array dimensions ({actual_output_features}). "
                                  f"Truncating to match stored data.")
                    output_features = output_features[:actual_output_features]
                else:
                    # Array has more features than we have names - ERROR
                    raise ValueError(
                        f"Output feature mismatch: Array has {actual_output_features} features but only "
                        f"{len(output_features)} feature names found ({output_features}). "
                        f"This indicates corrupted or incomplete training data."
                    )

            # Get time series parameters from metadata
            I_N = tst_x.shape[1]  # Input timesteps
            O_N = tst_y.shape[1]  # Output timesteps
            DELT = metadata.get('delt', 15)  # Default 15 minutes

            # ===================================================================
            # GENERATE PREDICTIONS
            # ===================================================================
            if trained_model and hasattr(trained_model, 'predict'):
                model_type = results.get('model_type', 'Unknown')

                if model_type in ['SVR_dir', 'SVR_MIMO', 'LIN']:
                    n_samples = tst_x.shape[0]

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
                    tst_fcst = trained_model.predict(tst_x)

                    if model_type == 'CNN':
                        tst_fcst = np.squeeze(tst_fcst, axis=-1)
            else:
                logger.error(f"Model is not available as an object for session {session_id}")
                raise ValueError('Model not available for predictions. Please retrain the model.')

            # Inverse transform forecast if scalers available - for "original" format
            tst_fcst_orig = tst_fcst.copy()
            if y_sbpl_fmt == 'original' and scalers:
                # Scalers are stored under 'output' key as dict - need to deserialize from base64 pickle
                o_scalers_raw = scalers.get('output', {})

                # Deserialize scalers - supports both old JSON and new pickle formats
                from utils.serialization_helpers import deserialize_scalers_dict
                o_scalers = deserialize_scalers_dict(o_scalers_raw)

                if o_scalers and len(tst_fcst.shape) == 3:
                    for i_feat in range(tst_fcst.shape[-1]):
                        has_scaler = i_feat in o_scalers and o_scalers[i_feat] is not None
                        if has_scaler:
                            try:
                                for i_sample in range(tst_fcst.shape[0]):
                                    tst_fcst_orig[i_sample, :, i_feat] = o_scalers[i_feat].inverse_transform(
                                        tst_fcst[i_sample, :, i_feat].reshape(-1, 1)
                                    ).ravel()
                            except Exception as e:
                                logger.warning(f"Could not inverse transform forecast feature {i_feat}: {e}")

            # ===================================================================
            # SUBPLOT SETUP - MATCHES ORIGINAL (lines 2432-2465)
            # ===================================================================

            # Calculate number of axes for "separate Achsen" mode
            n_ax = 0
            n_ax_l = 1
            n_ax_r = 0

            if y_sbpl_set == "separate Achsen":
                # Count selected input variables
                n_ax = sum(1 for v in df_plot_in.values() if v)
                # Count output/forecast (combined - same axis for both)
                for var_name in df_plot_out.keys():
                    if df_plot_out.get(var_name) or df_plot_fcst.get(var_name):
                        n_ax += 1

                n_ax_l = max(1, math.floor(n_ax / 2))
                n_ax_r = n_ax - n_ax_l

            # Limit subplots to available test data
            num_sbpl = min(num_sbpl, n_tst)

            # Calculate subplot grid - MATCHES ORIGINAL (lines 2448-2452)
            num_sbpl_x = math.ceil(math.sqrt(num_sbpl))
            num_sbpl_y = math.ceil(num_sbpl / num_sbpl_x)

            fig, axs = plt.subplots(num_sbpl_y, num_sbpl_x,
                                   figsize=(20, 13),
                                   layout='constrained')

            # Handle single subplot case
            if num_sbpl_y == 1 and num_sbpl_x == 1:
                axs = np.array([[axs]])
            elif num_sbpl_y == 1:
                axs = axs.reshape(1, -1)
            elif num_sbpl_x == 1:
                axs = axs.reshape(-1, 1)

            # Remove empty subplots - MATCHES ORIGINAL (lines 2459-2462)
            sbpl_del = num_sbpl_x * num_sbpl_y - num_sbpl
            for i in range(sbpl_del):
                axs[num_sbpl_y - 1, num_sbpl_x - 1 - i].axis('off')

            # Random selection of test datasets - MATCHES ORIGINAL (line 2465)
            tst_random = random.sample(range(n_tst), num_sbpl)

            # ===================================================================
            # BUILD tst_inf DICTIONARY - MATCHES ORIGINAL (lines 2475-2939)
            # ===================================================================
            tst_inf = {}

            for random_num in tst_random:
                # Get UTC reference for this sample
                utc_ref = None
                if utc_ref_log_tst and random_num < len(utc_ref_log_tst):
                    utc_ref = utc_ref_log_tst[random_num]

                tst_inf[random_num] = {"utc_ref": utc_ref}

                # Generate UTC timestamps for input data
                if utc_ref:
                    try:
                        utc_th_in = pd.date_range(
                            end=utc_ref,
                            periods=I_N,
                            freq=f'{DELT}min'
                        ).to_list()
                    except Exception:
                        utc_th_in = [utc_ref] * I_N
                else:
                    # Fallback: generate timestamps starting from now
                    utc_th_in = pd.date_range(
                        start=pd.Timestamp.now() - pd.Timedelta(minutes=DELT * I_N),
                        periods=I_N,
                        freq=f'{DELT}min'
                    ).to_list()

                # Generate UTC timestamps for output data
                if utc_ref:
                    try:
                        utc_th_out = pd.date_range(
                            start=utc_ref,
                            periods=O_N,
                            freq=f'{DELT}min'
                        ).to_list()
                    except Exception:
                        utc_th_out = [utc_ref] * O_N
                else:
                    utc_th_out = pd.date_range(
                        start=pd.Timestamp.now(),
                        periods=O_N,
                        freq=f'{DELT}min'
                    ).to_list()

                # INPUT DATA - MATCHES ORIGINAL (lines 2488-2530)
                for i_feat, feat_name in enumerate(input_file_names):
                    if i_feat < tst_x.shape[-1]:
                        if y_sbpl_fmt == "original":
                            value = tst_x_orig[random_num, :, i_feat]
                        else:
                            value = tst_x[random_num, :, i_feat]

                        df = pd.DataFrame({
                            "UTC": utc_th_in,
                            "ts": list(range(-I_N + 1, 1)),
                            "value": value
                        })
                        tst_inf[random_num]["IN: " + feat_name] = df

                # TIME COMPONENTS - MATCHES ORIGINAL (lines 2585-2897)
                # Convert lowercase TIME names to UPPERCASE for display (Y_sin, Y_cos, etc.)
                time_start_idx = len(input_file_names)
                for i_time, time_name in enumerate(time_components):
                    time_idx = time_start_idx + i_time
                    if time_idx < tst_x.shape[-1]:
                        if y_sbpl_fmt == "original":
                            value = tst_x_orig[random_num, :, time_idx]
                        else:
                            value = tst_x[random_num, :, time_idx]

                        df = pd.DataFrame({
                            "UTC": utc_th_in,
                            "ts": list(range(-I_N + 1, 1)),
                            "value": value
                        })
                        # Use UPPERCASE display name (Y_sin, Y_cos) - MATCHES ORIGINAL
                        display_name = TIME_NAME_DISPLAY_MAP.get(time_name, time_name)
                        tst_inf[random_num]["TIME: " + display_name] = df

                # OUTPUT DATA - MATCHES ORIGINAL (lines 2538-2580)
                for i_feat, feat_name in enumerate(output_features):
                    if i_feat < tst_y.shape[-1]:
                        if y_sbpl_fmt == "original":
                            value = tst_y_orig[random_num, :, i_feat]
                        else:
                            value = tst_y[random_num, :, i_feat]

                        df = pd.DataFrame({
                            "UTC": utc_th_out,
                            "ts": list(range(0, O_N)),
                            "value": value
                        })
                        tst_inf[random_num]["OUT: " + feat_name] = df

                # FORECAST DATA - MATCHES ORIGINAL (lines 2899-2939)
                for i_feat, feat_name in enumerate(output_features):
                    n_fcst_features = tst_fcst.shape[-1] if len(tst_fcst.shape) > 2 else 1
                    if i_feat < n_fcst_features:
                        if len(tst_fcst.shape) == 3:
                            if y_sbpl_fmt == "original":
                                value = tst_fcst_orig[random_num, :, i_feat]
                            else:
                                value = tst_fcst[random_num, :, i_feat]
                        else:
                            value = tst_fcst[random_num, :]

                        df = pd.DataFrame({
                            "UTC": utc_th_out,
                            "ts": list(range(0, O_N)),
                            "value": value
                        })
                        tst_inf[random_num]["FCST: " + feat_name] = df

            # ===================================================================
            # COLOR PALETTE - MATCHES ORIGINAL (line 2973, 3068, 3163)
            # ===================================================================
            n_total_features = len(input_features) + len(output_features)
            palette = sns.color_palette("tab20", max(20, n_total_features * 2))

            # ===================================================================
            # PLOT FILLING - MATCHES ORIGINAL (lines 2946-3269)
            # ===================================================================

            # Collect lines and labels for legend (only from first subplot)
            all_lines = []
            all_labels = []

            for i_sbpl in range(num_sbpl):
                # Row and column of current subplot - MATCHES ORIGINAL (lines 2949-2952)
                i_y_sbpl = math.floor(i_sbpl / num_sbpl_x)
                i_x_sbpl = i_sbpl - i_y_sbpl * num_sbpl_x

                # Get the key for this subplot's data
                key_1 = list(tst_inf.keys())[i_sbpl]

                # Main axis of subplot - MATCHES ORIGINAL (lines 2957-2959)
                ax_sbpl_orig = axs[i_y_sbpl, i_x_sbpl]
                ax_sbpl = [ax_sbpl_orig]

                # Counters for axis positioning - MATCHES ORIGINAL (lines 2961-2964)
                i_line = 0
                i_ax_l = 0
                i_ax_r = 0

                # Build name-to-index mapping
                name_to_idx = {name: idx for idx, name in enumerate(input_features)}
                output_name_to_idx = {name: idx for idx, name in enumerate(output_features)}

                # ===============================================================
                # PLOT INPUT DATA AND TIME COMPONENTS - MATCHES ORIGINAL (lines 2967-3059)
                # ===============================================================
                i_feat_plot = 0
                for var_name, selected in df_plot_in.items():
                    if selected:
                        # Determine key prefix - MATCHES ORIGINAL (lines 2976-2979)
                        if var_name in input_file_names:
                            key_2 = "IN: " + var_name
                            i_feat = input_file_names.index(var_name)
                        elif var_name in time_components:
                            # Use UPPERCASE display name for TIME components (Y_sin, Y_cos, etc.)
                            display_name = TIME_NAME_DISPLAY_MAP.get(var_name, var_name)
                            key_2 = "TIME: " + display_name
                            i_feat = len(input_file_names) + time_components.index(var_name)
                        else:
                            continue

                        if key_2 not in tst_inf[key_1]:
                            logger.warning(f"Key {key_2} not found in tst_inf for sample {key_1}")
                            continue

                        # Color - MATCHES ORIGINAL (line 2973): palette[i_feat]
                        color_plt = palette[i_feat]

                        # Get x and y values - MATCHES ORIGINAL (lines 2983-2988)
                        if x_sbpl == "UTC":
                            x_value = tst_inf[key_1][key_2]["UTC"]
                        else:
                            x_value = tst_inf[key_1][key_2]["ts"]

                        y_value = tst_inf[key_1][key_2]["value"]

                        if y_sbpl_set == "gemeinsame Achse":
                            # Single shared axis - MATCHES ORIGINAL (lines 2990-2999)
                            line, = ax_sbpl_orig.plot(
                                x_value, y_value,
                                label=key_2 if i_sbpl == 0 else None,
                                color=color_plt,
                                marker='o',
                                linewidth=1,
                                markersize=2
                            )
                            if i_sbpl == 0:
                                all_lines.append(line)
                                all_labels.append(key_2)

                        elif y_sbpl_set == "separate Achsen":
                            # Create new axis if not first line - MATCHES ORIGINAL (line 3004)
                            if i_line > 0:
                                ax_sbpl.append(ax_sbpl_orig.twinx())

                            # Plot - MATCHES ORIGINAL (lines 3006-3013)
                            line, = ax_sbpl[-1].plot(
                                x_value, y_value,
                                label=key_2 if i_sbpl == 0 else None,
                                color=color_plt,
                                marker='o',
                                linewidth=1,
                                markersize=2
                            )
                            if i_sbpl == 0:
                                all_lines.append(line)
                                all_labels.append(key_2)

                            # Position axis - MATCHES ORIGINAL (lines 3015-3056)
                            if i_line < n_ax_l:
                                pos = 'left'
                                i_ax = i_ax_l
                                i_ax_l += 1
                            else:
                                pos = 'right'
                                i_ax = i_ax_r
                                i_ax_r += 1

                            # Move axis line outward
                            ax_sbpl[-1].spines[pos].set_position(('outward', i_ax * 30))

                            # Hide top and opposite spine
                            for spine in ['top', 'left' if pos == 'right' else 'right']:
                                ax_sbpl[-1].spines[spine].set_visible(False)

                            # Y-axis tick position
                            ax_sbpl[-1].yaxis.set_ticks_position(pos)

                            # Color the axis
                            ax_sbpl[-1].spines[pos].set_color(color_plt)
                            ax_sbpl[-1].tick_params(
                                axis='y',
                                direction='inout',
                                colors=color_plt,
                                labelcolor=color_plt,
                                labelsize=8
                            )

                            # X-axis configuration
                            ax_sbpl[0].tick_params(axis="x", labelsize=8)
                            plt.setp(ax_sbpl[0].get_xticklabels(), rotation=45)

                            i_line += 1

                        i_feat_plot += 1

                # ===============================================================
                # PLOT OUTPUT DATA - MATCHES ORIGINAL (lines 3061-3153)
                # ===============================================================
                for var_name, selected in df_plot_out.items():
                    if selected:
                        key_2 = "OUT: " + var_name

                        if key_2 not in tst_inf[key_1]:
                            logger.warning(f"Key {key_2} not found in tst_inf for sample {key_1}")
                            continue

                        if var_name in output_name_to_idx:
                            i_feat = output_name_to_idx[var_name]
                        else:
                            continue

                        # Color - MATCHES ORIGINAL (line 3068): palette[n_input + i_feat]
                        color_plt = palette[len(input_features) + i_feat]

                        if x_sbpl == "UTC":
                            x_value = tst_inf[key_1][key_2]["UTC"]
                        else:
                            x_value = tst_inf[key_1][key_2]["ts"]

                        y_value = tst_inf[key_1][key_2]["value"]

                        if y_sbpl_set == "gemeinsame Achse":
                            line, = ax_sbpl_orig.plot(
                                x_value, y_value,
                                label=key_2 if i_sbpl == 0 else None,
                                color=color_plt,
                                marker='o',
                                linewidth=1,
                                markersize=2
                            )
                            if i_sbpl == 0:
                                all_lines.append(line)
                                all_labels.append(key_2)

                        elif y_sbpl_set == "separate Achsen":
                            if i_line > 0:
                                ax_sbpl.append(ax_sbpl_orig.twinx())

                            line, = ax_sbpl[-1].plot(
                                x_value, y_value,
                                label=key_2 if i_sbpl == 0 else None,
                                color=color_plt,
                                marker='o',
                                linewidth=1,
                                markersize=2
                            )
                            if i_sbpl == 0:
                                all_lines.append(line)
                                all_labels.append(key_2)

                            if i_line < n_ax_l:
                                pos = 'left'
                                i_ax = i_ax_l
                                i_ax_l += 1
                            else:
                                pos = 'right'
                                i_ax = i_ax_r
                                i_ax_r += 1

                            ax_sbpl[-1].spines[pos].set_position(('outward', i_ax * 30))
                            for spine in ['top', 'left' if pos == 'right' else 'right']:
                                ax_sbpl[-1].spines[spine].set_visible(False)
                            ax_sbpl[-1].yaxis.set_ticks_position(pos)
                            ax_sbpl[-1].spines[pos].set_color(color_plt)
                            ax_sbpl[-1].tick_params(
                                axis='y', direction='inout',
                                colors=color_plt, labelcolor=color_plt, labelsize=8
                            )
                            ax_sbpl[0].tick_params(axis="x", labelsize=8)
                            plt.setp(ax_sbpl[0].get_xticklabels(), rotation=45)

                            i_line += 1

                # ===============================================================
                # PLOT FORECAST DATA - MATCHES ORIGINAL (lines 3155-3255)
                # ===============================================================
                for var_name, selected in df_plot_fcst.items():
                    if selected:
                        key_2 = "FCST: " + var_name

                        if key_2 not in tst_inf[key_1]:
                            logger.warning(f"Key {key_2} not found in tst_inf for sample {key_1}")
                            continue

                        if var_name in output_name_to_idx:
                            i_feat = output_name_to_idx[var_name]
                        else:
                            continue

                        # Color - MATCHES ORIGINAL (line 3163): same as output
                        color_plt = palette[len(input_features) + i_feat]

                        if x_sbpl == "UTC":
                            x_value = tst_inf[key_1][key_2]["UTC"]
                        else:
                            x_value = tst_inf[key_1][key_2]["ts"]

                        y_value = tst_inf[key_1][key_2]["value"]

                        if y_sbpl_set == "gemeinsame Achse":
                            # FCST with marker='x' and linestyle='--' - MATCHES ORIGINAL (lines 3180-3187)
                            line, = ax_sbpl_orig.plot(
                                x_value, y_value,
                                label=key_2 if i_sbpl == 0 else None,
                                color=color_plt,
                                marker='x',
                                linestyle='--',
                                linewidth=1,
                                markersize=4
                            )
                            if i_sbpl == 0:
                                all_lines.append(line)
                                all_labels.append(key_2)

                        elif y_sbpl_set == "separate Achsen":
                            # Check if output for same variable is already plotted
                            out_selected = df_plot_out.get(var_name, False)

                            if not out_selected:
                                # Create new axis only if not the first line
                                # First line uses ax_sbpl_orig directly
                                if i_line > 0:
                                    ax_sbpl.append(ax_sbpl_orig.twinx())
                                i_pos = len(ax_sbpl) - 1
                            else:
                                # Use same axis as output - MATCHES ORIGINAL (lines 3192-3196)
                                n_in_selected = sum(1 for v in df_plot_in.values() if v)
                                n_out_before = sum(1 for k, v in list(df_plot_out.items())[:list(df_plot_out.keys()).index(var_name) + 1] if v)
                                i_pos = n_in_selected + n_out_before - 1
                                if i_pos >= len(ax_sbpl):
                                    i_pos = len(ax_sbpl) - 1

                            # FCST plot - MATCHES ORIGINAL (lines 3200-3207)
                            line, = ax_sbpl[i_pos].plot(
                                x_value, y_value,
                                label=key_2 if i_sbpl == 0 else None,
                                color=color_plt,
                                marker='x',
                                linestyle='--',
                                linewidth=1,
                                markersize=4
                            )
                            if i_sbpl == 0:
                                all_lines.append(line)
                                all_labels.append(key_2)

                            # Configure axis only if new axis was created
                            if not out_selected:
                                if i_line < n_ax_l:
                                    pos = 'left'
                                    i_ax = i_ax_l
                                    i_ax_l += 1
                                else:
                                    pos = 'right'
                                    i_ax = i_ax_r
                                    i_ax_r += 1

                                ax_sbpl[-1].spines[pos].set_position(('outward', i_ax * 30))
                                for spine in ['top', 'left' if pos == 'right' else 'right']:
                                    ax_sbpl[-1].spines[spine].set_visible(False)
                                ax_sbpl[-1].yaxis.set_ticks_position(pos)
                                ax_sbpl[-1].spines[pos].set_color(color_plt)
                                ax_sbpl[-1].tick_params(
                                    axis='y', direction='inout',
                                    colors=color_plt, labelcolor=color_plt, labelsize=8
                                )
                                ax_sbpl[0].tick_params(axis="x", labelsize=8)
                                plt.setp(ax_sbpl[0].get_xticklabels(), rotation=45)

                                i_line += 1

                # ===============================================================
                # VERTICAL REFERENCE LINE - MATCHES ORIGINAL (lines 3257-3267)
                # ===============================================================
                if x_sbpl == "UTC" and tst_inf[key_1]["utc_ref"] is not None:
                    ax_sbpl_orig.axvline(
                        x=tst_inf[key_1]["utc_ref"],
                        color="black",
                        linestyle='--',
                        label=None
                    )
                else:
                    ax_sbpl_orig.axvline(
                        x=0,
                        color="black",
                        linestyle='--',
                        label=None
                    )

                # ===============================================================
                # SUBPLOT TITLE - MATCHES ORIGINAL (line 3269)
                # ===============================================================
                if tst_inf[key_1]["utc_ref"] is not None:
                    try:
                        title = "UTC: " + tst_inf[key_1]["utc_ref"].strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        title = f"Sample {key_1 + 1}"
                else:
                    title = f"Sample {key_1 + 1}"

                ax_sbpl_orig.set_title(title, fontsize=10)

            # ===================================================================
            # LEGEND - MATCHES ORIGINAL (lines 3290-3297)
            # ===================================================================
            # Remove duplicate labels from child axes
            unique_labels = []
            unique_lines = []
            for line, label in zip(all_lines, all_labels):
                if label and not label.startswith('_') and label not in unique_labels:
                    unique_labels.append(label)
                    unique_lines.append(line)

            if unique_lines:
                fig.legend(
                    unique_lines, unique_labels,
                    loc="upper right",
                    ncol=5,
                    fontsize=8
                )

            # ===================================================================
            # TITLE - MATCHES ORIGINAL (lines 3299-3303)
            # ===================================================================
            plt.suptitle(
                "Auswertung der Testdatensätze",
                fontsize=20,
                fontweight='bold'
            )

            # ===================================================================
            # SAVE PLOT
            # ===================================================================
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


def delete_old_violin_plots(session_id: str, user_id: str = None) -> int:
    """
    Delete all existing violin plots for a session before saving new ones.

    This is necessary because the new violin plot structure has different names
    than the old structure (input_violin_plot vs "Test1 Violin Plot" per file).

    Args:
        session_id: Session identifier
        user_id: User ID for session ownership

    Returns:
        int: Number of deleted plots
    """
    from shared.database.operations import get_supabase_client, create_or_get_session_uuid

    try:
        # Get user_id from Flask context if not provided
        if user_id is None:
            try:
                from flask import g
                user_id = g.user_id
            except (RuntimeError, AttributeError):
                pass

        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = create_or_get_session_uuid(session_id, user_id)

        # Get count of existing violin plots
        existing = supabase.table('training_visualizations').select('id, plot_name').eq(
            'session_id', uuid_session_id
        ).eq('plot_type', 'violin_plot').execute()

        if existing.data:
            plot_names = [p['plot_name'] for p in existing.data]
            logger.info(f"Deleting {len(existing.data)} old violin plots for session {session_id}: {plot_names}")

            # Delete all violin plots for this session
            supabase.table('training_visualizations').delete().eq(
                'session_id', uuid_session_id
            ).eq('plot_type', 'violin_plot').execute()

            return len(existing.data)

        return 0

    except Exception as e:
        logger.error(f"Error deleting old violin plots for session {session_id}: {str(e)}")
        return 0


def save_visualization_to_database(session_id: str, viz_name: str, viz_data, user_id: str = None) -> bool:
    """
    Save a visualization to the database (with duplicate check).

    Args:
        session_id: Session identifier
        viz_name: Name of the visualization
        viz_data: Base64 encoded visualization data (string) or dict with {data, type}
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

        # Handle new format (dict with data and type) and legacy format (string)
        if isinstance(viz_data, dict) and 'data' in viz_data:
            image_data = viz_data.get('data', '')
            plot_type_category = viz_data.get('type', 'input')  # 'input' | 'output' | 'time'
        else:
            image_data = viz_data
            plot_type_category = 'input'  # Default for legacy

        # Use service_role to bypass RLS for visualization inserts
        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = create_or_get_session_uuid(session_id, user_id)

        existing = supabase.table('training_visualizations').select('id').eq(
            'session_id', uuid_session_id
        ).eq('plot_name', viz_name).execute()

        if existing.data:
            viz_record = {
                'image_data': image_data,
                'plot_type': 'violin_plot' if 'violin' in viz_name else 'other',
                'metadata': {'type': plot_type_category},
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
                'image_data': image_data,
                'plot_type': 'violin_plot' if 'violin' in viz_name else 'other',
                'metadata': {'type': plot_type_category},
                'created_at': datetime.now().isoformat()
            }
            response = supabase.table('training_visualizations').insert(viz_record).execute()
            logger.info(f"Created new visualization {viz_name} for session {session_id}")

        return True

    except Exception as e:
        logger.error(f"Error saving visualization {viz_name} for session {session_id}: {str(e)}")
        raise
