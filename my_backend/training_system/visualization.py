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

from .config import PLOT_SETTINGS

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
            
            # Create violin plots
            violin_plots = self.create_violin_plots(evaluation_results)
            visualizations.update(violin_plots)
            
            # Create forecast plots
            forecast_plots = self.create_forecast_plots(training_results, evaluation_results)
            visualizations.update(forecast_plots)
            
            # Create metrics comparison plots
            comparison_plots = self.create_metrics_comparison_plots(evaluation_results)
            visualizations.update(comparison_plots)
            
            # Create training history plots
            history_plots = self.create_training_history_plots(training_results)
            visualizations.update(history_plots)
            
            self.plots = visualizations
            return visualizations
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise
    
    def create_violin_plots(self, evaluation_results: Dict) -> Dict:
        """
        Create violin plots for error distribution
        Extracted from training_backend_test_2.py around lines 1874-2032
        
        Args:
            evaluation_results: Evaluation results
            
        Returns:
            Dict containing violin plots as base64 strings
        """
        try:
            violin_plots = {}
            
            for dataset_name, dataset_results in evaluation_results.get('evaluation_metrics', {}).items():
                # Prepare data for violin plot
                plot_data = []
                
                for model_name, metrics in dataset_results.items():
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            plot_data.append({
                                'Model': model_name,
                                'Metric': metric_name,
                                'Value': metric_value
                            })
                
                if plot_data:
                    df_plot = pd.DataFrame(plot_data)
                    
                    # Create violin plot for each metric
                    metrics_to_plot = ['mae', 'mse', 'rmse', 'mape']
                    
                    for metric in metrics_to_plot:
                        if metric in df_plot['Metric'].values:
                            fig, ax = plt.subplots(figsize=PLOT_SETTINGS['figure_size'])
                            
                            metric_data = df_plot[df_plot['Metric'] == metric]
                            
                            # TODO: Extract actual violin plot logic from training_backend_test_2.py
                            # This is placeholder implementation
                            
                            sns.violinplot(data=metric_data, x='Model', y='Value', ax=ax)
                            ax.set_title(f'{metric.upper()} Distribution - {dataset_name}')
                            ax.set_xlabel('Model')
                            ax.set_ylabel(f'{metric.upper()} Value')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            
                            # Convert to base64
                            plot_key = f'violin_{dataset_name}_{metric}'
                            violin_plots[plot_key] = self._figure_to_base64(fig)
                            
                            plt.close(fig)
            
            return violin_plots
            
        except Exception as e:
            logger.error(f"Error creating violin plots: {str(e)}")
            raise
    
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
                # Get evaluation DataFrame if available
                eval_dataframes = evaluation_results.get('evaluation_dataframes', {}).get(dataset_name, {})
                df_eval_ts = eval_dataframes.get('df_eval_ts', [])
                
                if df_eval_ts:
                    df_ts = pd.DataFrame(df_eval_ts)
                    
                    # Create forecast plot for each model
                    models = df_ts['model'].unique() if 'model' in df_ts.columns else []
                    
                    for model_name in models:
                        model_data = df_ts[df_ts['model'] == model_name]
                        
                        if len(model_data) > 0:
                            fig, ax = plt.subplots(figsize=PLOT_SETTINGS['figure_size'])
                            
                            # TODO: Extract actual forecast plot logic from training_backend_test_2.py
                            # This is placeholder implementation
                            
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
                            
                            # Convert to base64
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
                # Prepare data for comparison
                models = list(dataset_results.keys())
                metrics = ['mae', 'mse', 'rmse', 'mape']
                
                if models:
                    # Create comparison bar plot
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    axes = axes.flatten()
                    
                    for i, metric in enumerate(metrics):
                        if i < len(axes):
                            ax = axes[i]
                            
                            # Collect metric values
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
                                
                                # Add value labels on bars
                                for bar, value in zip(bars, metric_values):
                                    height = bar.get_height()
                                    ax.text(bar.get_x() + bar.get_width()/2., height,
                                           f'{value:.4f}', ha='center', va='bottom')
                    
                    plt.suptitle(f'Metrics Comparison - {dataset_name}', fontsize=16)
                    plt.tight_layout()
                    
                    # Convert to base64
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
                            
                            # Plot training loss
                            axes[0].plot(history['loss'], label='Training Loss')
                            if 'val_loss' in history:
                                axes[0].plot(history['val_loss'], label='Validation Loss')
                            axes[0].set_title(f'Training Loss - {model_name}')
                            axes[0].set_xlabel('Epoch')
                            axes[0].set_ylabel('Loss')
                            axes[0].legend()
                            axes[0].grid(True, alpha=0.3)
                            
                            # Plot training metrics
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
                            
                            # Convert to base64
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
                            
                            # Residual plot
                            axes[0].scatter(predicted, residuals, alpha=0.6)
                            axes[0].axhline(y=0, color='r', linestyle='--')
                            axes[0].set_xlabel('Predicted Values')
                            axes[0].set_ylabel('Residuals')
                            axes[0].set_title(f'Residual Plot - {model_name}')
                            axes[0].grid(True, alpha=0.3)
                            
                            # Q-Q plot
                            from scipy import stats
                            stats.probplot(residuals, dist="norm", plot=axes[1])
                            axes[1].set_title(f'Q-Q Plot - {model_name}')
                            axes[1].grid(True, alpha=0.3)
                            
                            plt.suptitle(f'Residual Analysis - {model_name} - {dataset_name}', fontsize=16)
                            plt.tight_layout()
                            
                            # Convert to base64
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
            
            # TODO: Implement saving plots to Supabase storage
            # This is placeholder implementation
            
            for plot_name, plot_data in self.plots.items():
                # Extract base64 data
                if plot_data.startswith('data:image/png;base64,'):
                    base64_data = plot_data.split(',')[1]
                    image_data = base64.b64decode(base64_data)
                    
                    # Save to storage
                    file_path = f"plots/{session_id}/{plot_name}.png"
                    
                    try:
                        response = supabase_client.storage.from_('visualizations').upload(
                            file_path, image_data, {'content-type': 'image/png'}
                        )
                        
                        if response:
                            logger.info(f"Saved plot {plot_name} to storage")
                    except Exception as e:
                        logger.error(f"Error saving plot {plot_name}: {str(e)}")
                        continue
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving plots to storage: {str(e)}")
            return False


# Factory function to create visualizer
def create_visualizer() -> Visualizer:
    """
    Create and return a Visualizer instance
    
    Returns:
        Visualizer instance
    """
    return Visualizer()