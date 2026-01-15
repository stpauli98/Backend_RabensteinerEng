"""
Violin plot generator for data visualization without model training.

Generates 3 separate plots:
- ONE plot for INPUT features only (Eingabedaten)
- ONE plot for TIME components only (Zeitkomponenten)
- ONE plot for OUTPUT features only (Ausgabedaten)
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def create_violin_plots_from_viz_data(session_id: str, data_arrays: Dict) -> Dict:
    """
    Adapter function to maintain compatibility with visualization.py interface.
    Converts data_arrays format to generate_violin_plots_from_data format.

    Args:
        session_id: Session identifier
        data_arrays: Dict containing 'i_combined_array' and 'o_combined_array'

    Returns:
        Dictionary containing violin plots (same format as visualization.py)
    """
    try:
        input_data = data_arrays.get('i_combined_array')
        output_data = data_arrays.get('o_combined_array')

        # Convert to new format: List[Tuple[str, np.ndarray]]
        input_features_list = None
        if input_data is not None and len(input_data.shape) > 1:
            input_features_list = [
                (f"Input_Feature_{i+1}", input_data[:, i])
                for i in range(input_data.shape[1])
            ]

        output_features_list = None
        if output_data is not None and len(output_data.shape) > 1:
            output_features_list = [
                (f"Output_Feature_{i+1}", output_data[:, i])
                for i in range(output_data.shape[1])
            ]

        result = generate_violin_plots_from_data(
            session_id=session_id,
            input_features=input_features_list,
            output_features=output_features_list
        )

        return result.get('plots', {})

    except Exception as e:
        logger.error(f"Error in create_violin_plots_from_viz_data: {str(e)}")
        return {}


def generate_violin_plots_from_data(
    session_id: str,
    input_features: Optional[List[Tuple[str, np.ndarray]]] = None,
    time_features: Optional[List[Tuple[str, np.ndarray]]] = None,
    output_features: Optional[List[Tuple[str, np.ndarray]]] = None,
    progress_tracker=None
) -> Dict[str, Any]:
    """
    Generate violin plots for data distributions.

    Creates 3 separate plots:
    - ONE plot for INPUT features only (Eingabedaten)
    - ONE plot for TIME components only (Zeitkomponenten)
    - ONE plot for OUTPUT features only (Ausgabedaten)

    Each FEATURE gets its own subplot and color from a shared palette.

    Args:
        session_id: Session identifier
        input_features: List of (feature_name, values_array) tuples for inputs
        time_features: List of (feature_name, values_array) tuples for TIME components
        output_features: List of (feature_name, values_array) tuples for outputs
        progress_tracker: Optional ViolinProgressTracker for emitting progress updates

    Returns:
        Dictionary containing base64-encoded plot images
    """
    try:
        plots = {}

        # Calculate total features for shared palette
        n_input = len(input_features) if input_features else 0
        n_time = len(time_features) if time_features else 0
        n_output = len(output_features) if output_features else 0
        total_features = n_input + n_time + n_output
        palette = sns.color_palette("tab20", max(total_features, 20))

        logger.info(f"Generating violin plots: {n_input} input, {n_time} time, {n_output} output features")

        # ═══════════════════════════════════════════════════════════════════
        # INPUT PLOT - One plot for INPUT features only
        # ═══════════════════════════════════════════════════════════════════
        if input_features and len(input_features) > 0:
            if progress_tracker:
                progress_tracker.generating_input_plot()

            n_ft = len(input_features)
            fig_width = max(2 * n_ft, 4)
            fig, axes = plt.subplots(1, n_ft, figsize=(fig_width, 6))

            if n_ft == 1:
                axes = [axes]

            for i, (feature_name, values) in enumerate(input_features):
                if isinstance(values, pd.Series):
                    is_all_nan = values.isna().all()
                else:
                    is_all_nan = np.isnan(values).all() if len(values) > 0 else True

                if not is_all_nan:
                    sns.violinplot(
                        y=values,
                        ax=axes[i],
                        color=palette[i],
                        inner="quartile",
                        linewidth=1.5
                    )
                    axes[i].set_title(feature_name)
                    axes[i].set_xlabel("")
                    axes[i].set_ylabel("")
                else:
                    axes[i].text(0.5, 0.5, 'No data',
                               ha='center', va='center',
                               transform=axes[i].transAxes)
                    axes[i].set_title(feature_name)

            plt.suptitle("Datenverteilung \nder Eingabedaten",
                        fontsize=15, fontweight="bold")
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            input_plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plots['input_violin_plot'] = {
                'data': f"data:image/png;base64,{input_plot_base64}",
                'type': 'input'
            }
            plt.close()

            if progress_tracker:
                progress_tracker.input_plot_complete()

            logger.info(f"Input violin plot generated: {n_ft} features")

        # ═══════════════════════════════════════════════════════════════════
        # TIME PLOT - One plot for TIME components only
        # ═══════════════════════════════════════════════════════════════════
        if time_features and len(time_features) > 0:
            if progress_tracker:
                progress_tracker.generating_time_plot()

            n_ft = len(time_features)
            fig_width = max(2 * n_ft, 4)
            fig, axes = plt.subplots(1, n_ft, figsize=(fig_width, 6))

            if n_ft == 1:
                axes = [axes]

            for i, (feature_name, values) in enumerate(time_features):
                if isinstance(values, pd.Series):
                    is_all_nan = values.isna().all()
                else:
                    is_all_nan = np.isnan(values).all() if len(values) > 0 else True

                if not is_all_nan:
                    # Offset for time colors
                    color_idx = i + n_input

                    sns.violinplot(
                        y=values,
                        ax=axes[i],
                        color=palette[color_idx],
                        inner="quartile",
                        linewidth=1.5
                    )
                    axes[i].set_title(feature_name)
                    axes[i].set_xlabel("")
                    axes[i].set_ylabel("")
                else:
                    axes[i].text(0.5, 0.5, 'No data',
                               ha='center', va='center',
                               transform=axes[i].transAxes)
                    axes[i].set_title(feature_name)

            plt.suptitle("Datenverteilung \nder Zeitkomponenten",
                        fontsize=15, fontweight="bold")
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            time_plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plots['time_violin_plot'] = {
                'data': f"data:image/png;base64,{time_plot_base64}",
                'type': 'time'
            }
            plt.close()

            if progress_tracker:
                progress_tracker.time_plot_complete()

            logger.info(f"Time violin plot generated: {n_ft} features")

        # ═══════════════════════════════════════════════════════════════════
        # OUTPUT PLOT - One plot for OUTPUT features only
        # ═══════════════════════════════════════════════════════════════════
        if output_features and len(output_features) > 0:
            if progress_tracker:
                progress_tracker.generating_output_plot()

            n_ft = len(output_features)
            fig_width = max(2 * n_ft, 4)
            fig, axes = plt.subplots(1, n_ft, figsize=(fig_width, 6))

            if n_ft == 1:
                axes = [axes]

            for i, (feature_name, values) in enumerate(output_features):
                if isinstance(values, pd.Series):
                    is_all_nan = values.isna().all()
                else:
                    is_all_nan = np.isnan(values).all() if len(values) > 0 else True

                if not is_all_nan:
                    # Offset for output colors (after input + time)
                    color_idx = i + n_input + n_time

                    sns.violinplot(
                        y=values,
                        ax=axes[i],
                        color=palette[color_idx],
                        inner="quartile",
                        linewidth=1.5
                    )
                    axes[i].set_title(feature_name)
                    axes[i].set_xlabel("")
                    axes[i].set_ylabel("")
                else:
                    axes[i].text(0.5, 0.5, 'No data',
                               ha='center', va='center',
                               transform=axes[i].transAxes)
                    axes[i].set_title(feature_name)

            plt.suptitle("Datenverteilung \nder Ausgabedaten",
                        fontsize=15, fontweight="bold")
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            output_plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plots['output_violin_plot'] = {
                'data': f"data:image/png;base64,{output_plot_base64}",
                'type': 'output'
            }
            plt.close()

            if progress_tracker:
                progress_tracker.output_plot_complete()

            logger.info(f"Output violin plot generated: {n_ft} features")

        return {
            'success': True,
            'plots': plots,
            'message': 'Violin plots generated successfully'
        }

    except Exception as e:
        if progress_tracker:
            progress_tracker.error(str(e))
        logger.error(f"Error generating violin plots: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e),
            'plots': {}
        }


def get_data_for_violin_plots(session_id: str) -> Dict[str, Any]:
    """
    Retrieve processed data from database for violin plot generation.

    Args:
        session_id: Session identifier

    Returns:
        Dictionary containing input/output data and feature names
    """
    try:
        from shared.database.operations import get_supabase_client, create_or_get_session_uuid

        supabase = get_supabase_client()
        # Note: This function should receive user_id from caller for proper validation
        # For now, uses None for backward compatibility (to be fixed in caller chain)
        uuid_session_id = create_or_get_session_uuid(session_id, user_id=None)

        response = supabase.table('training_results').select('*').eq(
            'session_id', uuid_session_id
        ).order('created_at.desc').limit(1).execute()

        if not response.data or len(response.data) == 0:
            return {
                'success': False,
                'error': 'No data found for session',
                'message': 'Please train a model first to generate violin plots'
            }

        result = response.data[0]
        data_splits = result.get('data_splits', {})

        model_metadata = result.get('model_metadata', {})
        input_feature_names = model_metadata.get('input_features', [])
        output_feature_names = model_metadata.get('output_features', [])

        input_data = []
        output_data = []

        for split in ['train', 'val', 'test']:
            if split in data_splits:
                split_data = data_splits[split]
                if 'X' in split_data and split_data['X']:
                    input_data.extend(split_data['X'])
                if 'y' in split_data and split_data['y']:
                    output_data.extend(split_data['y'])

        # Convert to new format: List[Tuple[str, np.ndarray]]
        input_features = None
        if input_data and input_feature_names:
            input_arr = np.array(input_data)
            input_features = [
                (name, input_arr[:, i])
                for i, name in enumerate(input_feature_names)
                if i < input_arr.shape[1]
            ]

        output_features = None
        if output_data and output_feature_names:
            output_arr = np.array(output_data)
            output_features = [
                (name, output_arr[:, i])
                for i, name in enumerate(output_feature_names)
                if i < output_arr.shape[1]
            ]

        return {
            'success': True,
            'input_features': input_features,
            'output_features': output_features
        }

    except Exception as e:
        logger.error(f"Error getting data for violin plots: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
