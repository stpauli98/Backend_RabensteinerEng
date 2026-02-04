"""
Violin plot generator for data visualization without model training.

Generates 3 separate plots:
  - INPUT features (Eingabedaten)
  - TIME components (Zeitmerkmale)
  - OUTPUT features (Ausgabedaten)

Uses shared color palette: tab20 with total features count
- Input colors: palette[0] to palette[n_input-1]
- TIME colors: palette[n_input] to palette[n_input+n_time-1]
- Output colors: palette[n_input+n_time] to palette[n_input+n_time+n_output-1]

TIME component names are UPPERCASE (Y_sin, Y_cos, M_sin, etc.)

Single matplotlib figure per plot group - matches original training.py approach:
  plt.subplots(1, n, figsize=(2*n, 6))
"""

import logging
import gc
import time as time_module
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, List, Any, Optional, Tuple, Union


from domains.training.constants import TIME_COMPONENT_NAMES, TIME_NAME_DISPLAY_MAP

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-FIGURE RENDERING (matches original training.py)
# ═══════════════════════════════════════════════════════════════════════════════

MAX_VIOLIN_POINTS = 50_000  # Subsample limit for KDE - preserves distribution shape


def _create_violin_plot_group(
    features: List,
    title: str,
    palette_colors: List,
) -> Optional[str]:
    """
    Create a violin plot group using single matplotlib figure.
    Matches original training.py approach: plt.subplots(1, n, figsize=(2*n, 6))

    Args:
        features: List of tuples. Supports:
            - 2-tuple: (name, values_array) - name as title, no y-label
            - 3-tuple: (bezeichnung, column_name, values_array) - bezeichnung as title, column_name as y-label
        title: Suptitle for the figure (unused, kept for API compat)
        palette_colors: List of colors for each violin

    Returns:
        base64-encoded PNG data URI string, or None if no features
    """
    n = len(features)
    if n == 0:
        return None

    t_start = time_module.time()

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 6))

    # Handle single subplot case (axes is not array)
    if n == 1:
        axes = [axes]

    for i, feature_tuple in enumerate(features):
        ax = axes[i]

        # Support both 2-tuple (name, values) and 3-tuple (bezeichnung, column_name, values)
        if len(feature_tuple) == 3:
            name, ylabel, values = feature_tuple
        else:
            name, values = feature_tuple
            ylabel = ""

        # Remove NaN values before subsampling
        valid_values = values[~np.isnan(values)] if len(values) > 0 else values

        # Subsample to prevent OOM - KDE doesn't need millions of points
        if len(valid_values) > MAX_VIOLIN_POINTS:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(valid_values), MAX_VIOLIN_POINTS, replace=False)
            valid_values = valid_values[indices]

        is_empty = len(valid_values) == 0

        if not is_empty:
            sns.violinplot(y=valid_values, ax=ax, color=palette_colors[i],
                          inner="quartile", linewidth=1.5)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes)

        ax.set_title(name)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)

    plt.tight_layout()

    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    elapsed = time_module.time() - t_start
    logger.debug(f"VIOLIN RENDER:   Plot '{title}' complete in {elapsed:.2f}s, "
                 f"base64 size: {len(plot_base64)} chars")

    # Cleanup
    plt.close(fig)
    buffer.close()
    del fig
    gc.collect()

    return f"data:image/png;base64,{plot_base64}"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_violin_plots_from_viz_data(
    session_id: str,
    data_arrays: Dict,
    input_feature_names: List[str] = None,
    output_feature_names: List[str] = None
) -> Dict:
    """
    Adapter function to maintain compatibility with visualization.py interface.
    Converts data_arrays format to generate_violin_plots_from_data format.

    This is called from middleman.py after training - it receives combined arrays
    where input features and TIME are already merged in i_combined_array.

    Args:
        session_id: Session identifier
        data_arrays: Dict containing 'i_combined_array' and 'o_combined_array'
        input_feature_names: List of input feature names (file names + TIME components)
        output_feature_names: List of output feature names (file names only)

    Returns:
        Dictionary containing violin plots (same format as visualization.py)
    """
    try:
        input_data = data_arrays.get('i_combined_array')
        output_data = data_arrays.get('o_combined_array')

        # Convert lowercase TIME names to UPPERCASE for display (matches original)
        def get_display_name(name: str) -> str:
            """Convert lowercase TIME names to UPPERCASE display format."""
            return TIME_NAME_DISPLAY_MAP.get(name, name)

        # Separate input file features from TIME components
        input_features_list = []
        time_features_list = []

        if input_data is not None and len(input_data.shape) > 1:
            if input_feature_names and len(input_feature_names) == input_data.shape[1]:
                for i, name in enumerate(input_feature_names):
                    display_name = get_display_name(name)
                    # Check if this is a TIME component
                    if name in TIME_COMPONENT_NAMES or name in TIME_NAME_DISPLAY_MAP.values():
                        time_features_list.append((display_name, input_data[:, i]))
                    else:
                        input_features_list.append((display_name, input_data[:, i]))
            else:
                # Fallback to generic names - all go to input
                input_features_list = [
                    (f"Input_Feature_{i+1}", input_data[:, i])
                    for i in range(input_data.shape[1])
                ]
                logger.warning(f"Using generic input names. Provided: {len(input_feature_names) if input_feature_names else 0}, Expected: {input_data.shape[1]}")

        # Build output features list
        output_features_list = []
        if output_data is not None and len(output_data.shape) > 1:
            if output_feature_names and len(output_feature_names) == output_data.shape[1]:
                output_features_list = [
                    (output_feature_names[i], output_data[:, i])
                    for i in range(output_data.shape[1])
                ]
            else:
                output_features_list = [
                    (f"Output_Feature_{i+1}", output_data[:, i])
                    for i in range(output_data.shape[1])
                ]
                logger.warning(f"Using generic output names. Provided: {len(output_feature_names) if output_feature_names else 0}, Expected: {output_data.shape[1]}")

        result = generate_violin_plots_from_data(
            session_id=session_id,
            input_features=input_features_list if input_features_list else None,
            time_features=time_features_list if time_features_list else None,
            output_features=output_features_list if output_features_list else None
        )

        return result.get('plots', {})

    except Exception as e:
        logger.error(f"Error in create_violin_plots_from_viz_data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def generate_violin_plots_from_data(
    session_id: str,
    input_features: Optional[List[Tuple[str, np.ndarray]]] = None,
    time_features: Optional[List[Tuple[str, np.ndarray]]] = None,
    output_features: Optional[List[Tuple[str, np.ndarray]]] = None,
    progress_tracker=None
) -> Dict[str, Any]:
    """
    Generate violin plots for data distributions using single matplotlib figure per group.
    Matches original training.py approach: plt.subplots(1, n, figsize=(2*n, 6))

    Creates 3 separate plots:
      1. Eingabedaten: INPUT features only
      2. Zeitmerkmale: TIME components only
      3. Ausgabedaten: OUTPUT features only

    Shared palette: tab20 with n_input + n_time + n_output colors

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

        logger.debug("=" * 60)
        logger.debug("VIOLIN RENDER: generate_violin_plots_from_data() START")
        logger.debug(f"VIOLIN RENDER: Palette size: {total_features} "
                     f"(input={n_input}, time={n_time}, output={n_output})")
        logger.debug("=" * 60)

        # ═══════════════════════════════════════════════════════════════════
        # INPUT PLOT
        # ═══════════════════════════════════════════════════════════════════
        if input_features and len(input_features) > 0:
            if progress_tracker:
                progress_tracker.generating_input_plot()

            input_colors = [palette[i] for i in range(n_input)]
            input_base64 = _create_violin_plot_group(
                input_features,
                "Datenverteilung \nder Eingabedaten",
                input_colors
            )
            if input_base64:
                plots['input_violin_plot'] = {'data': input_base64, 'type': 'input'}

            if progress_tracker:
                progress_tracker.input_plot_complete()

        # ═══════════════════════════════════════════════════════════════════
        # TIME PLOT
        # ═══════════════════════════════════════════════════════════════════
        if time_features and len(time_features) > 0:
            if progress_tracker:
                progress_tracker.generating_time_plot()

            time_colors = [palette[n_input + i] for i in range(n_time)]
            time_base64 = _create_violin_plot_group(
                time_features,
                "Datenverteilung \nder Zeitmerkmale",
                time_colors
            )
            if time_base64:
                plots['time_violin_plot'] = {'data': time_base64, 'type': 'time'}

            if progress_tracker:
                progress_tracker.time_plot_complete()

        # ═══════════════════════════════════════════════════════════════════
        # OUTPUT PLOT
        # ═══════════════════════════════════════════════════════════════════
        if output_features and len(output_features) > 0:
            if progress_tracker:
                progress_tracker.generating_output_plot()

            output_colors = [palette[n_input + n_time + i] for i in range(n_output)]
            output_base64 = _create_violin_plot_group(
                output_features,
                "Datenverteilung \nder Ausgabedaten",
                output_colors
            )
            if output_base64:
                plots['output_violin_plot'] = {'data': output_base64, 'type': 'output'}

            if progress_tracker:
                progress_tracker.output_plot_complete()

        logger.debug(f"VIOLIN RENDER: Generated {len(plots)} plots: {list(plots.keys())}")
        logger.debug("VIOLIN RENDER: generate_violin_plots_from_data() DONE")

        return {
            'success': True,
            'plots': plots,
            'message': 'Violin plots generated successfully (3 plots: input, time, output)'
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
