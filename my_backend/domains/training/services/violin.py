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

BATCH PROCESSING: Each violin is created separately and combined at the end
to prevent OOM with large datasets (12 TIME features × millions of rows).
"""

import logging
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, List, Any, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont



from domains.training.constants import TIME_COMPONENT_NAMES, TIME_NAME_DISPLAY_MAP

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _combine_images_horizontally(images: List[Image.Image], title: str = None) -> Image.Image:
    """
    Combine multiple PIL images horizontally into one image.

    Args:
        images: List of PIL Image objects
        title: Optional title to add at top

    Returns:
        Combined PIL Image
    """
    if not images:
        return None

    # Calculate dimensions
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    total_width = sum(widths)
    max_height = max(heights)

    # Add space for title if needed
    title_height = 50 if title else 0

    # Create combined image
    combined = Image.new('RGB', (total_width, max_height + title_height), 'white')

    # Paste images
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, title_height))
        x_offset += img.width

    # Add title if provided
    if title:
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        draw.text(((total_width - text_width) // 2, 15), title, fill='black', font=font)

    return combined


def _create_single_violin(values: np.ndarray, title: str, color,
                          ylabel: str = "", feature_index: int = 0,
                          total_features: int = 1, plot_type: str = "unknown") -> Image.Image:
    """
    Create a single violin plot and return as PIL Image.
    Memory-efficient: closes figure immediately after saving.

    Args:
        values: Data array for the violin plot
        title: Title for this subplot
        color: Color from the palette
        ylabel: Y-axis label
        feature_index: Index of this feature (for logging)
        total_features: Total number of features (for logging)
        plot_type: Type of plot for logging (INPUT/TIME/OUTPUT)

    Returns:
        PIL Image of the single violin plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(2, 6))

    # Handle NaN
    if isinstance(values, pd.Series):
        is_all_nan = values.isna().all()
    else:
        is_all_nan = np.isnan(values).all() if len(values) > 0 else True

    if not is_all_nan:
        sns.violinplot(y=values, ax=ax, color=color, inner="quartile", linewidth=1.5)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)

    # Save to PIL Image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img = Image.open(buffer).copy()  # .copy() to detach from buffer

    # Cleanup immediately
    plt.close(fig)
    buffer.close()
    del fig, ax
    gc.collect()

    return img


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
    Generate violin plots for data distributions using BATCH PROCESSING.

    Creates 3 separate plots:
      1. Eingabedaten: INPUT features only
      2. Zeitmerkmale: TIME components only
      3. Ausgabedaten: OUTPUT features only

    BATCH PROCESSING: Each violin is created separately and combined at the end.
    This prevents OOM when processing 12 TIME features × millions of rows.
    Memory usage reduced by ~92% compared to creating all subplots at once.

    Shared palette: tab20 with n_input + n_time + n_output colors
    - Input colors: palette[i] for i in range(n_input)
    - TIME colors: palette[n_input + i] for i in range(n_time)
    - Output colors: palette[n_input + n_time + i] for i in range(n_output)

    Args:
        session_id: Session identifier
        input_features: List of tuples for inputs. Supports:
            - 2-tuple: (feature_name, values_array) - feature_name used as title, no y-label
            - 3-tuple: (bezeichnung, column_name, values_array) - bezeichnung as title, column_name as y-label
        time_features: List of (feature_name, values_array) tuples for TIME components
        output_features: Same format as input_features
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

        # ═══════════════════════════════════════════════════════════════════
        # INPUT PLOT - Input features only (BATCH PROCESSING)
        # Title: "Datenverteilung \nder Eingabedaten"
        # Colors: palette[0] to palette[n_input-1]
        # ═══════════════════════════════════════════════════════════════════
        if input_features and len(input_features) > 0:
            if progress_tracker:
                progress_tracker.generating_input_plot()

            individual_images = []
            for i, feature_tuple in enumerate(input_features):
                # Support both 2-tuple (name, values) and 3-tuple (bezeichnung, column_name, values)
                if len(feature_tuple) == 3:
                    bezeichnung, column_name, values = feature_tuple
                else:
                    bezeichnung, values = feature_tuple
                    column_name = ""

                img = _create_single_violin(
                    values=values,
                    title=bezeichnung,
                    color=palette[i],
                    ylabel=column_name,
                    feature_index=i,
                    total_features=len(input_features),
                    plot_type="INPUT"
                )
                individual_images.append(img)

                # Free the values array reference (original data still exists in caller)
                del values
                gc.collect()

            # Combine into final image
            combined = _combine_images_horizontally(
                individual_images,
                title="Datenverteilung der Eingabedaten"
            )

            # Convert to base64
            buffer = io.BytesIO()
            combined.save(buffer, format='PNG')
            buffer.seek(0)
            input_plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plots['input_violin_plot'] = {
                'data': f"data:image/png;base64,{input_plot_base64}",
                'type': 'input'
            }

            # Cleanup
            buffer.close()
            for img in individual_images:
                img.close()
            if combined:
                combined.close()
            del individual_images, combined
            gc.collect()

            if progress_tracker:
                progress_tracker.input_plot_complete()

        # ═══════════════════════════════════════════════════════════════════
        # TIME PLOT - TIME components only (BATCH PROCESSING)
        # Title: "Datenverteilung \nder Zeitmerkmale"
        # Colors: palette[n_input] to palette[n_input + n_time - 1]
        # ═══════════════════════════════════════════════════════════════════
        if time_features and len(time_features) > 0:
            if progress_tracker:
                progress_tracker.generating_time_plot()

            individual_images = []
            for i, (feature_name, values) in enumerate(time_features):
                # Color offset for TIME: n_input + i
                color_idx = n_input + i

                img = _create_single_violin(
                    values=values,
                    title=feature_name,
                    color=palette[color_idx],
                    ylabel="",
                    feature_index=i,
                    total_features=len(time_features),
                    plot_type="TIME"
                )
                individual_images.append(img)

                # Free the values array reference
                del values
                gc.collect()

            # Combine into final image
            combined = _combine_images_horizontally(
                individual_images,
                title="Datenverteilung der Zeitmerkmale"
            )

            # Convert to base64
            buffer = io.BytesIO()
            combined.save(buffer, format='PNG')
            buffer.seek(0)
            time_plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plots['time_violin_plot'] = {
                'data': f"data:image/png;base64,{time_plot_base64}",
                'type': 'time'
            }

            # Cleanup
            buffer.close()
            for img in individual_images:
                img.close()
            if combined:
                combined.close()
            del individual_images, combined
            gc.collect()

            if progress_tracker:
                progress_tracker.time_plot_complete()

        # ═══════════════════════════════════════════════════════════════════
        # OUTPUT PLOT - Output features only (BATCH PROCESSING)
        # Title: "Datenverteilung \nder Ausgabedaten"
        # Colors: palette[n_input + n_time] to palette[n_input + n_time + n_output - 1]
        # ═══════════════════════════════════════════════════════════════════
        if output_features and len(output_features) > 0:
            if progress_tracker:
                progress_tracker.generating_output_plot()

            individual_images = []
            for i, feature_tuple in enumerate(output_features):
                # Support both 2-tuple (name, values) and 3-tuple (bezeichnung, column_name, values)
                if len(feature_tuple) == 3:
                    bezeichnung, column_name, values = feature_tuple
                else:
                    bezeichnung, values = feature_tuple
                    column_name = ""

                # Color offset for output: n_input + n_time + i
                color_idx = n_input + n_time + i

                img = _create_single_violin(
                    values=values,
                    title=bezeichnung,
                    color=palette[color_idx],
                    ylabel=column_name,
                    feature_index=i,
                    total_features=len(output_features),
                    plot_type="OUTPUT"
                )
                individual_images.append(img)

                # Free the values array reference
                del values
                gc.collect()

            # Combine into final image
            combined = _combine_images_horizontally(
                individual_images,
                title="Datenverteilung der Ausgabedaten"
            )

            # Convert to base64
            buffer = io.BytesIO()
            combined.save(buffer, format='PNG')
            buffer.seek(0)
            output_plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plots['output_violin_plot'] = {
                'data': f"data:image/png;base64,{output_plot_base64}",
                'type': 'output'
            }

            # Cleanup
            buffer.close()
            for img in individual_images:
                img.close()
            if combined:
                combined.close()
            del individual_images, combined
            gc.collect()

            if progress_tracker:
                progress_tracker.output_plot_complete()

        return {
            'success': True,
            'plots': plots,
            'message': 'Violin plots generated successfully (3 plots: input, time, output) - BATCH PROCESSING'
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
