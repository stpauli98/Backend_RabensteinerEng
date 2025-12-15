"""
Violin plot generator for data visualization without model training.
Based on the original implementation approach.
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
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def create_violin_plots_from_viz_data(session_id: str, data_arrays: Dict) -> Dict:
    """
    Adapter function to maintain compatibility with visualization.py interface
    Converts data_arrays format to generate_violin_plots_from_data format
    
    Args:
        session_id: Session identifier
        data_arrays: Dict containing 'i_combined_array' and 'o_combined_array'
        
    Returns:
        Dictionary containing violin plots (same format as visualization.py)
    """
    try:
        input_data = data_arrays.get('i_combined_array')
        output_data = data_arrays.get('o_combined_array')
        
        input_features = None
        if input_data is not None and len(input_data.shape) > 1:
            input_features = [f"Input_Feature_{i+1}" for i in range(input_data.shape[1])]
        
        output_features = None 
        if output_data is not None and len(output_data.shape) > 1:
            output_features = [f"Output_Feature_{i+1}" for i in range(output_data.shape[1])]
        
        result = generate_violin_plots_from_data(
            session_id=session_id,
            input_data=input_data,
            output_data=output_data,
            input_features=input_features,
            output_features=output_features
        )
        
        return result.get('plots', {})
        
    except Exception as e:
        logger.error(f"Error in create_violin_plots_from_viz_data: {str(e)}")
        return {}

def generate_violin_plots_from_data(
    session_id: str,
    input_data: Optional[np.ndarray] = None,
    output_data: Optional[np.ndarray] = None,
    input_features: Optional[List[str]] = None,
    output_features: Optional[List[str]] = None,
    files_data: Optional[List[Dict]] = None,
    progress_tracker=None
) -> Dict[str, Any]:
    """
    Generate violin plots for data distributions.

    This function creates violin plots WITHOUT training any models,
    following the original implementation approach where plots show
    raw data distribution.

    Args:
        session_id: Session identifier
        input_data: Input data array (X data) - legacy parameter
        output_data: Output data array (y data) - legacy parameter
        input_features: Names of input features - legacy parameter
        output_features: Names of output features - legacy parameter
        files_data: List of file data dicts with keys: bezeichnung, data, features, type
                   New parameter - if provided, generates one plot per file/bezeichnung
        progress_tracker: Optional ViolinProgressTracker for emitting progress updates

    Returns:
        Dictionary containing base64-encoded plot images
    """
    try:
        plots = {}
        palette = sns.color_palette("husl", 20)

        # NEW: If files_data is provided, generate one plot per bezeichnung
        if files_data is not None and len(files_data) > 0:
            logger.info(f"Generating violin plots for {len(files_data)} files")
            
            for idx, file_data in enumerate(files_data):
                bezeichnung = file_data.get('bezeichnung', f'file_{idx}')
                data = file_data.get('data')
                features = file_data.get('features', [])
                file_type = file_data.get('type', 'unknown')
                
                if data is None or len(data) == 0:
                    logger.warning(f"No data for {bezeichnung}, skipping")
                    continue
                
                # Emit progress
                if progress_tracker:
                    if file_type == 'input':
                        progress_tracker.generating_input_plot()
                    else:
                        progress_tracker.generating_output_plot()
                
                logger.info(f"Generating violin plot for '{bezeichnung}' ({file_type})")
                
                # Create DataFrame
                if features:
                    df = pd.DataFrame(data, columns=features)
                else:
                    n_features = data.shape[1] if len(data.shape) > 1 else 1
                    features = [f"Feature_{i+1}" for i in range(n_features)]
                    df = pd.DataFrame(data, columns=features)
                
                n_ft = len(features)
                fig_width = max(2 * n_ft, 6)
                fig, axes = plt.subplots(1, n_ft, figsize=(fig_width, 6))
                
                if n_ft == 1:
                    axes = [axes]
                
                for i, feature in enumerate(features):
                    values = df[feature]
                    
                    if not values.isna().all():
                        sns.violinplot(
                            y=values,
                            ax=axes[i],
                            color=palette[(idx * 3 + i) % len(palette)],
                            inner="quartile",
                            linewidth=1.5
                        )
                        axes[i].set_title(feature)
                        axes[i].set_xlabel("")
                        axes[i].set_ylabel("")
                    else:
                        axes[i].text(0.5, 0.5, 'No data', 
                                   ha='center', va='center', 
                                   transform=axes[i].transAxes)
                        axes[i].set_title(feature)
                
                # Determine type label
                if file_type == 'time':
                    type_label = "Zeit"
                elif file_type == 'input':
                    type_label = "Input"
                else:
                    type_label = "Output"
                plt.suptitle(f"{bezeichnung} ({type_label}) - Data Distribution", fontsize=15, fontweight="bold")
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                
                # Use bezeichnung as plot key
                plot_key = f"{bezeichnung}_violin_plot"
                plots[plot_key] = f"data:image/png;base64,{plot_base64}"
                plt.close()
                
                # Emit progress complete
                if progress_tracker:
                    if file_type == 'input':
                        progress_tracker.input_plot_complete()
                    else:
                        progress_tracker.output_plot_complete()
                
                logger.info(f"Violin plot for '{bezeichnung}' generated successfully")
            
            return {
                'success': True,
                'plots': plots,
                'message': f'Violin plots generated successfully for {len(files_data)} files'
            }

        # LEGACY: Original behavior for backward compatibility
        if input_data is not None and len(input_data) > 0:
            # Emit progress: generating input plot
            if progress_tracker:
                progress_tracker.generating_input_plot()

            logger.info(f"Generating input data violin plot for session {session_id}")
            
            if input_features:
                df_input = pd.DataFrame(input_data, columns=input_features)
            else:
                n_features = input_data.shape[1] if len(input_data.shape) > 1 else 1
                input_features = [f"Feature_{i+1}" for i in range(n_features)]
                df_input = pd.DataFrame(input_data, columns=input_features)
            
            n_ft_i = len(input_features)
            fig_width = max(2 * n_ft_i, 6)
            fig, axes = plt.subplots(1, n_ft_i, figsize=(fig_width, 6))
            
            if n_ft_i == 1:
                axes = [axes]
            
            for i, feature in enumerate(input_features):
                values = df_input[feature]
                
                if not values.isna().all():
                    sns.violinplot(
                        y=values,
                        ax=axes[i],
                        color=palette[i % len(palette)],
                        inner="quartile",
                        linewidth=1.5
                    )
                    
                    axes[i].set_title(feature)
                    axes[i].set_xlabel("")
                    axes[i].set_ylabel("")
                else:
                    axes[i].text(0.5, 0.5, 'No data', 
                               ha='center', va='center', 
                               transform=axes[i].transAxes)
                    axes[i].set_title(feature)
            
            plt.suptitle("Input Data Distribution", fontsize=15, fontweight="bold")
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            input_plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plots['input_violin_plot'] = f"data:image/png;base64,{input_plot_base64}"
            plt.close()

            # Emit progress: input plot complete
            if progress_tracker:
                progress_tracker.input_plot_complete()

            logger.info(f"Input violin plot generated successfully")

        if output_data is not None and len(output_data) > 0:
            # Emit progress: generating output plot
            if progress_tracker:
                progress_tracker.generating_output_plot()

            logger.info(f"Generating output data violin plot for session {session_id}")
            
            if output_features:
                df_output = pd.DataFrame(output_data, columns=output_features)
            else:
                n_features = output_data.shape[1] if len(output_data.shape) > 1 else 1
                output_features = [f"Output_{i+1}" for i in range(n_features)]
                df_output = pd.DataFrame(output_data, columns=output_features)
            
            n_ft_o = len(output_features)
            fig_width = max(2 * n_ft_o, 6)
            fig, axes = plt.subplots(1, n_ft_o, figsize=(fig_width, 6))
            
            if n_ft_o == 1:
                axes = [axes]
            
            for i, feature in enumerate(output_features):
                values = df_output[feature]
                
                if not values.isna().all():
                    color_idx = (i + len(input_features) if input_features else i) % len(palette)
                    
                    sns.violinplot(
                        y=values,
                        ax=axes[i],
                        color=palette[color_idx],
                        inner="quartile",
                        linewidth=1.5
                    )
                    
                    axes[i].set_title(feature)
                    axes[i].set_xlabel("")
                    axes[i].set_ylabel("")
                else:
                    axes[i].text(0.5, 0.5, 'No data', 
                               ha='center', va='center', 
                               transform=axes[i].transAxes)
                    axes[i].set_title(feature)
            
            plt.suptitle("Output Data Distribution", fontsize=15, fontweight="bold")
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            output_plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plots['output_violin_plot'] = f"data:image/png;base64,{output_plot_base64}"
            plt.close()

            # Emit progress: output plot complete
            if progress_tracker:
                progress_tracker.output_plot_complete()

            logger.info(f"Output violin plot generated successfully")

        return {
            'success': True,
            'plots': plots,
            'message': 'Violin plots generated successfully'
        }

    except Exception as e:
        # Emit error if tracker available
        if progress_tracker:
            progress_tracker.error(str(e))
        logger.error(f"Error generating violin plots: {str(e)}")
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
        input_features = model_metadata.get('input_features', [])
        output_features = model_metadata.get('output_features', [])
        
        input_data = []
        output_data = []
        
        for split in ['train', 'val', 'test']:
            if split in data_splits:
                split_data = data_splits[split]
                if 'X' in split_data and split_data['X']:
                    input_data.extend(split_data['X'])
                if 'y' in split_data and split_data['y']:
                    output_data.extend(split_data['y'])
        
        return {
            'success': True,
            'input_data': np.array(input_data) if input_data else None,
            'output_data': np.array(output_data) if output_data else None,
            'input_features': input_features,
            'output_features': output_features
        }
        
    except Exception as e:
        logger.error(f"Error getting data for violin plots: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
