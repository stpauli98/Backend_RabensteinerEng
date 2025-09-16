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
        # Extract data arrays
        input_data = data_arrays.get('i_combined_array')
        output_data = data_arrays.get('o_combined_array')
        
        # Generate feature names if data exists
        input_features = None
        if input_data is not None and len(input_data.shape) > 1:
            input_features = [f"Input_Feature_{i+1}" for i in range(input_data.shape[1])]
        
        output_features = None 
        if output_data is not None and len(output_data.shape) > 1:
            output_features = [f"Output_Feature_{i+1}" for i in range(output_data.shape[1])]
        
        # Generate plots using the unified function
        result = generate_violin_plots_from_data(
            session_id=session_id,
            input_data=input_data,
            output_data=output_data,
            input_features=input_features,
            output_features=output_features
        )
        
        # Return just the plots dict (to match visualization.py interface)
        return result.get('plots', {})
        
    except Exception as e:
        logger.error(f"Error in create_violin_plots_from_viz_data: {str(e)}")
        return {}

def generate_violin_plots_from_data(
    session_id: str,
    input_data: Optional[np.ndarray] = None,
    output_data: Optional[np.ndarray] = None,
    input_features: Optional[List[str]] = None,
    output_features: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate violin plots for input and output data distributions.
    
    This function creates violin plots WITHOUT training any models,
    following the original implementation approach where plots show
    raw data distribution.
    
    Args:
        session_id: Session identifier
        input_data: Input data array (X data)
        output_data: Output data array (y data)
        input_features: Names of input features
        output_features: Names of output features
    
    Returns:
        Dictionary containing base64-encoded plot images
    """
    try:
        plots = {}
        
        # Color palette for plots
        palette = sns.color_palette("husl", 20)
        
        # Generate input data violin plot
        if input_data is not None and len(input_data) > 0:
            logger.info(f"Generating input data violin plot for session {session_id}")
            
            # Convert to DataFrame for easier manipulation
            if input_features:
                df_input = pd.DataFrame(input_data, columns=input_features)
            else:
                # Generate default feature names if not provided
                n_features = input_data.shape[1] if len(input_data.shape) > 1 else 1
                input_features = [f"Feature_{i+1}" for i in range(n_features)]
                df_input = pd.DataFrame(input_data, columns=input_features)
            
            # Create violin plot for input data
            n_ft_i = len(input_features)
            fig_width = max(2 * n_ft_i, 6)
            fig, axes = plt.subplots(1, n_ft_i, figsize=(fig_width, 6))
            
            # Handle single feature case
            if n_ft_i == 1:
                axes = [axes]
            
            # Create violin plot for each feature
            for i, feature in enumerate(input_features):
                values = df_input[feature]
                
                # Skip if all values are NaN
                if not values.isna().all():
                    sns.violinplot(
                        y=values,
                        ax=axes[i],
                        color=palette[i % len(palette)],
                        inner="quartile",
                        linewidth=1.5
                    )
                    
                    # Set title and clean up axes
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
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            input_plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plots['input_violin_plot'] = f"data:image/png;base64,{input_plot_base64}"
            plt.close()
            
            logger.info(f"Input violin plot generated successfully")
        
        # Generate output data violin plot
        if output_data is not None and len(output_data) > 0:
            logger.info(f"Generating output data violin plot for session {session_id}")
            
            # Convert to DataFrame
            if output_features:
                df_output = pd.DataFrame(output_data, columns=output_features)
            else:
                # Generate default feature names if not provided
                n_features = output_data.shape[1] if len(output_data.shape) > 1 else 1
                output_features = [f"Output_{i+1}" for i in range(n_features)]
                df_output = pd.DataFrame(output_data, columns=output_features)
            
            # Create violin plot for output data
            n_ft_o = len(output_features)
            fig_width = max(2 * n_ft_o, 6)
            fig, axes = plt.subplots(1, n_ft_o, figsize=(fig_width, 6))
            
            # Handle single feature case
            if n_ft_o == 1:
                axes = [axes]
            
            # Create violin plot for each feature
            for i, feature in enumerate(output_features):
                values = df_output[feature]
                
                # Skip if all values are NaN
                if not values.isna().all():
                    # Use different colors for output features
                    color_idx = (i + len(input_features) if input_features else i) % len(palette)
                    
                    sns.violinplot(
                        y=values,
                        ax=axes[i],
                        color=palette[color_idx],
                        inner="quartile",
                        linewidth=1.5
                    )
                    
                    # Set title and clean up axes
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
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            output_plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plots['output_violin_plot'] = f"data:image/png;base64,{output_plot_base64}"
            plt.close()
            
            logger.info(f"Output violin plot generated successfully")
        
        return {
            'success': True,
            'plots': plots,
            'message': 'Violin plots generated successfully'
        }
        
    except Exception as e:
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
        from utils.database import get_supabase_client, create_or_get_session_uuid
        
        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)
        
        # Get latest training results to extract data
        response = supabase.table('training_results').select('*').eq(
            'session_id', uuid_session_id
        ).order('created_at.desc').limit(1).execute()
        
        if not response.data or len(response.data) == 0:
            # No training results available yet
            return {
                'success': False,
                'error': 'No data found for session',
                'message': 'Please train a model first to generate violin plots'
            }
        
        # Extract data from training results
        result = response.data[0]
        data_splits = result.get('data_splits', {})
        
        # Get feature names from model metadata
        model_metadata = result.get('model_metadata', {})
        input_features = model_metadata.get('input_features', [])
        output_features = model_metadata.get('output_features', [])
        
        # Combine all data for visualization (train + val + test)
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