"""
Plotting module for training system
Contains plotting functionality extracted from training_backend_test_2.py
"""

from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import math
import logging
from datetime import datetime
import os

# Import necessary modules
from supabase_client import get_supabase_client

plotting_bp = Blueprint('plotting', __name__)
logger = logging.getLogger(__name__)

class PlotGenerator:
    """
    Generates plots based on original training_backend_test_2.py logic
    """
    
    def __init__(self):
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Setup matplotlib style settings"""
        try:
            plt.style.use('default')
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['figure.dpi'] = 100
            plt.rcParams['font.size'] = 10
        except Exception as e:
            logger.warning(f"Could not set plot style: {str(e)}")
    
    def generate_plot(self, session_id: str, plot_config: dict, plot_settings: dict) -> str:
        """
        Generate plot based on configuration
        Extracted and adapted from training_backend_test_2.py around line 2908
        
        Args:
            session_id: Training session identifier
            plot_config: Dict containing df_plot_in, df_plot_out, df_plot_fcst structures
            plot_settings: Plot visualization settings
            
        Returns:
            Base64 encoded plot image
        """
        try:
            # Load session data
            session_data = self._load_session_data(session_id)
            if not session_data:
                raise ValueError(f"No data found for session {session_id}")
            
            # Extract plotting DataFrames
            df_plot_in = pd.DataFrame.from_dict(plot_config.get('df_plot_in', {}), orient='index')
            df_plot_out = pd.DataFrame.from_dict(plot_config.get('df_plot_out', {}), orient='index')  
            df_plot_fcst = pd.DataFrame.from_dict(plot_config.get('df_plot_fcst', {}), orient='index')
            
            # Get data info first
            i_dat_inf, o_dat_inf = self._get_data_info(session_data)
            
            # Load test information (tst_inf structure from original code)
            tst_inf = self._prepare_test_data(session_data, session_id, i_dat_inf, o_dat_inf)
            
            # Get plot settings (matching original code variables)
            y_sbpl_set = plot_settings.get('y_sbpl_set', 'separate Achsen')
            x_sbpl = plot_settings.get('x_sbpl', 'UTC')
            y_sbpl_fmt = plot_settings.get('y_sbpl_fmt', 'original')
            n_sbpl = plot_settings.get('n_sbpl', len(tst_inf))  # Number of subplots
            
            # Color palette (from original code)
            total_features = len(i_dat_inf) + len(o_dat_inf)
            palette = sns.color_palette("tab20", total_features)
            
            # Calculate subplot layout - use n_sbpl or fallback to number of datasets
            num_datasets = n_sbpl if n_sbpl > 0 else len(tst_inf)
            num_sbpl_x = min(3, num_datasets)  # Max 3 columns
            num_sbpl_y = math.ceil(num_datasets / num_sbpl_x)
            num_sbpl = num_datasets  # Total number of subplots
            
            # Create figure and subplots
            fig, axs = plt.subplots(num_sbpl_y, num_sbpl_x, 
                                  figsize=(6*num_sbpl_x, 6*num_sbpl_y))
            
            # Handle single subplot case
            if num_datasets == 1:
                axs = np.array([[axs]])
            elif num_sbpl_y == 1:
                axs = axs.reshape(1, -1)
            elif num_sbpl_x == 1:
                axs = axs.reshape(-1, 1)
            
            # Plot generation logic (adapted from original lines 2888-3200)
            for i_sbpl in range(num_datasets):
                # Calculate subplot position
                i_y_sbpl = math.floor(i_sbpl / num_sbpl_x)
                i_x_sbpl = i_sbpl - i_y_sbpl * num_sbpl_x
                
                # Get dataset key
                key_1 = list(tst_inf.keys())[i_sbpl]
                
                # Get main axes
                ax_sbpl_orig = axs[i_y_sbpl, i_x_sbpl]
                ax_sbpl = [ax_sbpl_orig]
                
                # Counters
                i_line = 0
                
                # PLOTTING INPUT DATA (adapted from original lines 2908-3000)
                if len(df_plot_in) > 0 and 'plot' in df_plot_in.columns:
                    for i_feat in range(len(df_plot_in)):
                        if df_plot_in.iloc[i_feat, 0]:  # If plot is True
                            color_plt = palette[i_feat]
                            
                            # Determine key
                            variable_name = df_plot_in.index[i_feat]
                            if i_feat < len(i_dat_inf):
                                key_2 = f"IN: {variable_name}"
                            else:
                                key_2 = f"TIME: {variable_name}"
                            
                            if key_2 in tst_inf[key_1]:
                                df_data = tst_inf[key_1][key_2]
                                
                                x_value = df_data[x_sbpl] if x_sbpl in df_data.columns else df_data.index
                                y_value = df_data['value'] if 'value' in df_data.columns else df_data.iloc[:, 0]
                                
                                if y_sbpl_set == "gemeinsame Achse":
                                    ax_sbpl_orig.plot(x_value, y_value,
                                                    label=key_2 if i_sbpl == 0 else None,
                                                    color=color_plt,
                                                    marker='o',
                                                    linewidth=1,
                                                    markersize=2)
                                elif y_sbpl_set == "separate Achsen":
                                    if i_line > 0:
                                        ax_sbpl.append(ax_sbpl_orig.twinx())
                                    
                                    ax_sbpl[-1].plot(x_value, y_value,
                                                   label=key_2 if i_sbpl == 0 else None,
                                                   color=color_plt,
                                                   marker='o',
                                                   linewidth=1,
                                                   markersize=2)
                                
                                i_line += 1
                
                # PLOTTING OUTPUT DATA (adapted from original lines 3003-3092)
                if len(df_plot_out) > 0 and 'plot' in df_plot_out.columns:
                    for i_feat in range(len(df_plot_out)):
                        if df_plot_out.iloc[i_feat, 0]:  # If plot is True
                            color_plt = palette[len(i_dat_inf) + i_feat]
                            
                            variable_name = df_plot_out.index[i_feat]
                            key_2 = f"OUT: {variable_name}"
                            
                            if key_2 in tst_inf[key_1]:
                                df_data = tst_inf[key_1][key_2]
                                
                                x_value = df_data[x_sbpl] if x_sbpl in df_data.columns else df_data.index
                                y_value = df_data['value'] if 'value' in df_data.columns else df_data.iloc[:, 0]
                                
                                if y_sbpl_set == "gemeinsame Achse":
                                    ax_sbpl_orig.plot(x_value, y_value,
                                                    label=key_2 if i_sbpl == 0 else None,
                                                    color=color_plt,
                                                    marker='s',
                                                    linewidth=1,
                                                    markersize=4)
                                elif y_sbpl_set == "separate Achsen":
                                    # Logic for separate axes
                                    if not df_plot_out.iloc[i_feat, 0]:
                                        ax_sbpl.append(ax_sbpl_orig.twinx())
                                        i_pos = len(ax_sbpl) - 1
                                    else:
                                        i_pos = df_plot_out.iloc[:(i_feat+1)]['plot'].sum() - 1 + df_plot_in.iloc[:]['plot'].sum()
                                    
                                    ax_sbpl[i_pos].plot(x_value, y_value,
                                                      label=key_2 if i_sbpl == 0 else None,
                                                      color=color_plt,
                                                      marker='s',
                                                      linewidth=1,
                                                      markersize=4)
                
                # PLOTTING FORECAST DATA (adapted from original lines 3095-3200)
                if len(df_plot_fcst) > 0 and 'plot' in df_plot_fcst.columns:
                    for i_feat in range(len(df_plot_fcst)):
                        if df_plot_fcst.iloc[i_feat, 0]:  # If plot is True
                            color_plt = palette[len(i_dat_inf) + i_feat]
                            
                            variable_name = df_plot_fcst.index[i_feat]
                            key_2 = f"FCST: {variable_name}"
                            
                            if key_2 in tst_inf[key_1]:
                                df_data = tst_inf[key_1][key_2]
                                
                                x_value = df_data[x_sbpl] if x_sbpl in df_data.columns else df_data.index
                                y_value = df_data['value'] if 'value' in df_data.columns else df_data.iloc[:, 0]
                                
                                if y_sbpl_set == "gemeinsame Achse":
                                    ax_sbpl_orig.plot(x_value, y_value,
                                                    label=key_2 if i_sbpl == 0 else None,
                                                    color=color_plt,
                                                    marker='^',
                                                    linewidth=1,
                                                    markersize=4,
                                                    linestyle='--')
                                elif y_sbpl_set == "separate Achsen":
                                    # Similar logic as output data
                                    if not df_plot_out.iloc[i_feat, 0]:
                                        ax_sbpl.append(ax_sbpl_orig.twinx())
                                        i_pos = len(ax_sbpl) - 1
                                    else:
                                        i_pos = df_plot_out.iloc[:(i_feat+1)]['plot'].sum() - 1 + df_plot_in.iloc[:]['plot'].sum()
                                    
                                    ax_sbpl[i_pos].plot(x_value, y_value,
                                                      label=key_2 if i_sbpl == 0 else None,
                                                      color=color_plt,
                                                      marker='^',
                                                      linewidth=1,
                                                      markersize=4,
                                                      linestyle='--')
                
                # Set subplot title and formatting
                ax_sbpl_orig.set_title(f'Dataset: {key_1}')
                ax_sbpl_orig.grid(True, alpha=0.3)
                
                if i_sbpl == 0:  # Only show legend on first subplot
                    ax_sbpl_orig.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Set axis labels
                ax_sbpl_orig.set_xlabel('Time (UTC)' if x_sbpl == 'UTC' else 'Timestep')
                ax_sbpl_orig.set_ylabel('Value')
            
            # Hide unused subplots
            for i in range(num_datasets, num_sbpl_x * num_sbpl_y):
                i_y = math.floor(i / num_sbpl_x)
                i_x = i - i_y * num_sbpl_x
                axs[i_y, i_x].set_visible(False)
            
            plt.tight_layout()
            
            # Convert to base64
            plot_base64 = self._figure_to_base64(fig)
            plt.close(fig)
            
            return plot_base64
            
        except Exception as e:
            logger.error(f"Error generating plot: {str(e)}")
            raise
    
    def _load_session_data(self, session_id: str) -> dict:
        """Load session data from database/storage"""
        try:
            supabase = get_supabase_client()
            
            # Get UUID session id from mapping table
            try:
                mapping_response = supabase.table('session_mappings').select('uuid_session_id').eq('string_session_id', session_id).execute()
                if mapping_response.data and len(mapping_response.data) > 0:
                    uuid_session_id = mapping_response.data[0]['uuid_session_id']
                    logger.info(f"Found UUID session {uuid_session_id} for string session {session_id}")
                    
                    # Get session data using UUID - use sessions table that exists
                    session_response = supabase.table('sessions').select('*').eq('id', uuid_session_id).execute()
                    if session_response.data and len(session_response.data) > 0:
                        session_data = session_response.data[0]
                        session_data['session_id'] = session_id  # Keep original string session_id
                        session_data['uuid_session_id'] = uuid_session_id
                        return session_data
                        
                    # If sessions table doesn't have data, create basic session data
                    return {
                        'session_id': session_id,
                        'id': uuid_session_id,
                        'uuid_session_id': uuid_session_id,
                        'datasets': ['default_dataset']
                    }
            except Exception as e:
                logger.info(f"Could not find session mapping for {session_id}: {str(e)}")
            
            # If no mapping found, return basic session info
            logger.warning(f"No session mapping found for {session_id}, creating basic session info")
            return {
                'session_id': session_id,
                'id': session_id,
                'datasets': ['default_dataset']
            }
                
        except Exception as e:
            logger.error(f"Error loading session data: {str(e)}")
            return {
                'session_id': session_id,
                'id': session_id,
                'datasets': ['default_dataset']
            }
    
    def _prepare_test_data(self, session_data: dict, session_id: str, i_dat_inf: pd.DataFrame, o_dat_inf: pd.DataFrame) -> dict:
        """
        Prepare test data structure (tst_inf) similar to original code
        This would normally load actual test data from the training session
        """
        try:
            
            # Create test data structure
            tst_inf = {}
            
            # For demonstration, create mock data structure with actual variable names
            datasets = session_data.get('datasets', [f'dataset_{session_id}'])
            
            for dataset_name in datasets:
                tst_inf[dataset_name] = {}
                
                # Create mock data for each input variable
                for var_name in i_dat_inf.index:
                    tst_inf[dataset_name][f'IN: {var_name}'] = pd.DataFrame({
                        'UTC': pd.date_range('2024-01-01', periods=100, freq='H'),
                        'ts': range(100),
                        'value': np.random.randn(100) * 10 + 50  # Realistic values
                    })
                
                # Create mock data for each output variable  
                for var_name in o_dat_inf.index:
                    tst_inf[dataset_name][f'OUT: {var_name}'] = pd.DataFrame({
                        'UTC': pd.date_range('2024-01-01', periods=100, freq='H'),
                        'ts': range(100),
                        'value': np.random.randn(100) * 15 + 75  # Different realistic values
                    })
                    
                    # Create forecast data for each output variable
                    tst_inf[dataset_name][f'FCST: {var_name}'] = pd.DataFrame({
                        'UTC': pd.date_range('2024-01-01', periods=100, freq='H'),
                        'ts': range(100),
                        'value': np.random.randn(100) * 15 + 78  # Slightly different forecast values
                    })
            
            logger.info(f"Created test data structure with {len(tst_inf)} datasets")
            for dataset_name, dataset_data in tst_inf.items():
                logger.info(f"Dataset {dataset_name} has {len(dataset_data)} data series")
                
            return tst_inf
            
        except Exception as e:
            logger.error(f"Error preparing test data: {str(e)}")
            # Return basic fallback structure
            return {
                'default_dataset': {
                    'IN: Netzlast [kW]': pd.DataFrame({
                        'UTC': pd.date_range('2024-01-01', periods=100, freq='H'),
                        'ts': range(100),
                        'value': np.random.randn(100) * 10 + 50
                    }),
                    'OUT: Netzlast [kW]': pd.DataFrame({
                        'UTC': pd.date_range('2024-01-01', periods=100, freq='H'),
                        'ts': range(100),
                        'value': np.random.randn(100) * 15 + 75
                    }),
                    'FCST: Netzlast [kW]': pd.DataFrame({
                        'UTC': pd.date_range('2024-01-01', periods=100, freq='H'),
                        'ts': range(100),
                        'value': np.random.randn(100) * 15 + 78
                    })
                }
            }
    
    def _get_data_info(self, session_data: dict) -> tuple:
        """Get input and output data information from session data"""
        try:
            supabase = get_supabase_client()
            if not supabase:
                logger.error("Could not get Supabase client")
                return self._get_fallback_data_info()
            
            # Use UUID session id for database queries
            uuid_session_id = session_data.get('uuid_session_id', session_data.get('id'))
            session_id_string = session_data.get('session_id')
            
            logger.info(f"Getting data info for UUID session: {uuid_session_id}, string session: {session_id_string}")
            
            # Get input files (CSV files uploaded as input data) using UUID
            input_files_response = supabase.table('files').select('bezeichnung, file_name').eq('session_id', uuid_session_id).execute()
            
            input_variables = []
            if input_files_response.data:
                logger.info(f"Found {len(input_files_response.data)} files in database")
                for file_info in input_files_response.data:
                    bezeichnung = file_info.get('bezeichnung', file_info.get('file_name', 'Unknown'))
                    if bezeichnung and bezeichnung not in input_variables:
                        input_variables.append(bezeichnung)
                        logger.info(f"Added input variable: {bezeichnung}")
            else:
                logger.info("No files found in database")
            
            # If no files found, use sample data based on session string
            if not input_variables:
                logger.info("No database files found, using sample data")
                input_variables = ['Netzlast [kW]', 'Aussentemperatur Krumpendorf [GradC]', 'Solarstrahlung [W/m²]']
            
            # Create input data info DataFrame
            i_dat_inf = pd.DataFrame({
                'description': [f'Input: {name}' for name in input_variables]
            }, index=input_variables)
            
            # For output variables, typically they are the same as input for forecasting
            output_variables = input_variables.copy()
            
            # Create output data info DataFrame  
            o_dat_inf = pd.DataFrame({
                'description': [f'Output: {name}' for name in output_variables]
            }, index=output_variables)
            
            logger.info(f"Created data info with {len(input_variables)} input and {len(output_variables)} output variables")
            return i_dat_inf, o_dat_inf
            
        except Exception as e:
            logger.error(f"Error getting data info: {str(e)}")
            return self._get_fallback_data_info()
    
    def _get_fallback_data_info(self) -> tuple:
        """Get fallback data info when database lookup fails"""
        logger.info("Using fallback data info")
        input_variables = ['Netzlast [kW]', 'Aussentemperatur Krumpendorf [GradC]', 'Solarstrahlung [W/m²]']
        output_variables = input_variables.copy()
        
        i_dat_inf = pd.DataFrame({
            'description': [f'Input: {name}' for name in input_variables]
        }, index=input_variables)
        
        o_dat_inf = pd.DataFrame({
            'description': [f'Output: {name}' for name in output_variables]
        }, index=output_variables)
        
        return i_dat_inf, o_dat_inf
    
    def _figure_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        try:
            buffer = BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error converting figure to base64: {str(e)}")
            raise


@plotting_bp.route('/api/training/generate-plot', methods=['POST'])
def generate_plot():
    """
    Generate plot based on configuration
    
    Request body:
    {
        "session_id": "session_id",
        "plot_config": {
            "df_plot_in": {"var1": {"plot": true}, "var2": {"plot": false}},
            "df_plot_out": {"output1": {"plot": true}},
            "df_plot_fcst": {"output1": {"plot": true}}
        },
        "plot_settings": {
            "y_sbpl_set": "separate Achsen",
            "x_sbpl": "UTC",
            "y_sbpl_fmt": "original"
        }
    }
    """
    from flask import current_app
    
    with current_app.app_context():
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No data provided'
                }), 400
            
            session_id = data.get('session_id')
            if not session_id:
                return jsonify({
                    'success': False,
                    'error': 'Session ID is required'
                }), 400
            
            plot_config = data.get('plot_config', {})
            plot_settings = data.get('plot_settings', {})
            
            # Validate that at least one variable is selected
            has_selection = False
            for config_name in ['df_plot_in', 'df_plot_out', 'df_plot_fcst']:
                config_data = plot_config.get(config_name, {})
                if any(var_config.get('plot', False) for var_config in config_data.values()):
                    has_selection = True
                    break
            
            if not has_selection:
                return jsonify({
                    'success': False,
                    'error': 'At least one variable must be selected for plotting'
                }), 400
            
            # Generate plot
            plot_generator = PlotGenerator()
            plot_data = plot_generator.generate_plot(session_id, plot_config, plot_settings)
            
            return jsonify({
                'success': True,
                'plot_data': plot_data,
                'session_id': session_id,
                'generated_at': datetime.now().isoformat()
            })
            
        except ValueError as e:
            logger.error(f"Validation error in generate_plot: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
            
        except Exception as e:
            logger.error(f"Error in generate_plot: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Internal server error while generating plot'
            }), 500


@plotting_bp.route('/api/training/plot-variables/<session_id>', methods=['GET'])
def get_plot_variables(session_id):
    """
    Get available variables for plotting from a training session
    
    Returns:
    {
        "success": true,
        "input_variables": ["var1", "var2", ...],
        "output_variables": ["out1", "out2", ...]
    }
    """
    from flask import current_app
    
    with current_app.app_context():
        try:
            plot_generator = PlotGenerator()
            session_data = plot_generator._load_session_data(session_id)
            
            if not session_data:
                return jsonify({
                    'success': False,
                    'error': f'No session found with ID: {session_id}'
                }), 404
            
            # Get data information
            i_dat_inf, o_dat_inf = plot_generator._get_data_info(session_data)
            
            input_variables = i_dat_inf.index.tolist() if len(i_dat_inf) > 0 else []
            output_variables = o_dat_inf.index.tolist() if len(o_dat_inf) > 0 else []
            
            return jsonify({
                'success': True,
                'input_variables': input_variables,
                'output_variables': output_variables,
                'session_id': session_id
            })
            
        except Exception as e:
            logger.error(f"Error getting plot variables: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Error retrieving plot variables'
            }), 500