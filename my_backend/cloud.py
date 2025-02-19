import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
from io import StringIO, BytesIO
import json
import os
import sys
import tempfile
import csv
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary to store temporary files
temp_files = {}


def calculate_bounds(predictions, tolerance_type, tol_cnt, tol_dep): 
    """Calculate upper and lower bounds based on tolerance type."""
    if tolerance_type == 'cnt':
        upper_bound = predictions + tol_cnt
        lower_bound = predictions - tol_cnt
    else:  # tolerance_type == 'dep'
        upper_bound = predictions * (1 + tol_dep) + tol_cnt
        lower_bound = predictions * (1 - tol_dep) - tol_cnt
    
    # Ensure lower bound is not negative
    lower_bound = np.maximum(lower_bound, 0)
    
    return upper_bound, lower_bound

def interpolate_data(df1, df2, x_col, y_col, max_time_span):
    """Perform linear interpolation on the data within the specified time span."""
    try:
        # Create combined dataframe
        cld = pd.DataFrame()
        cld['UTC'] = df1['UTC']
        cld[x_col] = df1[x_col]
        cld[y_col] = df2[y_col]
        
        # Store original number of points
        original_points = len(cld)
        
        # Drop any rows where either x or y is NaN
        cld = cld.dropna()
        
        if cld.empty:
            raise ValueError('No valid data points after combining datasets')
        
        print(f"Combined data shape: {cld.shape}")
        
        # Calculate time differences between consecutive points
        cld['time_diff'] = cld['UTC'].diff().dt.total_seconds() / 60  # Convert to minutes
        
        # Find gaps larger than max_time_span
        large_gaps = cld[cld['time_diff'] > max_time_span].index
        
        interpolated_points = 0
        
        if len(large_gaps) > 0:
            print(f"Found {len(large_gaps)} gaps larger than {max_time_span} minutes")
            
            # Split data into chunks where gaps are too large
            chunks = []
            start_idx = 0
            
            for gap_idx in large_gaps:
                if gap_idx > start_idx:
                    chunk = cld.loc[start_idx:gap_idx-1].copy()
                    # Perform linear interpolation within the chunk
                    interpolated_chunk = chunk.set_index('UTC').resample('1min').interpolate(method='linear')
                    chunks.append(interpolated_chunk)
                start_idx = gap_idx
            
            # Add the last chunk
            if start_idx < len(cld):
                chunk = cld.loc[start_idx:].copy()
                chunk = chunk.set_index('UTC').resample('1min').interpolate(method='linear')
                chunks.append(chunk)
            
            # Combine all interpolated chunks
            cld_interpolated = pd.concat(chunks)
            
            # Update the working dataframe
            cld = cld_interpolated.reset_index()
            
            # Calculate number of interpolated points
            interpolated_points = len(cld) - original_points
            
            print(f"After interpolation: {len(cld)} points (Added {interpolated_points} points)")
        
        return cld, interpolated_points
    except Exception as e:
        print(f"Error in interpolation: {str(e)}")
        raise

def clouddata(request):
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    try:
        print("\nReceived request to /clouddata")
        return _process_data()
    except Exception as e:
        print(f"Error in clouddata endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def _process_data():
    try:
        print("\nReceived request to /clouddata")
        data = request.json
        print("Received data:", data)
        
        # Check if files are present in request
        if 'files' not in data or 'temp_out.csv' not in data['files'] or 'load.csv' not in data['files']:
            print("Missing files. Required: temp_out.csv, load.csv")
            return jsonify({'error': 'Both temperature and load files are required'}), 400
            
        # Get files from request
        temp_data = data['files']['temp_out.csv']
        load_data = data['files']['load.csv']
        
        if not temp_data or not load_data:
            print("One or both files are empty")
            return jsonify({'error': 'One or both files are empty'}), 400
        
        try:
            # Convert base64 data to DataFrame
            print("Attempting to decode and read temperature data...")
            temp_decoded = base64.b64decode(temp_data).decode('utf-8')
            print("Temperature data preview:", temp_decoded[:200])
            df1 = pd.read_csv(StringIO(temp_decoded), sep=';')
            
            print("Attempting to decode and read load data...")
            load_decoded = base64.b64decode(load_data).decode('utf-8')
            print("Load data preview:", load_decoded[:200])
            df2 = pd.read_csv(StringIO(load_decoded), sep=';')
            
            print(f"Successfully read data. Shapes: {df1.shape}, {df2.shape}")
            print(f"Columns in temperature file: {df1.columns.tolist()}")
            print(f"Columns in load file: {df2.columns.tolist()}")
            
            print("First few rows of temperature file:")
            print(df1.head())
            print("First few rows of load file:")
            print(df2.head())
            
            # Try to find the correct column names
            temp_cols = [col for col in df1.columns if 'temp' in col.lower() or 'grad' in col.lower()]
            load_cols = [col for col in df2.columns if any(term in col.lower() 
                                                         for term in ['last', 'load', 'leistung', 'kw', 'w'])]
            
            if temp_cols:
                x = temp_cols[0]
                print(f"Found temperature column: {x}")
            else:
                print("No temperature column found. Available columns:", df1.columns.tolist())
                return jsonify({'error': f'Temperature column not found. Available columns: {df1.columns.tolist()}'}), 400
                
            if load_cols:
                y = load_cols[0]
                print(f"Found load column: {y}")
            else:
                print("No load column found. Available columns:", df2.columns.tolist())
                return jsonify({'error': f'Load column not found. Available columns: {df2.columns.tolist()}'}), 400
                
            # Convert time column to datetime with specific format and sort
            df1['UTC'] = pd.to_datetime(df1['UTC'], format="%Y-%m-%d %H:%M:%S")
            df2['UTC'] = pd.to_datetime(df2['UTC'], format="%Y-%m-%d %H:%M:%S")
            
            # Convert data columns to numeric
            df1[x] = pd.to_numeric(df1[x], errors='coerce')
            df2[y] = pd.to_numeric(df2[y], errors='coerce')
            
            # Sort both dataframes by time
            df1 = df1.sort_values('UTC')
            df2 = df2.sort_values('UTC')
            
            print(f"Data ranges after sorting:")
            print(f"Temperature range: {df1[x].min():.2f} to {df1[x].max():.2f} °C")
            print(f"Load range before conversion: {df2[y].min():.2f} to {df2[y].max():.2f} kW")
            
            # Convert kW to W if needed
            if 'kw' in y.lower():
                print("Converting kW to W")
                df2[y] = df2[y] * 1000
                print(f"Load range after conversion: {df2[y].min():.2f} to {df2[y].max():.2f} W")
            
            # Verify data alignment
            if not df1['UTC'].equals(df2['UTC']):
                print("Warning: Time stamps don't match exactly")
                print(f"Temperature times: {df1['UTC'].tolist()}")
                print(f"Load times: {df2['UTC'].tolist()}")
                return jsonify({'error': 'Time stamps in files do not match'}), 400
            
            # Clean and validate data
            try:
                # Drop any rows with NaN values
                df1 = df1.dropna()
                df2 = df2.dropna()
                
                if df1.empty or df2.empty:
                    return jsonify({'error': 'No valid numeric data found after cleaning'}), 400
                    
                print(f"Data cleaned. New shapes: {df1.shape}, {df2.shape}")
                
                # Create combined dataframe
                cld = pd.DataFrame()
                cld[x] = df1[x]
                cld[y] = df2[y]
                
                # Drop any rows where either x or y is NaN
                cld = cld.dropna()
                
                if cld.empty:
                    return jsonify({'error': 'No valid data points after combining datasets'}), 400
                
                print(f"Combined data shape: {cld.shape}")
                
                # Sort by x values
                cld_srt = cld.sort_values(by=x).copy()
                
                # Verify no NaN values remain
                if cld_srt[x].isna().any() or cld_srt[y].isna().any():
                    return jsonify({'error': 'NaN values found in processed data'}), 400
                    
            except Exception as e:
                print(f"Error cleaning data: {str(e)}")
                return jsonify({'error': f'Error cleaning data: {str(e)}'}), 400
            
            # Get parameters with reasonable defaults
            REG = data.get('REG', 'lin')  # Default to linear regression
            TR = data.get('TR', 'cnt')    # Default to constant tolerance
            
            # Set default tolerances based on data range
            y_range = df2[y].max() - df2[y].min()
            default_tol = y_range * 0.1  # 10% of data range
            
            # Get tolerance parameters
            try:
                TOL_CNT = float(data.get('TOL_CNT', default_tol))
                TOL_DEP = float(data.get('TOL_DEP', 0.1))  # 10% default for dependent tolerance
            except ValueError:
                print("Using default tolerances due to invalid input")
                TOL_CNT = default_tol
                TOL_DEP = 0.1
            
            # If tolerance is too small compared to data range, adjust it
            if TOL_CNT < y_range * 0.01:  # If tolerance is less than 1% of range
                TOL_CNT = y_range * 0.1  # Set to 10% of range
                print(f"Adjusted tolerance to {TOL_CNT:.2f} (10% of data range)")
            
            print(f"Parameters: REG={REG}, TR={TR}, TOL_CNT={TOL_CNT}, TOL_DEP={TOL_DEP}")
            
            # Perform regression
            if REG == "lin":
                try:
                    # Get min and max values for constraints
                    min_y = cld_srt[y].min()
                    max_y = cld_srt[y].max()
                    x_min = cld_srt[x].min()
                    x_max = cld_srt[x].max()
                    
                    # Print data ranges for debugging
                    print(f"Data ranges - X: [{x_min}, {x_max}], Y: [{min_y}, {max_y}]")
                    
                    # Fit linear regression
                    lin_mdl = LinearRegression()
                    lin_mdl.fit(cld_srt[[x]], cld_srt[y])
                    
                    # Calculate predictions
                    lin_prd = lin_mdl.predict(cld_srt[[x]])
                    
                    # Calculate average y value for high temperatures
                    high_temp_threshold = x_max * 0.8  # Last 20% of temperature range
                    high_temp_avg = cld_srt[cld_srt[x] >= high_temp_threshold][y].mean()
                    
                    # Replace predictions for high temperatures with average value
                    for i in range(len(lin_prd)):
                        curr_x = cld_srt[x].iloc[i]
                        if curr_x >= high_temp_threshold:
                            lin_prd[i] = high_temp_avg
                    
                    # Print last few predictions for debugging
                    print("Last 5 points:")
                    for i in range(-5, 0):
                        print(f"X: {cld_srt[x].iloc[i]}, Y: {cld_srt[y].iloc[i]}, Pred: {lin_prd[i]}")
                    
                    # Calculate bounds with validation
                    upper_bound, lower_bound = calculate_bounds(lin_prd, TR, TOL_CNT, TOL_DEP)
                    
                    # Ensure bounds are within reasonable limits and not zero
                    upper_bound = np.maximum(upper_bound, high_temp_avg * 1.2)  # At least 20% above average
                    lower_bound = np.maximum(lower_bound, high_temp_avg * 0.8)  # At least 20% below average
                    
                    # Get coefficients for equation
                    lin_fcn = f"y = {lin_mdl.coef_[0]:.2f}x + {lin_mdl.intercept_:.2f}"
                    print("Linear regression:", lin_fcn)
                    
                    # Calculate tolerance bounds
                    upper_bound, lower_bound = calculate_bounds(lin_prd, TR, TOL_CNT, TOL_DEP)
                    
                    # Filter points within tolerance
                    mask = (cld_srt[y] >= lower_bound) & (cld_srt[y] <= upper_bound)
                    cld_srt_flt = cld_srt[mask]
                    
                    if len(cld_srt_flt) == 0:
                        print(f"No points within tolerance bounds. Min y: {cld_srt[y].min()}, Max y: {cld_srt[y].max()}")
                        print(f"Bounds: lower={lower_bound.min()}, upper={upper_bound.max()}")
                        return jsonify({
                            'error': 'No points within tolerance bounds. Try increasing the tolerance values.',
                            'min_y': float(cld_srt[y].min()),
                            'max_y': float(cld_srt[y].max()),
                            'min_bound': float(lower_bound.min()),
                            'max_bound': float(upper_bound.max())
                        }), 400
                    
                    # Prepare response data
                    result_data = {
                        'x_values': cld_srt[x].tolist(),
                        'y_values': cld_srt[y].tolist(),
                        'predicted_y': lin_prd.tolist(),
                        'upper_bound': upper_bound.tolist(),
                        'lower_bound': lower_bound.tolist(),
                        'filtered_x': cld_srt_flt[x].tolist(),
                        'filtered_y': cld_srt_flt[y].tolist(),
                        'equation': lin_fcn,
                        'removed_points': len(cld_srt) - len(cld_srt_flt)
                    }
                except ValueError as ve:
                    return jsonify({'error': str(ve)}), 400
                except Exception as e:
                    print(f"Error in linear regression: {str(e)}")
                    return jsonify({'error': f'Error in linear regression: {str(e)}'}), 500
            else:  # REG == "poly"
                try:
                    # Polynomial regression
                    poly_features = PolynomialFeatures(degree=2)
                    X_poly = poly_features.fit_transform(cld_srt[[x]])
                    
                    poly_mdl = LinearRegression()
                    poly_mdl.fit(X_poly, cld_srt[y])
                    
                    # Generate predictions
                    poly_prd = poly_mdl.predict(X_poly)
                    
                    # Get coefficients for equation
                    coeffs = poly_mdl.coef_
                    intercept = poly_mdl.intercept_
                    poly_fcn = f"y = {coeffs[2]:.2f}x² + {coeffs[1]:.2f}x + {intercept:.2f}"
                    print("Polynomial regression:", poly_fcn)
                    
                    # Calculate tolerance bounds
                    upper_bound, lower_bound = calculate_bounds(poly_prd, TR, TOL_CNT, TOL_DEP)
                    
                    # Filter points within tolerance
                    mask = (cld_srt[y] >= lower_bound) & (cld_srt[y] <= upper_bound)
                    cld_srt_flt = cld_srt[mask]
                    
                    if len(cld_srt_flt) == 0:
                        print(f"No points within tolerance bounds. Min y: {cld_srt[y].min()}, Max y: {cld_srt[y].max()}")
                        print(f"Bounds: lower={lower_bound.min()}, upper={upper_bound.max()}")
                        return jsonify({
                            'error': 'No points within tolerance bounds. Try increasing the tolerance values.',
                            'min_y': float(cld_srt[y].min()),
                            'max_y': float(cld_srt[y].max()),
                            'min_bound': float(lower_bound.min()),
                            'max_bound': float(upper_bound.max())
                        }), 400
                    
                    # Prepare response data
                    result_data = {
                        'x_values': cld_srt[x].tolist(),
                        'y_values': cld_srt[y].tolist(),
                        'predicted_y': poly_prd.tolist(),
                        'upper_bound': upper_bound.tolist(),
                        'lower_bound': lower_bound.tolist(),
                        'filtered_x': cld_srt_flt[x].tolist(),
                        'filtered_y': cld_srt_flt[y].tolist(),
                        'equation': poly_fcn
                    }
                    
                except ValueError as ve:
                    return jsonify({'error': str(ve)}), 400
                except Exception as e:
                    print(f"Error in polynomial regression: {str(e)}")
                    return jsonify({'error': f'Error in polynomial regression: {str(e)}'}), 500

            # Send response
            print("Sending response:", {'success': True, 'data': result_data})
            return jsonify({'success': True, 'data': result_data})

        except UnicodeDecodeError as e:
            print(f"Error decoding CSV files: {str(e)}")
            return jsonify({'error': 'Error decoding CSV files. Please ensure files are UTF-8 encoded.'}), 400
        except pd.errors.EmptyDataError:
            print("Error: One or both CSV files are empty")
            return jsonify({'error': 'One or both CSV files are empty'}), 400
        except Exception as e:
            print(f"Error reading CSV files: {str(e)}")
            return jsonify({'error': f'Error reading CSV files: {str(e)}'}), 400

    except Exception as e:
        print("Error processing data:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def interpolate(request):
    try:
        print("\nReceived request to /interpolate")
        
        # Check if files are present in request
        if 'load.csv' not in request.files:
            return jsonify({'error': 'load files are required'}), 400
            
        # Get files and parameters from request
        load_file = request.files['load.csv']
        max_time_span = float(request.form.get('max_time_span', '60'))
        
        # Read CSV file
        df2 = pd.read_csv(StringIO(load_file.stream.read().decode('utf-8')), 
                         sep=';',      # Use semicolon as separator
                         decimal=',')   # Use comma as decimal separator
        
        # Find load column
        load_cols = [col for col in df2.columns if any(term in col.lower() 
                                                     for term in ['last', 'load', 'leistung', 'kw', 'w'])]
        
        if not load_cols:
            return jsonify({'error': 'Load column not found'}), 400
        
        y_col = load_cols[0]
        
        # Print first few rows to debug
        print("First few rows of data:")
        print(df2[y_col].head())
        
        # Clean the data - remove any non-numeric characters and convert to float
        df2[y_col] = df2[y_col].replace(r'[^\d\-\.,]', '', regex=True)
        # Replace comma with dot for decimal point
        df2[y_col] = df2[y_col].str.replace(',', '.')
        
        # Convert to float, replacing any invalid values with NaN
        df2[y_col] = pd.to_numeric(df2[y_col], errors='coerce')
        
        # Remove rows with NaN values
        df2 = df2.dropna(subset=[y_col])
        
        # Convert time column to datetime with specific format
        df2['UTC'] = pd.to_datetime(df2['UTC'], format="%Y-%m-%d %H:%M:%S")
        
        # Sort by time
        df2 = df2.sort_values('UTC')
        
        print(f"Data types after cleaning:")
        print(df2.dtypes)
        print("\nSample of cleaned data:")
        print(df2[[y_col, 'UTC']].head())
        
        # Create a DataFrame with just UTC and load data
        df_load = pd.DataFrame({
            'UTC': df2['UTC'],
            'load': df2[y_col]  # Already converted to float
        })
        
        # Set UTC as index
        df_load.set_index('UTC', inplace=True)
        
        # Resample to 1-minute intervals and interpolate
        df2_resampled = df_load.resample('1min').interpolate(method='linear')
        
        # Reset index to get UTC back as a column
        df2_resampled.reset_index(inplace=True)
        
        # Calculate added points
        added_points = len(df2_resampled) - len(df2)
        
        print(f"Original points: {len(df2)}")
        print(f"Interpolated points: {len(df2_resampled)}")
        print(f"Added points: {added_points}")
        
        # Prepare data for frontend chart
        chart_data = []
        for _, row in df2_resampled.iterrows():
            chart_data.append({
                'UTC': row['UTC'].strftime("%Y-%m-%d %H:%M:%S"),
                'value': float(row['load'])
            })

        print("Sample of chart data being sent:")
        print(chart_data[:5])  # Print first 5 points for debugging

        return jsonify({
            'success': True,
            'data': {
                'points': chart_data,
                'added_points': added_points,
                'total_points': len(df2_resampled),
                'removed_points': 0
            }
        })
        
    except Exception as e:
        print(f"Error in interpolation endpoint: {str(e)}")
        print("Full error details:", e)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def prepare_save(request):
    try:
        logger.info("Received prepare_save request")
        data = request.json
        if not data or 'data' not in data:
            logger.error("No data received in request")
            return jsonify({"error": "No data received"}), 400
            
        save_data = data['data']
        if not save_data:
            logger.error("Empty data received")
            return jsonify({"error": "Empty data"}), 400

        logger.info(f"Processing {len(save_data)} rows of data")

        # Create temporary file and write CSV data
        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
        writer = csv.writer(temp_file, delimiter=';')
        
        try:
            for row in save_data:
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Error writing to CSV: {str(e)}")
            os.unlink(temp_file.name)  # Clean up the file
            raise
            
        temp_file.close()
        logger.info(f"Successfully wrote data to temporary file: {temp_file.name}")

        # Generate unique ID based on current time
        file_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_files[file_id] = temp_file.name
        logger.info(f"Generated file ID: {file_id}")

        return jsonify({"message": "File prepared for download", "fileId": file_id}), 200
    except Exception as e:
        logger.error(f"Error in prepare_save: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

def download_file(file_id, request):
    """Download a previously prepared file.
    
    Args:
        file_id (str): The unique identifier for the file
        request: The Flask request object
    
    Returns:
        Flask response with the file or error message
    """
    try:
        logger.info(f"Received download request for file ID: {file_id}")
        if file_id not in temp_files:
            return jsonify({"error": "File not found"}), 404
        file_path = temp_files[file_id]
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        download_name = f"data_{file_id}.csv"
        return send_file(
            file_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='text/csv'
        )
    except Exception as e:
        logger.error(f"Error in download_file: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Pokušaj očistiti privremeni fajl
        if file_id in temp_files:
            try:
                os.unlink(temp_files[file_id])
                del temp_files[file_id]
            except Exception as ex:
                logger.error(f"Error cleaning up temp file: {ex}")