import pandas as pd
import numpy as np
import math
import sys
import datetime
import statistics
from flask import Flask, request, jsonify
from flask_cors import CORS
from io import StringIO
import traceback
import json

UTC_fmt = "%Y-%m-%d %H:%M:%S"

app = Flask(__name__)
CORS(app)

# Global dictionary to store DataFrames
stored_data = {}  # Dictionary to store DataFrames
info_df = pd.DataFrame()  # Initialize empty DataFrame for info

def allowed_file(filename):
    """Check if file has .csv extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

@app.route('/analysedata', methods=['POST'])
def analyse_data():
    try:
        global stored_data, info_df
        
        if 'files[]' not in request.files:
            return jsonify({"error": "No files provided"}), 400
            
        files = request.files.getlist('files[]')
        all_file_info = []
        processed_data = []
        
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    # Read CSV directly into pandas
                    df = pd.read_csv(file, parse_dates=['UTC'])
                    print(f"\nLoaded file {file.filename}")
                    print(f"Columns: {df.columns.tolist()}")
                    print(f"Sample data:\n{df.head()}")
                    
                    # Store the DataFrame for later use
                    stored_data[file.filename] = df
                    
                    # Calculate basic statistics
                    numeric_columns = df.select_dtypes(include=[np.number]).columns
                    numeric_count = int(df[numeric_columns].count().sum())
                    total_count = int(len(df) * len(numeric_columns))
                    numeric_percentage = float(numeric_count / total_count * 100) if total_count > 0 else 0.0
                    
                    # Calculate time step if UTC column exists
                    time_step = None
                    if 'UTC' in df.columns:
                        try:
                            time_diffs = pd.to_datetime(df['UTC']).diff().dropna()
                            avg_time_diff = time_diffs.mean().total_seconds() / 60  # Convert to minutes
                            time_step = float(avg_time_diff)
                        except Exception as e:
                            print(f"Error calculating time step: {str(e)}")
                    
                    # Get the measurement column (second column)
                    measurement_col = df.columns[1] if len(df.columns) > 1 else None
                    if measurement_col:
                        print(f"\nMeasurement column: {measurement_col}")
                        print(f"Measurement stats:\n{df[measurement_col].describe()}")
                        
                        file_info = {
                            'Name der Datei': str(file.filename),
                            'Name der Messreihe': str(measurement_col),
                            'Startzeit (UTC)': df['UTC'].iloc[0].strftime(UTC_fmt) if 'UTC' in df.columns else None,
                            'Endzeit (UTC)': df['UTC'].iloc[-1].strftime(UTC_fmt) if 'UTC' in df.columns else None,
                            'Zeitschrittweite [min]': float(time_step) if time_step is not None else None,
                            'Offset [min]': 0.0,
                            'Anzahl der Datenpunkte': int(len(df)),
                            'Anzahl der numerischen Datenpunkte': int(df[measurement_col].count()),
                            'Anteil an numerischen Datenpunkten': float(df[measurement_col].count() / len(df) * 100)
                        }
                        
                        print(f"\nFile info:")
                        print(json.dumps(file_info, indent=2))
                        
                        all_file_info.append(file_info)
                    
                    # Convert DataFrame to records with explicit type conversion
                    df_records = []
                    for record in df.to_dict('records'):
                        converted_record = {}
                        for key, value in record.items():
                            if pd.isna(value):
                                converted_record[key] = None
                            elif isinstance(value, (pd.Timestamp, np.datetime64)):
                                converted_record[key] = value.strftime(UTC_fmt)
                            elif isinstance(value, np.number):
                                converted_record[key] = float(value) if not pd.isna(value) else None
                            else:
                                converted_record[key] = value
                        df_records.append(converted_record)
                    
                    processed_data.append(df_records)
                    
                except Exception as e:
                    print(f"Error processing file {file.filename}: {str(e)}")
                    traceback.print_exc()
                    return jsonify({"error": f"Error processing file {file.filename}: {str(e)}"}), 400
            else:
                return jsonify({"error": f"Invalid file format for {file.filename}"}), 400
        
        # Update global info_df - append new info to existing
        if all_file_info:
            new_info_df = pd.DataFrame(all_file_info)
            if info_df.empty:
                info_df = new_info_df
            else:
                # Remove any existing entries for these files
                existing_files = new_info_df['Name der Datei'].tolist()
                info_df = info_df[~info_df['Name der Datei'].isin(existing_files)]
                # Append new info
                info_df = pd.concat([info_df, new_info_df], ignore_index=True)
        
        return jsonify({
            'success': True,
            'data': {
                'info_df': all_file_info,
                'dataframe': processed_data[0] if processed_data else []
            }
        })
        
    except Exception as e:
        print(f"Error in analyse_data: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

def process_data_detailed(df, start_time=None, end_time=None, time_step=None, offset=None, method='mean'):
    try:
        # Convert UTC column to datetime if it's not already
        df['UTC'] = pd.to_datetime(df['UTC'])
        
        # Get the measurement column (second column)
        measurement_col = df.columns[1]
        print(f"\nProcessing measurement column: {measurement_col}")
        print("Sample data:")
        print(df.head())
        
        # Convert measurement values to float, replacing non-numeric values with NaN
        df[measurement_col] = pd.to_numeric(df[measurement_col], errors='coerce')
        
        # Filter by time range if provided
        if start_time and end_time:
            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)
            df = df[(df['UTC'] >= start_time) & (df['UTC'] <= end_time)]
            print(f"\nFiltered by time range: {start_time} to {end_time}")
            print(f"Remaining points: {len(df)}")
        
        # Create regular time index
        if time_step:
            # Create new time index
            if start_time is None:
                start_time = df['UTC'].min()
            if end_time is None:
                end_time = df['UTC'].max()
                
            # Apply offset if provided
            if offset:
                start_time = start_time + pd.Timedelta(minutes=offset)
                
            # Create new index with specified time step
            new_index = pd.date_range(start=start_time, end=end_time, freq=f'{time_step}min')
            
            # Resample data based on method
            if method == 'mean':
                resampled = df.set_index('UTC').resample(f'{time_step}min', offset=f'{offset}min')[measurement_col].mean()
                result_df = pd.DataFrame({
                    'UTC': resampled.index,
                    measurement_col: resampled.values
                })
            elif method == 'max':
                resampled = df.set_index('UTC').resample(f'{time_step}min', offset=f'{offset}min')[measurement_col].max()
                result_df = pd.DataFrame({
                    'UTC': resampled.index,
                    measurement_col: resampled.values
                })
            elif method == 'min':
                resampled = df.set_index('UTC').resample(f'{time_step}min', offset=f'{offset}min')[measurement_col].min()
                result_df = pd.DataFrame({
                    'UTC': resampled.index,
                    measurement_col: resampled.values
                })
            elif method == 'nearest':
                df_indexed = df.set_index('UTC')
                resampled = df_indexed[measurement_col].reindex(new_index, method='nearest')
                result_df = pd.DataFrame({
                    'UTC': new_index,
                    measurement_col: resampled.values
                })
            elif method == 'nearest (mean)':
                df_indexed = df.set_index('UTC')
                nearest_vals = df_indexed[measurement_col].reindex(new_index, method='nearest')
                rolling_mean = nearest_vals.rolling(window=2, min_periods=1).mean()
                result_df = pd.DataFrame({
                    'UTC': new_index,
                    measurement_col: rolling_mean.values
                })
            elif method == 'nearest (max. delta)':
                df_indexed = df.set_index('UTC')
                nearest_vals = df_indexed[measurement_col].reindex(new_index, method='nearest')
                deltas = nearest_vals.diff().abs()
                max_delta = deltas.quantile(0.95)  # Use 95th percentile as threshold
                masked_values = nearest_vals.where(deltas <= max_delta)
                result_df = pd.DataFrame({
                    'UTC': new_index,
                    measurement_col: masked_values.values
                })
            else:
                result_df = df
        else:
            result_df = df
            
        # Calculate statistics
        total_points = len(result_df)
        numeric_points = result_df[measurement_col].count()
        numeric_ratio = (numeric_points / total_points * 100) if total_points > 0 else 0
        
        print("\nStatistics:")
        print(f"Total points: {total_points}")
        print(f"Numeric points: {numeric_points}")
        print(f"Numeric ratio: {numeric_ratio:.1f}%")
        
        # Create info record
        info_record = {
            'Name der Datei': f"{df.name}_processed" if hasattr(df, 'name') else "processed_data",
            'Name der Messreihe': measurement_col,
            'Startzeit (UTC)': result_df['UTC'].iloc[0].strftime(UTC_fmt) if len(result_df) > 0 else None,
            'Endzeit (UTC)': result_df['UTC'].iloc[-1].strftime(UTC_fmt) if len(result_df) > 0 else None,
            'Zeitschrittweite [min]': float(time_step) if time_step else None,
            'Offset [min]': float(offset) if offset else 0.0,
            'Anzahl der Datenpunkte': int(total_points),
            'Anzahl der numerischen Datenpunkte': int(numeric_points),
            'Anteil an numerischen Datenpunkten': float(numeric_ratio)
        }
        
        print("\nInfo record:")
        print(json.dumps(info_record, indent=2))
        
        # Convert to records
        records = []
        for _, row in result_df.iterrows():
            utc_timestamp = int(pd.to_datetime(row['UTC']).timestamp() * 1000)  # Convert to milliseconds
            value = float(row[measurement_col]) if pd.notnull(row[measurement_col]) else None
            record = {'UTC': utc_timestamp, measurement_col: value}
            records.append(record)
            
        print(f"\nConverted {len(records)} records")
        if records:
            print(f"Sample record: {records[0]}")
            
        return records, info_record
        
    except Exception as e:
        print(f"Error in process_data_detailed: {str(e)}")
        traceback.print_exc()
        raise

@app.route('/adjustdata', methods=['POST'])
def adjust_data():
    try:
        global stored_data, info_df
        print("\n=== Starting data adjustment ===")
        
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
            
        # Get parameters only once
        start_time = data.get('startTime')
        end_time = data.get('endTime')
        time_step = data.get('timeStepSize')
        offset = data.get('offset', 0)  # Default to 0
        methods = data.get('methods', {})  # Dictionary of filename -> method
        files = data.get('files', [])  # List of files to process
        
        print(f"\nReceived parameters:")
        print(f"Start time: {start_time}")
        print(f"End time: {end_time}")
        print(f"Time step: {time_step}")
        print(f"Offset: {offset}")
        print(f"Methods: {methods}")
        print(f"Files: {files}")
        
        if not stored_data or info_df.empty:
            return jsonify({"error": "No data available. Please upload a file first."}), 400

        # Create dictionary of files to process
        files_to_process = {}
        for filename in files:
            if filename in stored_data:
                files_to_process[filename] = stored_data[filename]
            else:
                return jsonify({"error": f"File {filename} not found"}), 400

        all_processed_data = []
        all_processed_info = []
        files_needing_methods = []
        processed_files = set()  # Keep track of processed files

        # First pass: identify files needing method selection
        for filename, df in files_to_process.items():
            if filename in processed_files:
                continue

            try:
                file_info = info_df[info_df['Name der Datei'] == filename].iloc[0].to_dict()
                current_timestep = float(file_info.get('Zeitschrittweite [min]', 0))
                current_offset = float(file_info.get('Offset [min]', 0))
                
                needs_method = False
                if time_step is not None:
                    time_step_float = float(time_step)
                    offset_float = float(offset) if offset is not None else 0.0
                    needs_method = not (math.isclose(current_timestep, time_step_float, rel_tol=1e-9) and 
                                     math.isclose(current_offset, offset_float, rel_tol=1e-9))
                    
                    if needs_method:
                        # Check if we have a method for this file
                        file_method = methods.get(filename)
                        files_needing_methods.append({
                            'filename': filename,
                            'current_timestep': current_timestep,
                            'requested_timestep': time_step_float,
                            'current_offset': current_offset,
                            'requested_offset': offset_float,
                            'method': file_method
                        })
                        if not file_method:
                            continue

                # Process data if method is provided or not needed
                if not needs_method or (needs_method and methods.get(filename)):
                    actual_start = pd.to_datetime(start_time) if start_time else pd.to_datetime(file_info['Startzeit (UTC)'])
                    actual_end = pd.to_datetime(end_time) if end_time else pd.to_datetime(file_info['Endzeit (UTC)'])
                    
                    df.name = filename  # Set name for the DataFrame
                    result, info_record = process_data_detailed(
                        df=df,
                        start_time=actual_start.strftime(UTC_fmt),
                        end_time=actual_end.strftime(UTC_fmt),
                        time_step=time_step,
                        offset=offset,
                        method=methods.get(filename) if needs_method else None
                    )
                    
                    if result:
                        all_processed_data.extend(result)
                        all_processed_info.append(info_record)
                        processed_files.add(filename)
                
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                traceback.print_exc()
                continue

        # Return appropriate response based on processing results
        if files_needing_methods and not all(f.get('method') for f in files_needing_methods):
            return jsonify({
                "success": True,
                "methodsRequired": True,
                "filesNeedingMethods": files_needing_methods,
                "data": None
            })
        elif all_processed_info:
            return jsonify({
                "success": True,
                "methodsRequired": False,
                "data": {
                    "info_df": all_processed_info,
                    "dataframe": all_processed_data
                }
            })
        else:
            return jsonify({"error": "No data was processed"}), 400

    except Exception as e:
        print(f"Error in adjust_data: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("Starting Flask server on Port 5007..")
    app.run(port=5007, debug=True)