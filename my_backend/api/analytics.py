"""
Analytics API - Consolidated Analysis and Visualization Services

Consolidates functionality from:
- cloud.py - Cloud-based data analysis and visualization
- training_system/plotting.py - Training visualization generation

API Endpoints:
    POST /api/analytics/upload-chunk - Upload data for analysis
    POST /api/analytics/analyze - Statistical analysis
    POST /api/analytics/interpolate - Data interpolation
    POST /api/analytics/visualize - Generate charts and plots
    POST /api/analytics/cloud-process - Cloud-based analysis
    GET  /api/analytics/chart/<chart_id> - Download generated charts
    POST /api/analytics/prepare-save - Prepare analysis results for download
    GET  /api/analytics/download/<file_id> - Download analysis results
"""

import os
import tempfile
import logging
import json
import csv
import time
import secrets
import threading
import traceback
import base64
from datetime import datetime
from io import StringIO, BytesIO
from typing import Dict, Any, Optional
from threading import Lock

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from flask import Blueprint, request, jsonify, send_file, Response, current_app

# ============================================================================
# CONFIGURATION
# ============================================================================

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Blueprint
bp = Blueprint('analytics', __name__)

# Helper function to get socketio instance
def get_socketio():
    return current_app.extensions['socketio']

# Thread-safe storage
temp_files: Dict[str, Dict[str, Any]] = {}
chunk_uploads: Dict[str, Dict[str, Any]] = {}
storage_lock = Lock()

# Configuration constants
CHUNK_DIR = os.path.join(tempfile.gettempdir(), 'analytics_chunks')
os.makedirs(CHUNK_DIR, exist_ok=True)
VALID_FILE_TYPES = ['temp_file', 'load_file', 'interpolate_file']

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_chunk_dir(upload_id: str) -> str:
    """Create and return a directory path for storing chunks of a specific upload."""
    chunk_dir = os.path.join(CHUNK_DIR, upload_id)
    os.makedirs(chunk_dir, exist_ok=True)
    return chunk_dir

def emit_progress(room: str, stage: str, progress: int, message: str = "", error: Optional[str] = None):
    """Emit progress update via SocketIO"""
    try:
        socketio = get_socketio()
        data = {
            'stage': stage,
            'progress': progress,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        if error:
            data['error'] = error
        
        socketio.emit('analytics_progress', data, room=room)
        logger.info(f"Emitted analytics progress to room {room}: {stage} - {progress}%")
    except Exception as e:
        logger.error(f"Failed to emit analytics progress: {e}")

def generate_chart_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close(fig)
    return image_base64

# ============================================================================
# FILE UPLOAD ENDPOINTS
# ============================================================================

@bp.route('/upload-chunk', methods=['POST'])
def upload_chunk():
    """Upload file chunks for analysis"""
    try:
        upload_id = request.form.get('uploadId')
        chunk_index = int(request.form.get('chunkIndex', 0))
        total_chunks = int(request.form.get('totalChunks', 1))
        filename = request.form.get('filename', '')
        file_type = request.form.get('fileType', 'temp_file')
        
        if not upload_id:
            return jsonify({'error': 'Upload ID is required'}), 400
            
        if file_type not in VALID_FILE_TYPES:
            return jsonify({'error': f'Invalid file type. Must be one of: {VALID_FILE_TYPES}'}), 400
            
        if 'chunk' not in request.files:
            return jsonify({'error': 'No chunk file provided'}), 400
            
        chunk_file = request.files['chunk']
        chunk_data = chunk_file.read()
        
        # Initialize or update chunk storage
        with storage_lock:
            if upload_id not in chunk_uploads:
                chunk_uploads[upload_id] = {
                    'chunks': {},
                    'filename': filename,
                    'file_type': file_type,
                    'total_chunks': total_chunks,
                    'chunk_dir': get_chunk_dir(upload_id),
                    'last_activity': time.time()
                }
            
            # Store chunk
            chunk_uploads[upload_id]['chunks'][chunk_index] = chunk_data
            chunk_uploads[upload_id]['last_activity'] = time.time()
            
            received_chunks = len(chunk_uploads[upload_id]['chunks'])
            progress = int((received_chunks / total_chunks) * 100)

        emit_progress(upload_id, 'upload', progress, f'Uploaded chunk {chunk_index + 1}/{total_chunks}')

        # Check if all chunks received
        if received_chunks == total_chunks:
            try:
                # Reassemble file
                full_data = b''.join([
                    chunk_uploads[upload_id]['chunks'][i] 
                    for i in range(total_chunks)
                ])
                
                # Save assembled file
                file_path = os.path.join(chunk_uploads[upload_id]['chunk_dir'], filename)
                with open(file_path, 'wb') as f:
                    f.write(full_data)
                
                chunk_uploads[upload_id]['assembled_file'] = file_path
                emit_progress(upload_id, 'upload', 100, f'File {filename} ready for analysis')
                
                return jsonify({
                    'status': 'complete',
                    'message': 'File uploaded and ready for analysis',
                    'filename': filename,
                    'file_size': len(full_data)
                })
                
            except Exception as e:
                logger.error(f"Error assembling analytics file: {e}")
                emit_progress(upload_id, 'upload', -1, error=f'Failed to assemble file: {str(e)}')
                return jsonify({'error': f'Failed to assemble file: {str(e)}'}), 500
        
        return jsonify({
            'status': 'chunk_received',
            'chunk_index': chunk_index,
            'received_chunks': received_chunks,
            'total_chunks': total_chunks,
            'progress': progress
        })
        
    except Exception as e:
        logger.error(f"Error in analytics upload_chunk: {e}")
        if upload_id:
            emit_progress(upload_id, 'upload', -1, error=str(e))
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ANALYSIS ENDPOINTS
# ============================================================================

@bp.route('/analyze', methods=['POST'])
def analyze_data():
    """
    Perform statistical analysis on uploaded data
    """
    try:
        data = request.get_json()
        upload_id = data.get('uploadId')
        analysis_config = data.get('analysisConfig', {})
        
        if not upload_id or upload_id not in chunk_uploads:
            return jsonify({'error': 'Invalid upload ID'}), 400
            
        upload_info = chunk_uploads[upload_id]
        if 'assembled_file' not in upload_info:
            return jsonify({'error': 'File not yet assembled'}), 400
            
        file_path = upload_info['assembled_file']
        emit_progress(upload_id, 'analysis', 10, 'Loading data for analysis')
        
        # Load data
        df = pd.read_csv(file_path)
        emit_progress(upload_id, 'analysis', 30, 'Performing statistical analysis')
        
        # Basic statistics
        stats = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': {}
        }
        
        # Detailed statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats['numeric_summary'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75))
            }
        
        # Correlation analysis if requested
        correlations = {}
        if analysis_config.get('include_correlations', False) and len(numeric_cols) > 1:
            emit_progress(upload_id, 'analysis', 60, 'Computing correlations')
            correlation_matrix = df[numeric_cols].corr()
            correlations = correlation_matrix.to_dict()
        
        emit_progress(upload_id, 'analysis', 100, 'Analysis completed')
        
        return jsonify({
            'status': 'success',
            'statistics': stats,
            'correlations': correlations,
            'message': 'Statistical analysis completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_data: {e}")
        if upload_id:
            emit_progress(upload_id, 'analysis', -1, error=str(e))
        return jsonify({'error': str(e)}), 500

@bp.route('/interpolate', methods=['POST'])
def interpolate_data():
    """
    Perform data interpolation
    """
    try:
        data = request.get_json()
        upload_id = data.get('uploadId')
        interpolation_config = data.get('interpolationConfig', {})
        
        if not upload_id or upload_id not in chunk_uploads:
            return jsonify({'error': 'Invalid upload ID'}), 400
            
        upload_info = chunk_uploads[upload_id]
        if 'assembled_file' not in upload_info:
            return jsonify({'error': 'File not yet assembled'}), 400
            
        file_path = upload_info['assembled_file']
        emit_progress(upload_id, 'interpolation', 10, 'Loading data for interpolation')
        
        # Load data
        df = pd.read_csv(file_path)
        original_nulls = df.isnull().sum().sum()
        
        emit_progress(upload_id, 'interpolation', 30, 'Applying interpolation')
        
        # Apply interpolation method
        method = interpolation_config.get('method', 'linear')
        limit = interpolation_config.get('limit', None)
        
        if method == 'polynomial':
            degree = interpolation_config.get('degree', 2)
            df_interpolated = df.interpolate(method='polynomial', order=degree, limit=limit)
        elif method == 'spline':
            order = interpolation_config.get('order', 3)
            df_interpolated = df.interpolate(method='spline', order=order, limit=limit)
        else:  # linear, nearest, etc.
            df_interpolated = df.interpolate(method=method, limit=limit)
        
        final_nulls = df_interpolated.isnull().sum().sum()
        interpolated_values = original_nulls - final_nulls
        
        # Save interpolated data
        interpolated_filename = f"interpolated_{upload_info['filename']}"
        interpolated_path = os.path.join(upload_info['chunk_dir'], interpolated_filename)
        df_interpolated.to_csv(interpolated_path, index=False)
        
        emit_progress(upload_id, 'interpolation', 100, 'Interpolation completed')
        
        return jsonify({
            'status': 'success',
            'interpolated_file': interpolated_filename,
            'interpolated_path': interpolated_path,
            'original_missing': int(original_nulls),
            'final_missing': int(final_nulls),
            'interpolated_values': int(interpolated_values),
            'method': method,
            'message': 'Data interpolation completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in interpolate_data: {e}")
        if upload_id:
            emit_progress(upload_id, 'interpolation', -1, error=str(e))
        return jsonify({'error': str(e)}), 500

@bp.route('/visualize', methods=['POST'])
def create_visualization():
    """
    Generate data visualizations
    """
    try:
        data = request.get_json()
        upload_id = data.get('uploadId')
        viz_config = data.get('vizConfig', {})
        
        if not upload_id or upload_id not in chunk_uploads:
            return jsonify({'error': 'Invalid upload ID'}), 400
            
        upload_info = chunk_uploads[upload_id]
        if 'assembled_file' not in upload_info:
            return jsonify({'error': 'File not yet assembled'}), 400
            
        file_path = upload_info['assembled_file']
        emit_progress(upload_id, 'visualization', 10, 'Loading data for visualization')
        
        # Load data
        df = pd.read_csv(file_path)
        emit_progress(upload_id, 'visualization', 30, 'Creating visualizations')
        
        charts = {}
        chart_type = viz_config.get('chart_type', 'line')
        columns_to_plot = viz_config.get('columns', [])
        
        # Auto-select columns if not specified
        if not columns_to_plot:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            columns_to_plot = list(numeric_cols[:5])  # Limit to first 5 numeric columns
        
        # Create different chart types
        if chart_type == 'line':
            fig, ax = plt.subplots(figsize=(12, 6))
            for col in columns_to_plot:
                if col in df.columns:
                    ax.plot(df.index, df[col], label=col)
            ax.legend()
            ax.set_title('Line Chart')
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            charts['line_chart'] = generate_chart_base64(fig)
            
        elif chart_type == 'histogram':
            for i, col in enumerate(columns_to_plot):
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(df[col].dropna(), bins=30, alpha=0.7)
                    ax.set_title(f'Histogram - {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
                    charts[f'histogram_{col}'] = generate_chart_base64(fig)
                    
        elif chart_type == 'scatter':
            if len(columns_to_plot) >= 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                x_col, y_col = columns_to_plot[0], columns_to_plot[1]
                if x_col in df.columns and y_col in df.columns:
                    ax.scatter(df[x_col], df[y_col], alpha=0.6)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
                    charts['scatter_plot'] = generate_chart_base64(fig)
        
        # Correlation heatmap for numeric data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = df[numeric_cols].corr()
            im = ax.imshow(correlation_matrix.values, cmap='coolwarm', aspect='auto')
            ax.set_xticks(range(len(correlation_matrix.columns)))
            ax.set_yticks(range(len(correlation_matrix.columns)))
            ax.set_xticklabels(correlation_matrix.columns, rotation=45)
            ax.set_yticklabels(correlation_matrix.columns)
            ax.set_title('Correlation Heatmap')
            
            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label('Correlation Coefficient')
            charts['correlation_heatmap'] = generate_chart_base64(fig)
        
        emit_progress(upload_id, 'visualization', 100, 'Visualizations created')
        
        return jsonify({
            'status': 'success',
            'charts': charts,
            'chart_count': len(charts),
            'message': 'Visualizations created successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in create_visualization: {e}")
        if upload_id:
            emit_progress(upload_id, 'visualization', -1, error=str(e))
        return jsonify({'error': str(e)}), 500

@bp.route('/cloud-process', methods=['POST'])
def cloud_process():
    """
    Perform comprehensive cloud-based data processing
    """
    try:
        data = request.get_json()
        upload_id = data.get('uploadId')
        process_config = data.get('processConfig', {})
        
        if not upload_id or upload_id not in chunk_uploads:
            return jsonify({'error': 'Invalid upload ID'}), 400
            
        upload_info = chunk_uploads[upload_id]
        if 'assembled_file' not in upload_info:
            return jsonify({'error': 'File not yet assembled'}), 400
            
        file_path = upload_info['assembled_file']
        emit_progress(upload_id, 'cloud_processing', 10, 'Starting cloud processing')
        
        # Load data
        df = pd.read_csv(file_path)
        results = {
            'data_summary': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
        }
        
        emit_progress(upload_id, 'cloud_processing', 40, 'Performing advanced analytics')
        
        # Advanced analytics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Trend analysis using linear regression
            trends = {}
            for col in numeric_cols:
                if df[col].notna().sum() > 1:
                    x = np.arange(len(df)).reshape(-1, 1)
                    y = df[col].fillna(df[col].mean())
                    
                    model = LinearRegression()
                    model.fit(x, y)
                    
                    trends[col] = {
                        'slope': float(model.coef_[0]),
                        'intercept': float(model.intercept_),
                        'r_squared': float(model.score(x, y))
                    }
            
            results['trend_analysis'] = trends
        
        emit_progress(upload_id, 'cloud_processing', 70, 'Generating insights')
        
        # Generate insights
        insights = []
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                cv = col_data.std() / col_data.mean() if col_data.mean() != 0 else 0
                if cv > 1:
                    insights.append(f"Column '{col}' shows high variability (CV={cv:.2f})")
                
                # Check for potential outliers
                q1, q3 = col_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = col_data[(col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)]
                if len(outliers) > 0:
                    outlier_pct = (len(outliers) / len(col_data)) * 100
                    insights.append(f"Column '{col}' has {len(outliers)} potential outliers ({outlier_pct:.1f}%)")
        
        results['insights'] = insights
        
        emit_progress(upload_id, 'cloud_processing', 100, 'Cloud processing completed')
        
        return jsonify({
            'status': 'success',
            'results': results,
            'message': 'Cloud processing completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in cloud_process: {e}")
        if upload_id:
            emit_progress(upload_id, 'cloud_processing', -1, error=str(e))
        return jsonify({'error': str(e)}), 500

# ============================================================================
# FILE DOWNLOAD ENDPOINTS
# ============================================================================

@bp.route('/prepare-save', methods=['POST'])
def prepare_save():
    """Prepare analysis results for download"""
    try:
        data = request.get_json()
        upload_id = data.get('uploadId')
        filename = data.get('filename')
        
        if not upload_id or upload_id not in chunk_uploads:
            return jsonify({'error': 'Invalid upload ID'}), 400
            
        upload_info = chunk_uploads[upload_id]
        file_path = os.path.join(upload_info['chunk_dir'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        # Generate download ID
        download_id = secrets.token_urlsafe(16)
        
        # Store file info for download
        with storage_lock:
            temp_files[download_id] = {
                'path': file_path,
                'filename': filename,
                'created_at': time.time()
            }
        
        return jsonify({
            'status': 'success',
            'download_id': download_id,
            'filename': filename,
            'message': 'Analysis results prepared for download'
        })
        
    except Exception as e:
        logger.error(f"Error preparing analytics save: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/download/<file_id>', methods=['GET'])
def download_file(file_id):
    """Download analysis results"""
    try:
        if file_id not in temp_files:
            return jsonify({'error': 'Invalid file ID or file expired'}), 404
            
        file_info = temp_files[file_id]
        file_path = file_info['path']
        filename = file_info['filename']
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        def cleanup_after_download():
            time.sleep(5)
            with storage_lock:
                if file_id in temp_files:
                    del temp_files[file_id]
        
        threading.Thread(target=cleanup_after_download, daemon=True).start()
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Error downloading analytics file: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/chart/<chart_id>', methods=['GET'])
def get_chart(chart_id):
    """Get generated chart by ID"""
    try:
        # This would typically retrieve from a chart storage system
        # For now, return a placeholder response
        return jsonify({
            'error': 'Chart retrieval not yet implemented',
            'chart_id': chart_id
        }), 501
        
    except Exception as e:
        logger.error(f"Error retrieving chart: {e}")
        return jsonify({'error': str(e)}), 500