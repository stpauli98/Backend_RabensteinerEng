"""
Backward Compatibility Layer

This module provides URL redirects and compatibility endpoints to ensure
existing API clients continue to work during the transition to consolidated blueprints.

Legacy URL patterns are mapped to new consolidated endpoints with proper redirects.
"""

import logging
from flask import Blueprint, redirect, url_for, request, jsonify

logger = logging.getLogger(__name__)

# Create backward compatibility blueprint
bp = Blueprint('backward_compatibility', __name__)

# ============================================================================
# LEGACY URL MAPPINGS
# ============================================================================

LEGACY_MAPPINGS = {
    # firstProcessing -> data_pipeline
    '/api/firstProcessing/upload_chunk': '/api/data/upload-chunk',
    '/api/firstProcessing/prepare-save': '/api/data/prepare-save', 
    '/api/firstProcessing/cancel-upload': '/api/data/cancel-upload',
    '/api/firstProcessing/download/<file_id>': '/api/data/download/<file_id>',
    
    # load_row_data -> data_pipeline
    '/api/loadRowData/upload-chunk': '/api/data/upload-chunk',
    '/api/loadRowData/finalize-upload': '/api/data/finalize-upload',
    '/api/loadRowData/cancel-upload': '/api/data/cancel-upload',
    '/api/loadRowData/upload-status/<upload_id>': '/api/data/upload-status/<upload_id>',
    '/api/loadRowData/prepare-save': '/api/data/prepare-save',
    '/api/loadRowData/merge-and-prepare': '/api/data/prepare-save',
    '/api/loadRowData/prepare-download/<file_id>': '/api/data/download/<file_id>',
    
    # data_processing_main -> data_pipeline  
    '/api/dataProcessingMain/upload-chunk': '/api/data/upload-chunk',
    '/api/dataProcessingMain/prepare-save': '/api/data/prepare-save',
    '/api/dataProcessingMain/download/<file_id>': '/api/data/download/<file_id>',
    
    # adjustmentsOfData -> data_pipeline
    '/api/adjustmentsOfData/upload-chunk': '/api/data/upload-chunk',
    '/api/adjustmentsOfData/adjust-data-chunk': '/api/data/adjust',
    '/api/adjustmentsOfData/adjustdata/complete': '/api/data/adjust',
    '/api/adjustmentsOfData/prepare-save': '/api/data/prepare-save',
    '/api/adjustmentsOfData/download/<file_id>': '/api/data/download/<file_id>',
    
    # cloud -> analytics
    '/api/cloud/upload-chunk': '/api/analytics/upload-chunk',
    '/api/cloud/complete': '/api/analytics/analyze',
    '/api/cloud/clouddata': '/api/analytics/cloud-process',
    '/api/cloud/interpolate-chunked': '/api/analytics/interpolate',
    '/api/cloud/prepare-save': '/api/analytics/prepare-save',
    '/api/cloud/download/<file_id>': '/api/analytics/download/<file_id>',
    
    # training -> machine_learning
    '/api/training/upload-chunk': '/api/ml/upload-data',
    '/api/training/finalize-session': '/api/ml/train',
    '/api/training/list-sessions': '/api/ml/sessions',
    '/api/training/session/<session_id>': '/api/ml/session/<session_id>',
    '/api/training/session-status/<session_id>': '/api/ml/status/<session_id>',
    '/api/training/get-file-metadata/<session_id>': '/api/ml/status/<session_id>',
    '/api/training/init-session': '/api/ml/init-session',
    '/api/training/get-all-files-metadata/<session_id>': '/api/ml/status/<session_id>',
    
    # training_system endpoints -> machine_learning
    '/api/training/results/<session_id>': '/api/ml/results/<session_id>',
    '/api/training/status/<session_id>': '/api/ml/status/<session_id>',
    '/api/training/progress/<session_id>': '/api/ml/status/<session_id>',
    '/api/training/visualizations/<session_id>': '/api/ml/visualizations/<session_id>',
    '/api/training/metrics/<session_id>': '/api/ml/metrics/<session_id>',
    '/api/training/logs/<session_id>': '/api/ml/logs/<session_id>',
    '/api/training/cancel/<session_id>': '/api/ml/cancel/<session_id>',
    '/api/training/generate-plot': '/api/analytics/visualize',
    '/api/training/plot-variables/<session_id>': '/api/ml/visualizations/<session_id>',
    
    # health -> system
    '/health': '/api/system/health'
}

# ============================================================================
# COMPATIBILITY ROUTE HANDLERS
# ============================================================================

def create_compatibility_routes():
    """Create compatibility routes for all legacy endpoints"""
    from flask import redirect
    
    # Create individual route handlers for each legacy endpoint
    legacy_routes = []
    
    for legacy_url, new_url in LEGACY_MAPPINGS.items():
        # Create a closure to capture the URLs
        def make_redirect_handler(legacy_path, new_path):
            def redirect_handler(*args, **kwargs):
                logger.info(f"Legacy URL accessed: {legacy_path} -> redirecting to {new_path}")
                
                # Handle URL parameters in the redirect
                if kwargs:
                    # Replace URL parameters in new_path
                    formatted_new_path = new_path
                    if '<' in new_path:
                        # Extract parameter names and values
                        for key, value in kwargs.items():
                            formatted_new_path = formatted_new_path.replace(f'<{key}>', str(value))
                    
                    return redirect(formatted_new_path, code=301)
                else:
                    return redirect(new_path, code=301)
            
            return redirect_handler
        
        # Store route info for registration
        endpoint_name = legacy_url.replace("/", "_").replace("<", "_").replace(">", "_").replace("-", "_")
        legacy_routes.append({
            'rule': legacy_url,
            'endpoint': f'legacy{endpoint_name}',
            'handler': make_redirect_handler(legacy_url, new_url),
            'methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH']
        })
    
    return legacy_routes

# ============================================================================
# MIGRATION STATUS ENDPOINTS
# ============================================================================

@bp.route('/api/migration/status', methods=['GET'])
def migration_status():
    """Get migration status and available endpoints"""
    try:
        status = {
            'migration_active': True,
            'legacy_endpoints_supported': True,
            'new_api_structure': {
                'data_pipeline': '/api/data/*',
                'analytics': '/api/analytics/*',
                'machine_learning': '/api/ml/*', 
                'system': '/api/system/*'
            },
            'legacy_mappings_count': len(LEGACY_MAPPINGS),
            'deprecation_notice': 'Legacy endpoints are deprecated and will be removed in a future version. Please update to use new API structure.',
            'migration_guide': '/api/migration/guide'
        }
        
        return jsonify({
            'status': 'success',
            'migration_status': status
        })
        
    except Exception as e:
        logger.error(f"Error getting migration status: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/api/migration/guide', methods=['GET'])
def migration_guide():
    """Get migration guide for updating to new API structure"""
    try:
        guide = {
            'overview': 'API endpoints have been consolidated into 4 logical domains for better organization and maintainability.',
            'new_structure': {
                'data_pipeline': {
                    'base_url': '/api/data',
                    'description': 'All data processing, loading, transformations, and cleaning operations',
                    'key_endpoints': [
                        'POST /api/data/upload-chunk - File chunk uploads',
                        'POST /api/data/process - Data processing with resampling', 
                        'POST /api/data/clean - Advanced cleaning and filtering',
                        'POST /api/data/adjust - Data adjustments',
                        'GET /api/data/download/<file_id> - Download processed files'
                    ]
                },
                'analytics': {
                    'base_url': '/api/analytics',
                    'description': 'Statistical analysis, cloud-based processing, and visualization',
                    'key_endpoints': [
                        'POST /api/analytics/analyze - Statistical analysis',
                        'POST /api/analytics/interpolate - Data interpolation',
                        'POST /api/analytics/visualize - Generate charts and plots',
                        'POST /api/analytics/cloud-process - Cloud-based analysis'
                    ]
                },
                'machine_learning': {
                    'base_url': '/api/ml',
                    'description': 'ML model training, management, and training visualizations',
                    'key_endpoints': [
                        'POST /api/ml/upload-data - Upload training data',
                        'POST /api/ml/train - Start training',
                        'GET /api/ml/status/<session_id> - Training status',
                        'GET /api/ml/results/<session_id> - Training results',
                        'GET /api/ml/visualizations/<session_id> - Training plots'
                    ]
                },
                'system': {
                    'base_url': '/api/system',
                    'description': 'Health checks, session management, system utilities',
                    'key_endpoints': [
                        'GET /api/system/health - Health check',
                        'GET /api/system/sessions - List all sessions',
                        'POST /api/system/cleanup - Manual cleanup',
                        'GET /api/system/status - System status'
                    ]
                }
            },
            'migration_steps': [
                '1. Review the new API structure and identify equivalent endpoints',
                '2. Update client code to use new endpoint URLs',
                '3. Test functionality with new endpoints',
                '4. Remove references to legacy endpoints',
                '5. Update documentation and API client libraries'
            ],
            'legacy_support': {
                'status': 'Active - Legacy endpoints redirect to new endpoints',
                'deprecation_timeline': 'Legacy support will be removed in version 2.0.0',
                'recommendation': 'Migrate to new API structure as soon as possible'
            },
            'breaking_changes': [
                'URL paths have changed to follow new domain structure',
                'Some endpoint names have been standardized for consistency',
                'Response formats remain the same for compatibility'
            ]
        }
        
        return jsonify({
            'status': 'success',
            'migration_guide': guide
        })
        
    except Exception as e:
        logger.error(f"Error getting migration guide: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/api/migration/mappings', methods=['GET'])
def endpoint_mappings():
    """Get detailed endpoint mappings from legacy to new structure"""
    try:
        mappings = {
            'total_mappings': len(LEGACY_MAPPINGS),
            'mappings': []
        }
        
        for legacy_url, new_url in LEGACY_MAPPINGS.items():
            mappings['mappings'].append({
                'legacy_endpoint': legacy_url,
                'new_endpoint': new_url,
                'status': 'redirect_active',
                'http_status': 301
            })
        
        return jsonify({
            'status': 'success',
            'endpoint_mappings': mappings
        })
        
    except Exception as e:
        logger.error(f"Error getting endpoint mappings: {e}")
        return jsonify({'error': str(e)}), 500

# Register compatibility routes when module is imported
COMPATIBILITY_ROUTES = create_compatibility_routes()