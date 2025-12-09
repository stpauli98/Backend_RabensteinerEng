"""Example protected routes demonstrating auth and subscription middleware"""
from flask import Blueprint, jsonify, g
from shared.auth.jwt import require_auth, optional_auth
from shared.auth.subscription import (
    require_subscription,
    check_upload_limit,
    check_processing_limit,
    check_storage_limit,
    get_user_usage
)
from shared.tracking.usage import increment_upload_count

auth_example_bp = Blueprint('auth_example', __name__, url_prefix='/api/auth-example')


@auth_example_bp.route('/public', methods=['GET'])
def public_route():
    """Public route - no authentication required"""
    return jsonify({
        'message': 'This is a public route',
        'authenticated': False
    })


@auth_example_bp.route('/optional', methods=['GET'])
@optional_auth
def optional_route():
    """Route with optional authentication"""
    if hasattr(g, 'user_id'):
        return jsonify({
            'message': f'Hello {g.user_email}',
            'authenticated': True,
            'user_id': g.user_id
        })
    else:
        return jsonify({
            'message': 'Hello anonymous user',
            'authenticated': False
        })


@auth_example_bp.route('/protected', methods=['GET'])
@require_auth
def protected_route():
    """Protected route - requires authentication"""
    return jsonify({
        'message': f'Hello {g.user_email}',
        'user_id': g.user_id,
        'authenticated': True
    })


@auth_example_bp.route('/profile', methods=['GET'])
@require_auth
@require_subscription
def profile_route():
    """Protected route with subscription - shows user plan and usage"""
    usage = get_user_usage(g.user_id, g.access_token)

    return jsonify({
        'user': {
            'id': g.user_id,
            'email': g.user_email
        },
        'subscription': {
            'plan': g.plan.get('name'),
            'status': g.subscription.get('status'),
            'billing_cycle': g.subscription.get('billing_cycle'),
            'started_at': g.subscription.get('started_at'),
            'expires_at': g.subscription.get('expires_at')
        },
        'usage': usage,
        'limits': {
            'uploads_per_month': g.plan.get('uploads_per_month'),
            'processing_per_month': g.plan.get('processing_per_month'),
            'storage_limit_mb': g.plan.get('storage_limit_mb')
        }
    })


@auth_example_bp.route('/upload-check', methods=['POST'])
@require_auth
@require_subscription
@check_upload_limit
def upload_check():
    """Example upload endpoint with limit checking"""
    increment_upload_count(g.user_id)

    return jsonify({
        'message': 'Upload successful',
        'uploads_remaining': g.uploads_remaining - 1,
        'plan': g.plan.get('name')
    })


@auth_example_bp.route('/process-check', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def process_check():
    """Example processing endpoint with limit checking"""
    return jsonify({
        'message': 'Processing can proceed',
        'processing_remaining': g.processing_remaining,
        'plan': g.plan.get('name')
    })


@auth_example_bp.route('/storage-check', methods=['POST'])
@require_auth
@require_subscription
@check_storage_limit
def storage_check():
    """Example storage endpoint with limit checking"""
    return jsonify({
        'message': 'Storage available',
        'storage_remaining_mb': g.storage_remaining_mb,
        'plan': g.plan.get('name')
    })


@auth_example_bp.route('/full-check', methods=['POST'])
@require_auth
@require_subscription
@check_upload_limit
@check_processing_limit
@check_storage_limit
def full_check():
    """Example endpoint with all limit checks"""
    return jsonify({
        'message': 'All checks passed',
        'plan': g.plan.get('name'),
        'limits': {
            'uploads_remaining': g.uploads_remaining,
            'processing_remaining': g.processing_remaining,
            'storage_remaining_mb': g.storage_remaining_mb
        }
    })
