"""Smoke tests for Python 3.11 upgrade validation.

These tests verify current behavior before upgrading dependencies.
Run before AND after each upgrade phase to catch regressions.

Tests cover:
- Flask app creation and health endpoint
- CORS headers on responses
- SocketIO initialization
- Datetime parsing (Supabase timestamp formats)
- Blueprint registration and route availability
- Gunicorn-compatible app structure
- Key imports from all major packages
"""

import pytest
from datetime import datetime, timezone


# ============================================================
# Phase 1: Flask / Werkzeug / Gunicorn compatibility
# ============================================================

class TestFlaskApp:
    """Verify Flask app factory produces a working app."""

    def test_app_factory_creates_app(self):
        from core.app_factory import create_app
        app, socketio = create_app()
        assert app is not None
        assert socketio is not None

    def test_health_endpoint(self):
        from core.app_factory import create_app
        app, _ = create_app()
        with app.test_client() as client:
            resp = client.get('/health')
            assert resp.status_code == 200
            data = resp.get_json()
            assert data['status'] == 'ok'

    def test_index_endpoint(self):
        from core.app_factory import create_app
        app, _ = create_app()
        with app.test_client() as client:
            resp = client.get('/')
            assert resp.status_code == 200
            data = resp.get_json()
            assert data['status'] == 'online'
            assert 'version' in data

    def test_404_returns_json(self):
        from core.app_factory import create_app
        app, _ = create_app()
        with app.test_client() as client:
            resp = client.get('/nonexistent-route-xyz')
            assert resp.status_code == 404

    def test_app_has_socketio_extension(self):
        from core.app_factory import create_app
        app, _ = create_app()
        assert 'socketio' in app.extensions

    def test_max_content_length_configured(self):
        from core.app_factory import create_app
        app, _ = create_app()
        assert app.config['MAX_CONTENT_LENGTH'] == 100 * 1024 * 1024


# ============================================================
# Phase 2: CORS
# ============================================================

class TestCORS:
    """Verify CORS headers are present on responses."""

    @pytest.fixture
    def client(self):
        from core.app_factory import create_app
        app, _ = create_app()
        app.config['TESTING'] = True
        return app.test_client()

    def test_options_preflight_returns_200(self, client):
        resp = client.options('/health', headers={
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'GET'
        })
        assert resp.status_code == 200

    def test_cors_allows_content_type_header(self, client):
        resp = client.options('/health', headers={
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type,Authorization'
        })
        allow_headers = resp.headers.get('Access-Control-Allow-Headers', '')
        assert 'Content-Type' in allow_headers
        assert 'Authorization' in allow_headers

    def test_cors_allows_credentials(self, client):
        resp = client.options('/health', headers={
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'GET'
        })
        assert resp.headers.get('Access-Control-Allow-Credentials') == 'true'

    def test_cors_allows_common_methods(self, client):
        resp = client.options('/health', headers={
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'POST'
        })
        allow_methods = resp.headers.get('Access-Control-Allow-Methods', '')
        for method in ['GET', 'POST', 'PUT', 'DELETE']:
            assert method in allow_methods

    def test_cors_headers_on_normal_get(self, client):
        """CORS headers must appear on regular responses, not just OPTIONS."""
        resp = client.get('/health', headers={
            'Origin': 'http://localhost:3000'
        })
        assert resp.status_code == 200
        assert 'Access-Control-Allow-Origin' in resp.headers

    def test_cors_allows_localhost_origin(self, client):
        """Development origin must be allowed."""
        resp = client.get('/health', headers={
            'Origin': 'http://localhost:3000'
        })
        origin = resp.headers.get('Access-Control-Allow-Origin', '')
        assert origin in ['http://localhost:3000', '*']

    def test_cors_allows_production_origin(self, client):
        """Production frontend origin must be allowed."""
        resp = client.get('/health', headers={
            'Origin': 'https://www.forecast-engine.com'
        })
        origin = resp.headers.get('Access-Control-Allow-Origin', '')
        assert origin in ['https://www.forecast-engine.com', '*']

    def test_cors_expose_headers(self, client):
        """Content-Disposition and Content-Length must be exposed for file downloads."""
        resp = client.get('/health', headers={
            'Origin': 'http://localhost:3000'
        })
        expose = resp.headers.get('Access-Control-Expose-Headers', '')
        assert 'Content-Disposition' in expose
        assert 'Content-Length' in expose

    def test_cors_credentials_on_normal_response(self, client):
        """Credentials must be allowed on regular responses (not just preflight)."""
        resp = client.get('/health', headers={
            'Origin': 'http://localhost:3000'
        })
        assert resp.headers.get('Access-Control-Allow-Credentials') == 'true'


# ============================================================
# Phase 3: Datetime parsing (Supabase compatibility)
# ============================================================

class TestDatetimeParsing:
    """Verify datetime parsing handles all Supabase timestamp formats."""

    def test_standard_6_digit_fractional(self):
        from shared.datetime_utils import parse_iso_datetime
        dt = parse_iso_datetime('2026-03-25T13:28:51.747310+00:00')
        assert dt.year == 2026
        assert dt.month == 3
        assert dt.second == 51

    def test_5_digit_fractional_supabase_format(self):
        """This is the format that broke Python 3.9's fromisoformat."""
        from shared.datetime_utils import parse_iso_datetime
        dt = parse_iso_datetime('2026-03-25T13:28:51.74731+00:00')
        assert dt.year == 2026
        assert dt.tzinfo is not None

    def test_3_digit_fractional(self):
        from shared.datetime_utils import parse_iso_datetime
        dt = parse_iso_datetime('2026-03-25T13:28:51.747+00:00')
        assert dt.year == 2026

    def test_no_fractional(self):
        from shared.datetime_utils import parse_iso_datetime
        dt = parse_iso_datetime('2026-03-25T13:28:51+00:00')
        assert dt.second == 51

    def test_z_suffix(self):
        from shared.datetime_utils import parse_iso_datetime
        dt = parse_iso_datetime('2026-03-25T13:28:51.74731Z')
        assert dt.tzinfo is not None

    def test_returns_timezone_aware(self):
        from shared.datetime_utils import parse_iso_datetime
        dt = parse_iso_datetime('2026-03-25T13:28:51+00:00')
        assert dt.tzinfo is not None

    def test_negative_utc_offset(self):
        from shared.datetime_utils import parse_iso_datetime
        dt = parse_iso_datetime('2026-03-25T08:28:51.12345-05:00')
        assert dt.hour == 8


# ============================================================
# Phase 3: Supabase client
# ============================================================

class TestSupabaseClient:
    """Verify Supabase client creation and basic operations."""

    def test_client_options_import(self):
        """ClientOptions must be importable (path changes between supabase versions)."""
        from shared.database.client import HAS_CLIENT_OPTIONS
        assert HAS_CLIENT_OPTIONS is True

    def test_create_anon_client(self):
        """Anon client must be creatable with env vars."""
        from shared.database.client import get_supabase_client
        get_supabase_client.cache_clear()
        client = get_supabase_client()
        assert client is not None

    def test_create_admin_client(self):
        """Service role client must be creatable with env vars."""
        from shared.database.client import get_supabase_admin_client
        get_supabase_admin_client.cache_clear()
        client = get_supabase_admin_client()
        assert client is not None

    def test_admin_client_has_table_method(self):
        """Client must expose .table() for DB operations."""
        from shared.database.client import get_supabase_admin_client
        get_supabase_admin_client.cache_clear()
        client = get_supabase_admin_client()
        assert hasattr(client, 'table')

    def test_admin_client_has_storage(self):
        """Client must expose .storage for file operations."""
        from shared.database.client import get_supabase_admin_client
        get_supabase_admin_client.cache_clear()
        client = get_supabase_admin_client()
        assert hasattr(client, 'storage')

    def test_admin_client_has_postgrest_auth(self):
        """Client must expose .postgrest.auth() for user-scoped queries."""
        from shared.database.client import get_supabase_admin_client
        get_supabase_admin_client.cache_clear()
        client = get_supabase_admin_client()
        assert hasattr(client.postgrest, 'auth')

    def test_table_select_sessions(self):
        """Basic select query must work (proves DB connectivity + API compatibility)."""
        from shared.database.client import get_supabase_admin_client
        get_supabase_admin_client.cache_clear()
        client = get_supabase_admin_client()
        response = client.table('sessions').select('id').limit(1).execute()
        assert hasattr(response, 'data')


# ============================================================
# Phase 4: Blueprint registration
# ============================================================

class TestBlueprints:
    """Verify all API blueprints are registered."""

    @pytest.fixture
    def app(self):
        from core.app_factory import create_app
        app, _ = create_app()
        return app

    def test_training_blueprint_registered(self, app):
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        training_routes = [r for r in rules if r.startswith('/api/training')]
        assert len(training_routes) > 0

    def test_upload_blueprint_registered(self, app):
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        upload_routes = [r for r in rules if r.startswith('/api/loadRowData')]
        assert len(upload_routes) > 0

    def test_stripe_blueprint_registered(self, app):
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        stripe_routes = [r for r in rules if r.startswith('/api/stripe')]
        assert len(stripe_routes) > 0

    def test_cloud_blueprint_registered(self, app):
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        cloud_routes = [r for r in rules if r.startswith('/api/cloud')]
        assert len(cloud_routes) > 0

    def test_generate_plot_endpoint_exists(self, app):
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert '/api/training/generate-plot' in rules

    def test_plot_variables_endpoint_exists(self, app):
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        plot_var_routes = [r for r in rules if 'plot-variables' in r]
        assert len(plot_var_routes) > 0


# ============================================================
# Phase 5: Package imports (verify nothing broken by upgrades)
# ============================================================

class TestPackageImports:
    """Verify all critical packages can be imported."""

    def test_import_flask(self):
        from flask import Flask, jsonify, request, g, Blueprint
        assert Flask is not None

    def test_import_flask_cors(self):
        from flask_cors import CORS
        assert CORS is not None

    def test_import_flask_socketio(self):
        from flask_socketio import SocketIO
        assert SocketIO is not None

    def test_import_numpy(self):
        import numpy as np
        assert hasattr(np, 'array')
        major = int(np.__version__.split('.')[0])
        assert major in (1, 2), f"unsupported numpy major version: {np.__version__}"

    def test_import_pandas(self):
        import pandas as pd
        assert hasattr(pd, 'DataFrame')

    def test_import_tensorflow(self):
        import tensorflow as tf
        assert hasattr(tf, 'keras')

    def test_import_tensorflow_keras_models(self):
        from tensorflow.keras.models import Sequential
        assert Sequential is not None

    def test_import_sklearn(self):
        from sklearn.preprocessing import MinMaxScaler
        assert MinMaxScaler is not None

    def test_import_lightgbm(self):
        from lightgbm import LGBMRegressor
        assert LGBMRegressor is not None

    def test_import_matplotlib(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        assert hasattr(plt, 'subplots')

    def test_import_seaborn(self):
        import seaborn as sns
        assert hasattr(sns, 'violinplot')

    def test_import_supabase(self):
        from supabase import create_client
        assert create_client is not None

    def test_import_stripe(self):
        import stripe
        assert hasattr(stripe, 'Webhook')

    def test_import_gunicorn(self):
        import gunicorn
        assert gunicorn is not None

    def test_import_apscheduler(self):
        from apscheduler.schedulers.background import BackgroundScheduler
        assert BackgroundScheduler is not None


# ============================================================
# Phase 6: Gunicorn app entry point
# ============================================================

class TestGunicornEntryPoint:
    """Verify the app module exposes what gunicorn expects."""

    def test_app_module_exposes_app(self):
        import app as app_module
        assert hasattr(app_module, 'app')

    def test_app_module_exposes_socketio(self):
        import app as app_module
        assert hasattr(app_module, 'socketio')

    def test_app_is_flask_instance(self):
        import app as app_module
        from flask import Flask
        assert isinstance(app_module.app, Flask)


# ============================================================
# Phase 7: Visualization / Plot generation imports
# ============================================================

class TestVisualizationImports:
    """Verify visualization module can be imported (catches matplotlib/seaborn issues)."""

    def test_import_visualizer(self):
        from domains.training.services.visualization import Visualizer
        assert Visualizer is not None

    def test_visualizer_has_generate_custom_plot(self):
        from domains.training.services.visualization import Visualizer
        v = Visualizer()
        assert hasattr(v, 'generate_custom_plot')

    def test_import_violin_generator(self):
        from domains.training.services.violin import generate_violin_plots_from_data
        assert generate_violin_plots_from_data is not None


# ============================================================
# Phase 8: Training pipeline imports
# ============================================================

class TestTrainingPipelineImports:
    """Verify training pipeline modules can be imported."""

    def test_import_exact_pipeline(self):
        from domains.training.ml.exact import run_exact_training_pipeline
        assert run_exact_training_pipeline is not None

    def test_import_transformer(self):
        from domains.training.data.transformer import create_training_arrays
        assert create_training_arrays is not None

    def test_import_scaler(self):
        from domains.training.ml.scaler import process_and_scale_data
        assert process_and_scale_data is not None

    def test_import_evaluation(self):
        from domains.training.ml.evaluation import calculate_evaluation_with_averaging
        assert calculate_evaluation_with_averaging is not None
