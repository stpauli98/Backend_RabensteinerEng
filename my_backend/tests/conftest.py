"""Pytest configuration for backend tests."""
import os
import sys

# Ensure the backend root is on sys.path so `from domains...` and `from shared...` work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test environment defaults — overridden by individual tests as needed.
os.environ.setdefault('STRIPE_SECRET_KEY', 'sk_test_dummy_for_tests')
os.environ.setdefault('STRIPE_WEBHOOK_SECRET', 'whsec_dummy_for_tests')
os.environ.setdefault('STRIPE_PUBLISHABLE_KEY', 'pk_test_dummy_for_tests')
os.environ.setdefault('SUPABASE_URL', 'https://example.supabase.co')
os.environ.setdefault('SUPABASE_KEY', 'test_anon_key')
os.environ.setdefault('SUPABASE_SERVICE_ROLE_KEY', 'test_service_role_key')
os.environ.setdefault('FRONTEND_URL', 'http://localhost:3000')
