#!/usr/bin/env python3
"""
Test script for backend auth integration
Tests all auth and subscription middleware endpoints
"""
import os
import sys
import requests
import json
from supabase import create_client

# Supabase config
SUPABASE_URL = "https://luvjebsltuttakatnzaa.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx1dmplYnNsdHV0dGFrYXRuemFhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDkwMzcyMjEsImV4cCI6MjA2NDYxMzIyMX0.IZqPkAkUu0paFqRP8a6BLXk-K9h183wmX9QtdVRACEY"

# Backend URL
BACKEND_URL = "http://localhost:8080"

# Test credentials
TEST_EMAIL = "test@rabensteiner.com"
TEST_PASSWORD = "TestPassword123!"

def print_section(title):
    """Print section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

def print_result(test_name, success, response=None, error=None):
    """Print test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {test_name}")
    if response:
        print(f"   Response: {json.dumps(response, indent=2)}")
    if error:
        print(f"   Error: {error}")
    print()

def get_auth_token():
    """Login and get JWT token"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        response = supabase.auth.sign_in_with_password({
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        })

        if response.session:
            return response.session.access_token
        else:
            print("❌ Failed to get session")
            return None
    except Exception as e:
        print(f"❌ Login error: {str(e)}")
        return None

def test_public_route():
    """Test public route (no auth)"""
    print_section("Test 1: Public Route (No Auth)")

    try:
        response = requests.get(f"{BACKEND_URL}/api/auth-example/public")
        data = response.json()

        success = response.status_code == 200 and data.get('authenticated') == False
        print_result("Public route accessible without auth", success, data)
        return success
    except Exception as e:
        print_result("Public route", False, error=str(e))
        return False

def test_optional_route_without_auth():
    """Test optional auth route without token"""
    print_section("Test 2: Optional Auth Route (No Token)")

    try:
        response = requests.get(f"{BACKEND_URL}/api/auth-example/optional")
        data = response.json()

        success = response.status_code == 200 and data.get('authenticated') == False
        print_result("Optional route works without auth", success, data)
        return success
    except Exception as e:
        print_result("Optional route (no auth)", False, error=str(e))
        return False

def test_optional_route_with_auth(token):
    """Test optional auth route with token"""
    print_section("Test 3: Optional Auth Route (With Token)")

    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{BACKEND_URL}/api/auth-example/optional", headers=headers)
        data = response.json()

        success = response.status_code == 200 and data.get('authenticated') == True
        print_result("Optional route recognizes authenticated user", success, data)
        return success
    except Exception as e:
        print_result("Optional route (with auth)", False, error=str(e))
        return False

def test_protected_route_without_auth():
    """Test protected route without token (should fail)"""
    print_section("Test 4: Protected Route (No Token - Should Fail)")

    try:
        response = requests.get(f"{BACKEND_URL}/api/auth-example/protected")
        data = response.json()

        success = response.status_code == 401 and 'error' in data
        print_result("Protected route blocks unauthenticated users", success, data)
        return success
    except Exception as e:
        print_result("Protected route (no auth)", False, error=str(e))
        return False

def test_protected_route_with_auth(token):
    """Test protected route with valid token"""
    print_section("Test 5: Protected Route (With Token)")

    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{BACKEND_URL}/api/auth-example/protected", headers=headers)
        data = response.json()

        success = response.status_code == 200 and data.get('authenticated') == True
        print_result("Protected route allows authenticated users", success, data)
        return success
    except Exception as e:
        print_result("Protected route (with auth)", False, error=str(e))
        return False

def test_profile_route(token):
    """Test profile route (requires subscription)"""
    print_section("Test 6: Profile Route (Auth + Subscription)")

    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{BACKEND_URL}/api/auth-example/profile", headers=headers)
        data = response.json()

        success = response.status_code == 200 and 'subscription' in data and 'usage' in data
        print_result("Profile route shows subscription and usage", success, data)
        return success
    except Exception as e:
        print_result("Profile route", False, error=str(e))
        return False

def test_upload_limit_check(token):
    """Test upload limit check"""
    print_section("Test 7: Upload Limit Check")

    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(f"{BACKEND_URL}/api/auth-example/upload-check", headers=headers)
        data = response.json()

        # Should succeed if under limit, or fail with 403 if at limit
        success = response.status_code in [200, 403]

        if response.status_code == 200:
            print_result("Upload limit check passed (under limit)", success, data)
        else:
            print_result("Upload limit reached (expected behavior)", success, data)

        return success
    except Exception as e:
        print_result("Upload limit check", False, error=str(e))
        return False

def test_processing_limit_check(token):
    """Test processing limit check"""
    print_section("Test 8: Processing Limit Check")

    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(f"{BACKEND_URL}/api/auth-example/process-check", headers=headers)
        data = response.json()

        # Should succeed if under limit, or fail with 403 if at limit
        success = response.status_code in [200, 403]

        if response.status_code == 200:
            print_result("Processing limit check passed (under limit)", success, data)
        else:
            print_result("Processing limit reached (expected behavior)", success, data)

        return success
    except Exception as e:
        print_result("Processing limit check", False, error=str(e))
        return False

def test_storage_limit_check(token):
    """Test storage limit check"""
    print_section("Test 9: Storage Limit Check")

    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(f"{BACKEND_URL}/api/auth-example/storage-check", headers=headers)
        data = response.json()

        # Should succeed if under limit, or fail with 403 if at limit
        success = response.status_code in [200, 403]

        if response.status_code == 200:
            print_result("Storage limit check passed (under limit)", success, data)
        else:
            print_result("Storage limit reached (expected behavior)", success, data)

        return success
    except Exception as e:
        print_result("Storage limit check", False, error=str(e))
        return False

def test_invalid_token():
    """Test with invalid token"""
    print_section("Test 10: Invalid Token Handling")

    try:
        headers = {"Authorization": "Bearer invalid_token_12345"}
        response = requests.get(f"{BACKEND_URL}/api/auth-example/protected", headers=headers)
        data = response.json()

        success = response.status_code == 401 and 'error' in data
        print_result("Invalid token is rejected", success, data)
        return success
    except Exception as e:
        print_result("Invalid token test", False, error=str(e))
        return False

def main():
    """Run all tests"""
    print("\n" + "🧪 BACKEND AUTH INTEGRATION TESTS".center(60, "="))
    print(f"\nBackend URL: {BACKEND_URL}")
    print(f"Test Email: {TEST_EMAIL}\n")

    results = []

    # Tests that don't require auth
    results.append(test_public_route())
    results.append(test_optional_route_without_auth())
    results.append(test_protected_route_without_auth())
    results.append(test_invalid_token())

    # Get auth token
    print_section("Getting Auth Token")
    token = get_auth_token()

    if not token:
        print("\n❌ Failed to get auth token. Cannot continue with authenticated tests.")
        print("Please check:")
        print(f"  1. User {TEST_EMAIL} exists in Supabase")
        print(f"  2. Password is correct")
        print(f"  3. Supabase connection is working")
        sys.exit(1)

    print(f"✅ Got auth token: {token[:20]}...")

    # Tests that require auth
    results.append(test_optional_route_with_auth(token))
    results.append(test_protected_route_with_auth(token))
    results.append(test_profile_route(token))
    results.append(test_upload_limit_check(token))
    results.append(test_processing_limit_check(token))
    results.append(test_storage_limit_check(token))

    # Summary
    print_section("TEST SUMMARY")
    passed = sum(results)
    total = len(results)

    print(f"Tests Passed: {passed}/{total}")
    print(f"Tests Failed: {total - passed}/{total}")

    if passed == total:
        print("\n✅ ALL TESTS PASSED! Backend integration is working correctly.")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please review errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
