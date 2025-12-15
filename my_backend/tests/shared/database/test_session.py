"""Tests for shared.database.session module.

These are critical path tests covering UUID management functions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from shared.database.session import (
    get_supabase_client,
    retry_database_operation,
    create_or_get_session_uuid,
    get_string_id_from_uuid,
    get_session_uuid,
    _get_session_uuid
)
from shared.database.exceptions import (
    DatabaseError,
    SessionNotFoundError,
    ValidationError,
    ConfigurationError
)
from shared.database.config import DatabaseConfig


class TestGetSupabaseClient:
    """Tests for get_supabase_client function."""

    @patch('shared.database.session.get_shared_client')
    def test_returns_anon_client_by_default(self, mock_shared_client):
        """Test that anon client is returned when use_service_role=False."""
        mock_client = Mock()
        mock_shared_client.return_value = mock_client

        result = get_supabase_client(use_service_role=False)

        mock_shared_client.assert_called_once()
        assert result == mock_client

    @patch('shared.database.session.get_supabase_admin_client')
    def test_returns_admin_client_when_service_role(self, mock_admin_client):
        """Test that admin client is returned when use_service_role=True."""
        mock_client = Mock()
        mock_admin_client.return_value = mock_client

        result = get_supabase_client(use_service_role=True)

        mock_admin_client.assert_called_once()
        assert result == mock_client

    @patch('shared.database.session.get_shared_client')
    def test_raises_configuration_error_when_client_unavailable(self, mock_shared_client):
        """Test ConfigurationError is raised when client is None."""
        mock_shared_client.return_value = None

        with pytest.raises(ConfigurationError) as exc_info:
            get_supabase_client(use_service_role=False)

        assert "Supabase client not available" in str(exc_info.value)


class TestRetryDatabaseOperation:
    """Tests for retry_database_operation function."""

    def test_successful_operation_returns_immediately(self):
        """Test successful operation returns without retries."""
        operation = Mock(return_value="success")

        result = retry_database_operation(operation)

        assert result == "success"
        operation.assert_called_once()

    def test_retries_on_timeout_error(self):
        """Test operation retries on timeout errors."""
        operation = Mock(side_effect=[
            Exception("Lookup timed out"),
            "success"
        ])

        with patch('shared.database.session.time.sleep'):
            result = retry_database_operation(operation, max_retries=2, initial_delay=0.1)

        assert result == "success"
        assert operation.call_count == 2

    def test_raises_after_max_retries_on_timeout(self):
        """Test raises DatabaseError after max retries on timeout."""
        operation = Mock(side_effect=Exception("Lookup timed out"))

        with patch('shared.database.session.time.sleep'):
            with pytest.raises(DatabaseError) as exc_info:
                retry_database_operation(operation, max_retries=2, initial_delay=0.1)

        assert "failed after 3 attempts" in str(exc_info.value)
        assert operation.call_count == 3

    def test_raises_immediately_on_non_timeout_error(self):
        """Test non-timeout errors raise immediately without retry."""
        operation = Mock(side_effect=Exception("Connection refused"))

        with pytest.raises(DatabaseError) as exc_info:
            retry_database_operation(operation)

        assert "Connection refused" in str(exc_info.value)
        operation.assert_called_once()

    def test_exponential_backoff_delays(self):
        """Test exponential backoff timing."""
        operation = Mock(side_effect=[
            Exception("timeout"),
            Exception("timeout"),
            "success"
        ])
        sleep_calls = []

        with patch('shared.database.session.time.sleep', side_effect=lambda x: sleep_calls.append(x)):
            retry_database_operation(operation, max_retries=3, initial_delay=1.0)

        assert sleep_calls == [1.0, 2.0]  # 1.0 * 2^0, 1.0 * 2^1


class TestCreateOrGetSessionUuid:
    """Tests for create_or_get_session_uuid function."""

    def test_returns_uuid_as_is_if_already_uuid(self):
        """Test that valid UUIDs are returned unchanged."""
        valid_uuid = "b2be65df-ce96-4305-b4c7-6530c7bc7096"

        result = create_or_get_session_uuid(valid_uuid)

        assert result == valid_uuid

    def test_raises_validation_error_for_invalid_session_id(self):
        """Test ValidationError for invalid session_id format."""
        with pytest.raises(ValidationError) as exc_info:
            create_or_get_session_uuid("invalid-session")

        assert "Invalid session_id format" in str(exc_info.value)

    @patch('shared.database.session.get_supabase_client')
    @patch('shared.database.session.retry_database_operation')
    def test_returns_existing_uuid_from_mapping(self, mock_retry, mock_get_client):
        """Test returns existing UUID from session_mappings table."""
        mock_retry.return_value = "existing-uuid-123"

        result = create_or_get_session_uuid("session_1234567890_abc123", user_id="user123")

        assert result == "existing-uuid-123"

    @patch('shared.database.session.get_supabase_client')
    @patch('shared.database.session.retry_database_operation')
    def test_raises_value_error_when_creating_without_user_id(self, mock_retry, mock_get_client):
        """Test ValueError when creating new session without user_id."""
        mock_retry.return_value = None  # No existing mapping found

        with pytest.raises(ValueError) as exc_info:
            create_or_get_session_uuid("session_1234567890_abc123", user_id=None)

        assert "user_id is required" in str(exc_info.value)

    @patch('shared.database.session.get_supabase_client')
    def test_validates_legacy_session_format(self, mock_get_client):
        """Test validation of legacy session format."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [{'uuid_session_id': 'uuid-123', 'sessions': {'user_id': 'user123'}}]
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Should not raise ValidationError for valid legacy format
        result = create_or_get_session_uuid("session_1234567890_abc123", user_id="user123")
        assert result == "uuid-123"

    @patch('shared.database.session.get_supabase_client')
    def test_validates_session_uuid_format(self, mock_get_client):
        """Test validation of session_UUID format."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [{'uuid_session_id': 'uuid-456', 'sessions': {'user_id': 'user123'}}]
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = create_or_get_session_uuid(
            "session_b2be65df-ce96-4305-b4c7-6530c7bc7096",
            user_id="user123"
        )
        assert result == "uuid-456"


class TestSessionOwnershipValidation:
    """Tests for session ownership validation (SECURITY)."""

    @patch('shared.database.session.get_supabase_client')
    def test_raises_permission_error_for_wrong_user(self, mock_get_client):
        """Test PermissionError when user doesn't own the session."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [{'uuid_session_id': 'uuid-123', 'sessions': {'user_id': 'owner123'}}]
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client

        with pytest.raises(PermissionError) as exc_info:
            create_or_get_session_uuid("session_1234567890_abc123", user_id="attacker456")

        assert "does not belong to user" in str(exc_info.value)

    @patch('shared.database.session.get_supabase_client')
    def test_allows_access_for_correct_user(self, mock_get_client):
        """Test access is allowed for session owner."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [{'uuid_session_id': 'uuid-123', 'sessions': {'user_id': 'owner123'}}]
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = create_or_get_session_uuid("session_1234567890_abc123", user_id="owner123")

        assert result == "uuid-123"


class TestGetStringIdFromUuid:
    """Tests for get_string_id_from_uuid function."""

    def test_returns_none_for_empty_uuid(self):
        """Test None is returned for empty/None UUID."""
        assert get_string_id_from_uuid(None) is None
        assert get_string_id_from_uuid("") is None

    @patch('shared.database.session.get_supabase_client')
    def test_returns_string_id_for_valid_uuid(self, mock_get_client):
        """Test correct string ID is returned for valid UUID."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [{'string_session_id': 'session_1234567890_abc123'}]
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = get_string_id_from_uuid("b2be65df-ce96-4305-b4c7-6530c7bc7096")

        assert result == "session_1234567890_abc123"

    @patch('shared.database.session.get_supabase_client')
    def test_returns_none_when_uuid_not_found(self, mock_get_client):
        """Test None is returned when UUID is not found."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = []
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = get_string_id_from_uuid("nonexistent-uuid")

        assert result is None

    @patch('shared.database.session.get_supabase_client')
    def test_returns_none_on_exception(self, mock_get_client):
        """Test None is returned when exception occurs."""
        mock_client = Mock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.side_effect = Exception("DB error")
        mock_get_client.return_value = mock_client

        result = get_string_id_from_uuid("some-uuid")

        assert result is None


class TestGetSessionUuid:
    """Tests for get_session_uuid convenience wrapper."""

    def test_returns_uuid_as_is_for_valid_uuid(self):
        """Test valid UUIDs are returned unchanged."""
        valid_uuid = "b2be65df-ce96-4305-b4c7-6530c7bc7096"

        result = get_session_uuid(valid_uuid)

        assert result == valid_uuid

    def test_returns_uuid_as_is_for_uppercase_uuid(self):
        """Test uppercase UUIDs are returned unchanged."""
        valid_uuid = "B2BE65DF-CE96-4305-B4C7-6530C7BC7096"

        result = get_session_uuid(valid_uuid)

        assert result == valid_uuid

    @patch('shared.database.session.create_or_get_session_uuid')
    def test_calls_create_or_get_for_string_session_id(self, mock_create):
        """Test create_or_get_session_uuid is called for string IDs."""
        mock_create.return_value = "new-uuid-123"

        result = get_session_uuid("session_1234567890_abc123", user_id="user123")

        mock_create.assert_called_once_with("session_1234567890_abc123", user_id="user123")
        assert result == "new-uuid-123"

    @patch('shared.database.session.create_or_get_session_uuid')
    def test_propagates_exceptions_from_create_or_get(self, mock_create):
        """Test exceptions from create_or_get_session_uuid are propagated."""
        mock_create.side_effect = SessionNotFoundError("Session not found")

        with pytest.raises(SessionNotFoundError):
            get_session_uuid("session_1234567890_abc123", user_id="user123")


class TestBackwardCompatibility:
    """Tests for backward compatibility aliases."""

    def test_get_session_uuid_alias_exists(self):
        """Test _get_session_uuid alias is available."""
        assert _get_session_uuid is get_session_uuid

    def test_alias_works_same_as_original(self):
        """Test alias produces same result as original."""
        valid_uuid = "b2be65df-ce96-4305-b4c7-6530c7bc7096"

        result = _get_session_uuid(valid_uuid)

        assert result == valid_uuid


class TestDatabaseConfigIntegration:
    """Tests for DatabaseConfig integration in session module."""

    def test_default_retry_attempts_used(self):
        """Test default retry attempts from config are used."""
        assert DatabaseConfig.DEFAULT_RETRY_ATTEMPTS == 3

    def test_uuid_pattern_validates_correctly(self):
        """Test UUID pattern from config works correctly."""
        valid_uuid = "b2be65df-ce96-4305-b4c7-6530c7bc7096"
        invalid_uuid = "not-a-uuid"

        assert DatabaseConfig.UUID_PATTERN.match(valid_uuid)
        assert not DatabaseConfig.UUID_PATTERN.match(invalid_uuid)

