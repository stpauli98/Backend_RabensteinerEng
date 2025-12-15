"""Tests for shared.database.persistence module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from shared.database.persistence import (
    save_time_info,
    save_zeitschritte,
    save_file_info,
    transform_time_info_to_jsonb,
    _transform_time_info_to_jsonb
)
from shared.database.exceptions import (
    DatabaseError,
    SessionNotFoundError,
    ValidationError,
    ConfigurationError
)
from shared.database.config import DomainDefaults


class TestSaveTimeInfo:
    """Tests for save_time_info function."""

    def test_raises_validation_error_for_invalid_session_id(self):
        """Test ValidationError for invalid session_id."""
        with pytest.raises(ValidationError) as exc_info:
            save_time_info("invalid-session", {})

        assert "Invalid session_id format" in str(exc_info.value)

    def test_raises_validation_error_for_invalid_time_info(self):
        """Test ValidationError for invalid time_info structure."""
        with pytest.raises(ValidationError) as exc_info:
            save_time_info("b2be65df-ce96-4305-b4c7-6530c7bc7096", {"jahr": "true"})

        assert "Invalid time_info structure" in str(exc_info.value)

    @patch('shared.database.persistence.get_supabase_client')
    @patch('shared.database.persistence.get_session_uuid')
    def test_creates_new_time_info_record(self, mock_get_uuid, mock_get_client):
        """Test creating new time_info record."""
        mock_get_uuid.return_value = "uuid-123"
        mock_client = Mock()
        mock_existing_response = Mock()
        mock_existing_response.data = []
        mock_insert_response = Mock()
        mock_insert_response.error = None

        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_existing_response
        mock_client.table.return_value.insert.return_value.execute.return_value = mock_insert_response
        mock_get_client.return_value = mock_client

        result = save_time_info(
            "b2be65df-ce96-4305-b4c7-6530c7bc7096",
            {"jahr": True, "monat": False}
        )

        assert result is True
        mock_client.table.return_value.insert.assert_called_once()

    @patch('shared.database.persistence.get_supabase_client')
    @patch('shared.database.persistence.get_session_uuid')
    def test_updates_existing_time_info_record(self, mock_get_uuid, mock_get_client):
        """Test updating existing time_info record."""
        mock_get_uuid.return_value = "uuid-123"
        mock_client = Mock()
        mock_existing_response = Mock()
        mock_existing_response.data = [{'id': 1}]
        mock_update_response = Mock()
        mock_update_response.error = None

        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_existing_response
        mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_update_response
        mock_get_client.return_value = mock_client

        result = save_time_info(
            "b2be65df-ce96-4305-b4c7-6530c7bc7096",
            {"jahr": True}
        )

        assert result is True
        mock_client.table.return_value.update.assert_called_once()

    @patch('shared.database.persistence.get_supabase_client')
    @patch('shared.database.persistence.get_session_uuid')
    def test_raises_database_error_on_insert_failure(self, mock_get_uuid, mock_get_client):
        """Test DatabaseError when insert fails."""
        mock_get_uuid.return_value = "uuid-123"
        mock_client = Mock()
        mock_existing_response = Mock()
        mock_existing_response.data = []
        mock_insert_response = Mock()
        mock_insert_response.error = "Insert failed"

        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_existing_response
        mock_client.table.return_value.insert.return_value.execute.return_value = mock_insert_response
        mock_get_client.return_value = mock_client

        with pytest.raises(DatabaseError) as exc_info:
            save_time_info("b2be65df-ce96-4305-b4c7-6530c7bc7096", {})

        assert "Error saving time_info" in str(exc_info.value)


class TestSaveZeitschritte:
    """Tests for save_zeitschritte function."""

    def test_raises_validation_error_for_invalid_session_id(self):
        """Test ValidationError for invalid session_id."""
        with pytest.raises(ValidationError) as exc_info:
            save_zeitschritte("invalid-session", {})

        assert "Invalid session_id format" in str(exc_info.value)

    @patch('shared.database.persistence.get_supabase_client')
    def test_uses_uuid_directly_if_valid(self, mock_get_client):
        """Test using UUID directly when session_id is valid UUID."""
        mock_client = Mock()
        mock_existing_response = Mock()
        mock_existing_response.data = []
        mock_insert_response = Mock()
        mock_insert_response.error = None

        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_existing_response
        mock_client.table.return_value.insert.return_value.execute.return_value = mock_insert_response
        mock_get_client.return_value = mock_client

        result = save_zeitschritte(
            "b2be65df-ce96-4305-b4c7-6530c7bc7096",
            {"eingabe": "test"}
        )

        assert result is True

    @patch('shared.database.persistence.get_supabase_client')
    @patch('shared.database.persistence.create_or_get_session_uuid')
    def test_converts_string_session_id(self, mock_create_uuid, mock_get_client):
        """Test converting string session_id to UUID."""
        mock_create_uuid.return_value = "uuid-123"
        mock_client = Mock()
        mock_existing_response = Mock()
        mock_existing_response.data = []
        mock_insert_response = Mock()
        mock_insert_response.error = None

        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_existing_response
        mock_client.table.return_value.insert.return_value.execute.return_value = mock_insert_response
        mock_get_client.return_value = mock_client

        result = save_zeitschritte(
            "session_1234567890_abc123",
            {"eingabe": "test"},
            user_id="user123"
        )

        assert result is True
        mock_create_uuid.assert_called_once_with("session_1234567890_abc123", user_id="user123")

    @patch('shared.database.persistence.get_supabase_client')
    def test_handles_offset_vs_offsett(self, mock_get_client):
        """Test handling both offset and offsett field names."""
        mock_client = Mock()
        mock_existing_response = Mock()
        mock_existing_response.data = []
        mock_insert_response = Mock()
        mock_insert_response.error = None

        insert_data = {}

        def capture_insert(data):
            nonlocal insert_data
            insert_data = data
            mock_result = Mock()
            mock_result.execute.return_value = mock_insert_response
            return mock_result

        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_existing_response
        mock_client.table.return_value.insert = capture_insert
        mock_get_client.return_value = mock_client

        # Test with offsett
        save_zeitschritte(
            "b2be65df-ce96-4305-b4c7-6530c7bc7096",
            {"offsett": "10"}
        )
        assert insert_data.get("offset") == "10"

    @patch('shared.database.persistence.get_supabase_client')
    def test_clean_value_handles_empty_strings(self, mock_get_client):
        """Test clean_value converts empty strings to None."""
        mock_client = Mock()
        mock_existing_response = Mock()
        mock_existing_response.data = []
        mock_insert_response = Mock()
        mock_insert_response.error = None

        insert_data = {}

        def capture_insert(data):
            nonlocal insert_data
            insert_data = data
            mock_result = Mock()
            mock_result.execute.return_value = mock_insert_response
            return mock_result

        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_existing_response
        mock_client.table.return_value.insert = capture_insert
        mock_get_client.return_value = mock_client

        save_zeitschritte(
            "b2be65df-ce96-4305-b4c7-6530c7bc7096",
            {"eingabe": "", "ausgabe": None}
        )

        assert insert_data.get("eingabe") is None
        assert insert_data.get("ausgabe") is None


class TestSaveFileInfo:
    """Tests for save_file_info function."""

    def test_raises_validation_error_for_invalid_file_info(self):
        """Test ValidationError for invalid file_info."""
        with pytest.raises(ValidationError) as exc_info:
            save_file_info("b2be65df-ce96-4305-b4c7-6530c7bc7096", {})

        assert "Invalid file_info structure" in str(exc_info.value)

    @patch('shared.database.persistence.get_supabase_client')
    def test_generates_uuid_if_not_provided(self, mock_get_client):
        """Test UUID generation when not provided in file_info."""
        mock_client = Mock()
        mock_insert_response = Mock()
        mock_insert_response.error = None
        mock_insert_response.data = [{'id': 'generated-uuid'}]
        mock_client.table.return_value.insert.return_value.execute.return_value = mock_insert_response
        mock_get_client.return_value = mock_client

        success, file_uuid = save_file_info(
            "b2be65df-ce96-4305-b4c7-6530c7bc7096",
            {"fileName": "test.csv"}
        )

        assert success is True
        assert file_uuid is not None

    @patch('shared.database.persistence.get_supabase_client')
    def test_uses_provided_valid_uuid(self, mock_get_client):
        """Test using provided valid UUID."""
        mock_client = Mock()
        mock_insert_response = Mock()
        mock_insert_response.error = None
        mock_client.table.return_value.insert.return_value.execute.return_value = mock_insert_response
        mock_get_client.return_value = mock_client

        provided_uuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        success, file_uuid = save_file_info(
            "b2be65df-ce96-4305-b4c7-6530c7bc7096",
            {"fileName": "test.csv", "id": provided_uuid}
        )

        assert success is True
        assert file_uuid == provided_uuid

    @patch('shared.database.persistence.get_supabase_client')
    def test_generates_storage_path_from_filename(self, mock_get_client):
        """Test storage path generation from fileName."""
        mock_client = Mock()
        mock_insert_response = Mock()
        mock_insert_response.error = None

        insert_data = {}

        def capture_insert(data):
            nonlocal insert_data
            insert_data = data
            mock_result = Mock()
            mock_result.execute.return_value = mock_insert_response
            return mock_result

        mock_client.table.return_value.insert = capture_insert
        mock_get_client.return_value = mock_client

        save_file_info(
            "b2be65df-ce96-4305-b4c7-6530c7bc7096",
            {"fileName": "test.csv"}
        )

        assert "b2be65df-ce96-4305-b4c7-6530c7bc7096" in insert_data.get("storage_path", "")
        assert "test.csv" in insert_data.get("storage_path", "")

    @patch('shared.database.persistence.get_supabase_client')
    def test_returns_tuple_with_success_and_uuid(self, mock_get_client):
        """Test return value is tuple (success, uuid)."""
        mock_client = Mock()
        mock_insert_response = Mock()
        mock_insert_response.error = None
        mock_client.table.return_value.insert.return_value.execute.return_value = mock_insert_response
        mock_get_client.return_value = mock_client

        result = save_file_info(
            "b2be65df-ce96-4305-b4c7-6530c7bc7096",
            {"fileName": "test.csv"}
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is True
        assert result[1] is not None


class TestTransformTimeInfoToJsonb:
    """Tests for transform_time_info_to_jsonb function."""

    def test_preserves_existing_category_data(self):
        """Test existing category_data is preserved."""
        time_info = {
            "jahr": True,
            "monat": False,
            "category_data": {"existing": "data"}
        }

        result = transform_time_info_to_jsonb(time_info)

        assert result["category_data"] == {"existing": "data"}

    def test_creates_category_data_for_enabled_categories(self):
        """Test category_data creation for enabled categories."""
        time_info = {
            "jahr": True,
            "monat": True,
            "woche": False,
            "feiertag": False,
            "tag": False
        }

        result = transform_time_info_to_jsonb(time_info)

        assert "jahr" in result["category_data"]
        assert "monat" in result["category_data"]
        assert "woche" not in result["category_data"]

    def test_feiertag_includes_land(self):
        """Test feiertag category includes land field."""
        time_info = {
            "jahr": False,
            "monat": False,
            "woche": False,
            "feiertag": True,
            "tag": False,
            "land": "Austria"
        }

        result = transform_time_info_to_jsonb(time_info)

        assert result["category_data"]["feiertag"]["land"] == "Austria"

    def test_uses_default_land_if_not_provided(self):
        """Test default land value is used when not provided."""
        time_info = {
            "jahr": False,
            "monat": False,
            "woche": False,
            "feiertag": True,
            "tag": False
        }

        result = transform_time_info_to_jsonb(time_info)

        assert result["category_data"]["feiertag"]["land"] == DomainDefaults.LAND

    def test_includes_category_template_fields(self):
        """Test category data includes all template fields."""
        time_info = {
            "jahr": True,
            "monat": False,
            "woche": False,
            "feiertag": False,
            "tag": False,
            "detaillierteBerechnung": True,
            "datenform": "test",
            "zeithorizontStart": "2024-01-01",
            "zeithorizontEnd": "2024-12-31",
            "skalierung": "ja",
            "skalierungMin": "0",
            "skalierungMax": "100"
        }

        result = transform_time_info_to_jsonb(time_info)

        jahr_data = result["category_data"]["jahr"]
        assert jahr_data["detaillierteBerechnung"] is True
        assert jahr_data["datenform"] == "test"
        assert jahr_data["zeithorizontStart"] == "2024-01-01"
        assert jahr_data["zeithorizontEnd"] == "2024-12-31"
        assert jahr_data["skalierung"] == "ja"
        assert jahr_data["skalierungMin"] == "0"
        assert jahr_data["skalierungMax"] == "100"

    def test_preserves_boolean_fields(self):
        """Test boolean fields are preserved correctly."""
        time_info = {
            "jahr": True,
            "monat": False,
            "woche": True,
            "feiertag": False,
            "tag": True,
            "zeitzone": "CET"
        }

        result = transform_time_info_to_jsonb(time_info)

        assert result["jahr"] is True
        assert result["monat"] is False
        assert result["woche"] is True
        assert result["feiertag"] is False
        assert result["tag"] is True
        assert result["zeitzone"] == "CET"


class TestBackwardCompatibility:
    """Tests for backward compatibility aliases."""

    def test_transform_alias_exists(self):
        """Test _transform_time_info_to_jsonb alias exists."""
        assert _transform_time_info_to_jsonb is transform_time_info_to_jsonb

    def test_alias_produces_same_result(self):
        """Test alias produces same result as original."""
        time_info = {"jahr": True, "monat": False}

        result1 = transform_time_info_to_jsonb(time_info)
        result2 = _transform_time_info_to_jsonb(time_info)

        assert result1 == result2

