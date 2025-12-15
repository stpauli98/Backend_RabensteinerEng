"""Tests for shared.database.lifecycle module."""

import pytest
import os
import json
from unittest.mock import Mock, patch, mock_open
from datetime import datetime

from shared.database.lifecycle import (
    load_session_metadata,
    save_metadata_to_database,
    save_files_to_database,
    finalize_session,
    update_session_name,
    save_session_to_supabase,
    _load_session_metadata,
    _save_metadata_to_database,
    _save_files_to_database,
    _finalize_session
)
from shared.database.exceptions import (
    DatabaseError,
    SessionNotFoundError,
    ValidationError,
    ConfigurationError
)


class TestLoadSessionMetadata:
    """Tests for load_session_metadata function."""

    @patch('os.path.exists')
    def test_returns_none_for_missing_directory(self, mock_exists):
        """Test None returned when session directory doesn't exist."""
        mock_exists.return_value = False

        result = load_session_metadata("session-123")

        assert result is None

    @patch('os.path.exists')
    def test_returns_none_for_missing_metadata_file(self, mock_exists):
        """Test None returned when metadata file doesn't exist."""
        mock_exists.side_effect = lambda p: 'metadata' not in p

        result = load_session_metadata("session-123")

        assert result is None

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"key": "value"}')
    def test_loads_valid_json_metadata(self, mock_file, mock_exists):
        """Test valid JSON metadata is loaded."""
        mock_exists.return_value = True

        result = load_session_metadata("session-123")

        assert result == {"key": "value"}

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='invalid json')
    def test_raises_database_error_for_invalid_json(self, mock_file, mock_exists):
        """Test DatabaseError for invalid JSON."""
        mock_exists.return_value = True

        with pytest.raises(DatabaseError) as exc_info:
            load_session_metadata("session-123")

        assert "Invalid session metadata format" in str(exc_info.value)

    @patch('os.path.exists')
    def test_returns_none_on_read_exception(self, mock_exists):
        """Test None returned when file read fails."""
        mock_exists.return_value = True

        with patch('builtins.open', side_effect=IOError("Read error")):
            result = load_session_metadata("session-123")

        assert result is None


class TestSaveMetadataToDatabase:
    """Tests for save_metadata_to_database function."""

    @patch('shared.database.lifecycle.save_time_info')
    @patch('shared.database.lifecycle.save_zeitschritte')
    def test_saves_time_info_when_present(self, mock_save_zeit, mock_save_time):
        """Test time_info is saved when present."""
        metadata = {"timeInfo": {"jahr": True}}

        result = save_metadata_to_database("uuid-123", metadata)

        assert result is True
        mock_save_time.assert_called_once_with("uuid-123", {"jahr": True})

    @patch('shared.database.lifecycle.save_time_info')
    @patch('shared.database.lifecycle.save_zeitschritte')
    def test_saves_zeitschritte_when_present(self, mock_save_zeit, mock_save_time):
        """Test zeitschritte is saved when present."""
        metadata = {"zeitschritte": {"eingabe": "test"}}

        result = save_metadata_to_database("uuid-123", metadata)

        assert result is True
        mock_save_zeit.assert_called_once_with("uuid-123", {"eingabe": "test"})

    @patch('shared.database.lifecycle.save_time_info')
    @patch('shared.database.lifecycle.save_zeitschritte')
    def test_skips_missing_fields(self, mock_save_zeit, mock_save_time):
        """Test missing fields are skipped."""
        metadata = {}

        result = save_metadata_to_database("uuid-123", metadata)

        assert result is True
        mock_save_time.assert_not_called()
        mock_save_zeit.assert_not_called()


class TestSaveFilesToDatabase:
    """Tests for save_files_to_database function."""

    @patch('shared.database.lifecycle.get_supabase_client')
    def test_returns_true_when_no_files(self, mock_get_client):
        """Test True returned when no files in metadata."""
        result = save_files_to_database("uuid-123", "session-123", {})

        assert result is True

    @patch('shared.database.lifecycle.get_supabase_client')
    def test_returns_true_when_files_not_list(self, mock_get_client):
        """Test True returned when files is not a list."""
        result = save_files_to_database("uuid-123", "session-123", {"files": "not-a-list"})

        assert result is True

    @patch('shared.database.lifecycle.get_supabase_client')
    @patch('shared.database.lifecycle.prepare_file_batch_data')
    @patch('shared.database.lifecycle.batch_upsert_files')
    @patch('shared.database.lifecycle.save_csv_file_content')
    @patch('os.path.exists')
    def test_processes_files_successfully(
        self, mock_exists, mock_save_csv, mock_batch_upsert, mock_prepare, mock_get_client
    ):
        """Test successful file processing."""
        mock_exists.return_value = True
        mock_prepare.return_value = [
            {"file_name": "test.csv", "type": "input"}
        ]
        mock_batch_upsert.return_value = ["uuid-1"]
        mock_get_client.return_value = Mock()

        metadata = {"files": [{"fileName": "test.csv"}]}

        result = save_files_to_database("uuid-123", "session-123", metadata)

        assert result is True
        mock_save_csv.assert_called_once()

    @patch('shared.database.lifecycle.get_supabase_client')
    @patch('shared.database.lifecycle.prepare_file_batch_data')
    @patch('shared.database.lifecycle.batch_upsert_files')
    def test_raises_database_error_when_upsert_fails(
        self, mock_batch_upsert, mock_prepare, mock_get_client
    ):
        """Test DatabaseError when batch upsert returns empty."""
        mock_prepare.return_value = [{"file_name": "test.csv"}]
        mock_batch_upsert.return_value = []
        mock_get_client.return_value = Mock()

        metadata = {"files": [{"fileName": "test.csv"}]}

        with pytest.raises(DatabaseError) as exc_info:
            save_files_to_database("uuid-123", "session-123", metadata)

        assert "no files were processed" in str(exc_info.value)


class TestFinalizeSession:
    """Tests for finalize_session function."""

    @patch('shared.database.lifecycle.get_supabase_client')
    def test_returns_true_when_no_params(self, mock_get_client):
        """Test True returned when no parameters provided."""
        result = finalize_session("uuid-123")

        assert result is True
        mock_get_client.assert_not_called()

    @patch('shared.database.lifecycle.get_supabase_client')
    def test_updates_with_n_dat(self, mock_get_client):
        """Test session is updated with n_dat."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.error = None
        mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = finalize_session("uuid-123", n_dat=1000)

        assert result is True
        mock_client.table.return_value.update.assert_called_once()

    @patch('shared.database.lifecycle.get_supabase_client')
    def test_updates_with_file_count(self, mock_get_client):
        """Test session is updated with file_count."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.error = None
        mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = finalize_session("uuid-123", file_count=5)

        assert result is True

    @patch('shared.database.lifecycle.get_supabase_client')
    def test_raises_database_error_on_failure(self, mock_get_client):
        """Test DatabaseError on update failure."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.error = "Update failed"
        mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client

        with pytest.raises(DatabaseError) as exc_info:
            finalize_session("uuid-123", n_dat=100)

        assert "Error updating sessions table" in str(exc_info.value)


class TestUpdateSessionName:
    """Tests for update_session_name function."""

    def test_raises_validation_error_for_invalid_session_id(self):
        """Test ValidationError for invalid session_id."""
        with pytest.raises(ValidationError) as exc_info:
            update_session_name("invalid-session", "New Name")

        assert "Invalid session_id format" in str(exc_info.value)

    def test_raises_validation_error_for_empty_name(self):
        """Test ValidationError for empty session name."""
        with pytest.raises(ValidationError) as exc_info:
            update_session_name("b2be65df-ce96-4305-b4c7-6530c7bc7096", "")

        assert "Invalid session_name" in str(exc_info.value)

    def test_raises_validation_error_for_whitespace_only_name(self):
        """Test ValidationError for whitespace-only name."""
        with pytest.raises(ValidationError) as exc_info:
            update_session_name("b2be65df-ce96-4305-b4c7-6530c7bc7096", "   ")

        assert "cannot be empty" in str(exc_info.value)

    def test_raises_validation_error_for_too_long_name(self):
        """Test ValidationError for name exceeding 255 characters."""
        with pytest.raises(ValidationError) as exc_info:
            update_session_name("b2be65df-ce96-4305-b4c7-6530c7bc7096", "x" * 256)

        assert "too long" in str(exc_info.value)

    @patch('shared.database.lifecycle.get_supabase_client')
    @patch('shared.database.lifecycle.get_session_uuid')
    def test_raises_session_not_found_error(self, mock_get_uuid, mock_get_client):
        """Test SessionNotFoundError when session doesn't exist."""
        mock_get_uuid.return_value = "uuid-123"
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = []
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client

        with pytest.raises(SessionNotFoundError):
            update_session_name("b2be65df-ce96-4305-b4c7-6530c7bc7096", "New Name")

    @patch('shared.database.lifecycle.get_supabase_client')
    @patch('shared.database.lifecycle.get_session_uuid')
    def test_raises_permission_error_for_wrong_user(self, mock_get_uuid, mock_get_client):
        """Test PermissionError when user doesn't own session."""
        mock_get_uuid.return_value = "uuid-123"
        mock_client = Mock()
        mock_select_response = Mock()
        mock_select_response.data = [{'id': 'uuid-123', 'user_id': 'owner123'}]
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_select_response
        mock_get_client.return_value = mock_client

        with pytest.raises(PermissionError):
            update_session_name(
                "b2be65df-ce96-4305-b4c7-6530c7bc7096",
                "New Name",
                user_id="attacker456"
            )

    @patch('shared.database.lifecycle.get_supabase_client')
    @patch('shared.database.lifecycle.get_session_uuid')
    def test_updates_session_name_successfully(self, mock_get_uuid, mock_get_client):
        """Test successful session name update."""
        mock_get_uuid.return_value = "uuid-123"
        mock_client = Mock()
        mock_select_response = Mock()
        mock_select_response.data = [{'id': 'uuid-123', 'user_id': 'user123'}]
        mock_update_response = Mock()
        mock_update_response.error = None

        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_select_response
        mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_update_response
        mock_get_client.return_value = mock_client

        result = update_session_name(
            "b2be65df-ce96-4305-b4c7-6530c7bc7096",
            "New Name",
            user_id="user123"
        )

        assert result is True


class TestSaveSessionToSupabase:
    """Tests for save_session_to_supabase function."""

    @patch('shared.database.lifecycle.get_session_uuid')
    @patch('shared.database.lifecycle.load_session_metadata')
    def test_raises_session_not_found_for_missing_metadata(self, mock_load, mock_get_uuid):
        """Test SessionNotFoundError when metadata can't be loaded."""
        mock_get_uuid.return_value = "uuid-123"
        mock_load.return_value = None

        with pytest.raises(SessionNotFoundError):
            save_session_to_supabase("session-123")

    @patch('shared.database.lifecycle.get_session_uuid')
    @patch('shared.database.lifecycle.load_session_metadata')
    @patch('shared.database.lifecycle.save_metadata_to_database')
    @patch('shared.database.lifecycle.save_files_to_database')
    @patch('shared.database.lifecycle.finalize_session')
    def test_orchestrates_all_saves(
        self, mock_finalize, mock_save_files, mock_save_meta, mock_load, mock_get_uuid
    ):
        """Test all save operations are called."""
        mock_get_uuid.return_value = "uuid-123"
        mock_load.return_value = {"timeInfo": {}, "files": []}

        result = save_session_to_supabase("session-123", n_dat=100, file_count=5)

        assert result is True
        mock_save_meta.assert_called_once()
        mock_save_files.assert_called_once()
        mock_finalize.assert_called_once_with("uuid-123", 100, 5)

    @patch('shared.database.lifecycle.get_session_uuid')
    @patch('shared.database.lifecycle.load_session_metadata')
    @patch('shared.database.lifecycle.save_metadata_to_database')
    @patch('shared.database.lifecycle.save_files_to_database')
    @patch('shared.database.lifecycle.finalize_session')
    def test_continues_on_metadata_failure(
        self, mock_finalize, mock_save_files, mock_save_meta, mock_load, mock_get_uuid
    ):
        """Test continues with files even if metadata save fails."""
        mock_get_uuid.return_value = "uuid-123"
        mock_load.return_value = {"files": []}
        mock_save_meta.side_effect = Exception("Metadata error")

        result = save_session_to_supabase("session-123")

        assert result is True
        mock_save_files.assert_called_once()

    @patch('shared.database.lifecycle.get_session_uuid')
    @patch('shared.database.lifecycle.load_session_metadata')
    @patch('shared.database.lifecycle.save_metadata_to_database')
    @patch('shared.database.lifecycle.save_files_to_database')
    @patch('shared.database.lifecycle.finalize_session')
    def test_continues_on_file_save_failure(
        self, mock_finalize, mock_save_files, mock_save_meta, mock_load, mock_get_uuid
    ):
        """Test continues with finalization even if file save fails."""
        mock_get_uuid.return_value = "uuid-123"
        mock_load.return_value = {}
        mock_save_files.side_effect = Exception("File error")

        result = save_session_to_supabase("session-123")

        assert result is True
        mock_finalize.assert_called_once()


class TestBackwardCompatibility:
    """Tests for backward compatibility aliases."""

    def test_load_alias_exists(self):
        """Test _load_session_metadata alias exists."""
        assert _load_session_metadata is load_session_metadata

    def test_save_metadata_alias_exists(self):
        """Test _save_metadata_to_database alias exists."""
        assert _save_metadata_to_database is save_metadata_to_database

    def test_save_files_alias_exists(self):
        """Test _save_files_to_database alias exists."""
        assert _save_files_to_database is save_files_to_database

    def test_finalize_alias_exists(self):
        """Test _finalize_session alias exists."""
        assert _finalize_session is finalize_session

