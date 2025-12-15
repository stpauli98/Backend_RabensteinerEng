"""Tests for shared.database.batch module."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from shared.database.batch import (
    _map_file_info_to_db_record,
    prepare_file_batch_data,
    batch_upsert_files,
    _prepare_file_batch_data,
    _batch_upsert_files
)
from shared.database.exceptions import DatabaseError
from shared.database.config import DomainDefaults


class TestMapFileInfoToDbRecord:
    """Tests for _map_file_info_to_db_record function."""

    def test_generates_uuid_if_invalid(self):
        """Test UUID generation when provided id is invalid."""
        file_info = {"fileName": "test.csv", "id": "invalid-id"}

        result = _map_file_info_to_db_record(file_info, "session-123")

        # Should have generated a valid UUID
        assert result["id"] is not None
        assert len(result["id"]) == 36  # Standard UUID format

    def test_uses_valid_uuid_from_file_info(self):
        """Test valid UUID from file_info is used."""
        valid_uuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        file_info = {"fileName": "test.csv", "id": valid_uuid}

        result = _map_file_info_to_db_record(file_info, "session-123")

        assert result["id"] == valid_uuid

    def test_generates_storage_path_from_filename(self):
        """Test storage path generation when not provided."""
        file_info = {"fileName": "test.csv"}

        result = _map_file_info_to_db_record(file_info, "session-123")

        assert result["storage_path"] == "session-123/test.csv"

    def test_uses_provided_storage_path(self):
        """Test provided storage path is used."""
        file_info = {"fileName": "test.csv", "storagePath": "custom/path/file.csv"}

        result = _map_file_info_to_db_record(file_info, "session-123")

        assert result["storage_path"] == "custom/path/file.csv"

    def test_uses_default_skalierung(self):
        """Test default skalierung value is used."""
        file_info = {"fileName": "test.csv"}

        result = _map_file_info_to_db_record(file_info, "session-123")

        assert result["skalierung"] == DomainDefaults.SKALIERUNG

    def test_uses_default_mittelwertbildung(self):
        """Test default mittelwertbildung value is used."""
        file_info = {"fileName": "test.csv"}

        result = _map_file_info_to_db_record(file_info, "session-123")

        assert result["mittelwertbildung_uber_den_zeithorizont"] == DomainDefaults.MITTELWERTBILDUNG

    def test_uses_default_file_type(self):
        """Test default file type is used."""
        file_info = {"fileName": "test.csv"}

        result = _map_file_info_to_db_record(file_info, "session-123")

        assert result["type"] == DomainDefaults.FILE_TYPE

    def test_parses_utc_timestamps(self):
        """Test UTC timestamp parsing."""
        file_info = {
            "fileName": "test.csv",
            "utcMin": "2024-01-01T00:00:00",
            "utcMax": "2024-12-31T23:59:59"
        }

        result = _map_file_info_to_db_record(file_info, "session-123")

        assert "utc_min" in result
        assert "utc_max" in result

    def test_handles_invalid_utc_timestamps(self):
        """Test invalid UTC timestamps are handled gracefully."""
        file_info = {
            "fileName": "test.csv",
            "utcMin": "invalid-date",
            "utcMax": "also-invalid"
        }

        # Should not raise, just skip invalid dates
        result = _map_file_info_to_db_record(file_info, "session-123")

        assert "utc_min" not in result
        assert "utc_max" not in result

    def test_applies_fallback_zeitschrittweite_for_output(self):
        """Test fallback zeitschrittweite values for output files."""
        file_info = {"fileName": "output.csv", "type": "output"}

        result = _map_file_info_to_db_record(
            file_info,
            "session-123",
            zeitschrittweite_mittelwert="15",
            zeitschrittweite_min="10"
        )

        assert result["zeitschrittweite_mittelwert"] == "15"
        assert result["zeitschrittweite_min"] == "10"

    def test_no_fallback_for_input_files(self):
        """Test no fallback applied for input files."""
        file_info = {"fileName": "input.csv", "type": "input"}

        result = _map_file_info_to_db_record(
            file_info,
            "session-123",
            zeitschrittweite_mittelwert="15",
            zeitschrittweite_min="10"
        )

        assert result["zeitschrittweite_mittelwert"] is None
        assert result["zeitschrittweite_min"] is None


class TestPrepareFileBatchData:
    """Tests for prepare_file_batch_data function."""

    def test_returns_empty_list_for_empty_input(self):
        """Test empty list returned for empty input."""
        result = prepare_file_batch_data("session-123", [])

        assert result == []

    def test_skips_invalid_file_info(self):
        """Test invalid file info is skipped."""
        files_list = [
            {"fileName": "valid.csv"},
            {"invalid": "no fileName"},  # Should be skipped
            {"fileName": "another.csv"}
        ]

        result = prepare_file_batch_data("session-123", files_list)

        assert len(result) == 2
        assert all(r["file_name"] in ["valid.csv", "another.csv"] for r in result)

    def test_extracts_zeitschrittweite_from_input_files(self):
        """Test zeitschrittweite extraction from input files."""
        files_list = [
            {
                "fileName": "input.csv",
                "type": "input",
                "zeitschrittweiteAvgValue": "15"
            },
            {
                "fileName": "output.csv",
                "type": "output"
            }
        ]

        result = prepare_file_batch_data("session-123", files_list)

        # Output file should have inherited zeitschrittweite from input
        output_record = next(r for r in result if r["file_name"] == "output.csv")
        assert output_record["zeitschrittweite_mittelwert"] == "15"

    def test_all_files_have_session_id(self):
        """Test all prepared files have correct session_id."""
        files_list = [
            {"fileName": "file1.csv"},
            {"fileName": "file2.csv"},
            {"fileName": "file3.csv"}
        ]

        result = prepare_file_batch_data("session-123", files_list)

        assert all(r["session_id"] == "session-123" for r in result)


class TestBatchUpsertFiles:
    """Tests for batch_upsert_files function."""

    def test_returns_empty_list_for_empty_batch(self):
        """Test empty list returned for empty batch."""
        mock_client = Mock()

        result = batch_upsert_files(mock_client, "session-123", [])

        assert result == []

    def test_inserts_new_files(self):
        """Test new files are inserted."""
        mock_client = Mock()
        mock_existing_response = Mock()
        mock_existing_response.data = []
        mock_insert_response = Mock()
        mock_insert_response.data = [{'id': 'new-uuid-1'}]
        mock_insert_response.error = None

        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_existing_response
        mock_client.table.return_value.insert.return_value.execute.return_value = mock_insert_response

        batch_data = [{"id": "new-uuid-1", "bezeichnung": "New File", "file_name": "new.csv"}]

        result = batch_upsert_files(mock_client, "session-123", batch_data)

        assert "new-uuid-1" in result
        mock_client.table.return_value.insert.assert_called_once()

    def test_updates_existing_files_by_bezeichnung(self):
        """Test existing files are updated by bezeichnung match."""
        mock_client = Mock()
        mock_existing_response = Mock()
        mock_existing_response.data = [
            {'id': 'existing-uuid', 'bezeichnung': 'Existing File'}
        ]
        mock_update_response = Mock()
        mock_update_response.error = None

        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_existing_response
        mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_update_response

        batch_data = [{"id": "new-uuid", "bezeichnung": "Existing File", "file_name": "update.csv"}]

        result = batch_upsert_files(mock_client, "session-123", batch_data)

        assert "existing-uuid" in result
        mock_client.table.return_value.update.assert_called_once()

    def test_raises_database_error_on_insert_failure(self):
        """Test DatabaseError on insert failure."""
        mock_client = Mock()
        mock_existing_response = Mock()
        mock_existing_response.data = []
        mock_insert_response = Mock()
        mock_insert_response.error = "Insert failed"
        mock_insert_response.data = []

        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_existing_response
        mock_client.table.return_value.insert.return_value.execute.return_value = mock_insert_response

        batch_data = [{"id": "new-uuid", "bezeichnung": "File", "file_name": "test.csv"}]

        with pytest.raises(DatabaseError) as exc_info:
            batch_upsert_files(mock_client, "session-123", batch_data)

        assert "Batch file insert failed" in str(exc_info.value)

    def test_raises_database_error_on_exception(self):
        """Test DatabaseError on unexpected exception."""
        mock_client = Mock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.side_effect = Exception(
            "Network error"
        )

        batch_data = [{"id": "uuid", "bezeichnung": "File", "file_name": "test.csv"}]

        with pytest.raises(DatabaseError) as exc_info:
            batch_upsert_files(mock_client, "session-123", batch_data)

        assert "Batch file upsert failed" in str(exc_info.value)

    def test_preserves_existing_files_not_in_batch(self):
        """Test existing files not in batch are preserved (no DELETE)."""
        mock_client = Mock()
        mock_existing_response = Mock()
        mock_existing_response.data = [
            {'id': 'existing-1', 'bezeichnung': 'File1'},
            {'id': 'existing-2', 'bezeichnung': 'File2'}
        ]
        mock_insert_response = Mock()
        mock_insert_response.data = [{'id': 'new-uuid'}]
        mock_insert_response.error = None

        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_existing_response
        mock_client.table.return_value.insert.return_value.execute.return_value = mock_insert_response

        # Only inserting new file, existing files should remain untouched
        batch_data = [{"id": "new-uuid", "bezeichnung": "NewFile", "file_name": "new.csv"}]

        result = batch_upsert_files(mock_client, "session-123", batch_data)

        # Verify no delete operation was called
        assert not mock_client.table.return_value.delete.called


class TestBackwardCompatibility:
    """Tests for backward compatibility aliases."""

    def test_prepare_alias_exists(self):
        """Test _prepare_file_batch_data alias exists."""
        assert _prepare_file_batch_data is prepare_file_batch_data

    def test_upsert_alias_exists(self):
        """Test _batch_upsert_files alias exists."""
        assert _batch_upsert_files is batch_upsert_files

