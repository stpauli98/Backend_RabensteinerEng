"""Tests for shared.database.storage module."""

import pytest
import os
from unittest.mock import Mock, patch, mock_open

from shared.database.storage import (
    save_csv_file_content,
    delete_csv_file_content,
    get_csv_file_url
)
from shared.database.exceptions import StorageError, ConfigurationError
from shared.database.config import BucketNames


class TestSaveCsvFileContent:
    """Tests for save_csv_file_content function."""

    def test_raises_file_not_found_for_missing_file(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            save_csv_file_content(
                file_id="file-123",
                session_id="session-456",
                file_name="test.csv",
                file_path="/nonexistent/path/file.csv",
                file_type="input"
            )

        assert "File not found" in str(exc_info.value)

    @patch('shared.database.storage.get_supabase_client')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data=b'csv,data')
    def test_uploads_new_file_successfully(self, mock_file, mock_exists, mock_get_client):
        """Test successful upload of new file."""
        mock_exists.return_value = True
        mock_client = Mock()
        mock_storage = Mock()
        mock_storage.list.return_value = []
        mock_storage.upload.return_value = {'Key': 'path/to/file'}
        mock_client.storage.from_.return_value = mock_storage
        mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock()
        mock_get_client.return_value = mock_client

        result = save_csv_file_content(
            file_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            session_id="session-456",
            file_name="test.csv",
            file_path="/path/to/file.csv",
            file_type="input"
        )

        assert result is True
        mock_storage.upload.assert_called_once()

    @patch('shared.database.storage.get_supabase_client')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data=b'csv,data')
    def test_updates_existing_file(self, mock_file, mock_exists, mock_get_client):
        """Test update of existing file."""
        mock_exists.return_value = True
        mock_client = Mock()
        mock_storage = Mock()
        mock_storage.list.return_value = [{'name': 'test.csv'}]
        mock_storage.update.return_value = {'Key': 'path/to/file'}
        mock_client.storage.from_.return_value = mock_storage
        mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock()
        mock_get_client.return_value = mock_client

        result = save_csv_file_content(
            file_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            session_id="session-456",
            file_name="test.csv",
            file_path="/path/to/file.csv",
            file_type="input"
        )

        assert result is True
        mock_storage.update.assert_called_once()

    @patch('shared.database.storage.get_supabase_client')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data=b'csv,data')
    def test_uses_correct_bucket_for_input(self, mock_file, mock_exists, mock_get_client):
        """Test correct bucket is used for input files."""
        mock_exists.return_value = True
        mock_client = Mock()
        mock_storage = Mock()
        mock_storage.list.return_value = []
        mock_storage.upload.return_value = {}
        mock_client.storage.from_.return_value = mock_storage
        mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock()
        mock_get_client.return_value = mock_client

        save_csv_file_content(
            file_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            session_id="session-456",
            file_name="test.csv",
            file_path="/path/to/file.csv",
            file_type="input"
        )

        mock_client.storage.from_.assert_called_with(BucketNames.CSV_INPUT)

    @patch('shared.database.storage.get_supabase_client')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data=b'csv,data')
    def test_uses_correct_bucket_for_output(self, mock_file, mock_exists, mock_get_client):
        """Test correct bucket is used for output files."""
        mock_exists.return_value = True
        mock_client = Mock()
        mock_storage = Mock()
        mock_storage.list.return_value = []
        mock_storage.upload.return_value = {}
        mock_client.storage.from_.return_value = mock_storage
        mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock()
        mock_get_client.return_value = mock_client

        save_csv_file_content(
            file_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            session_id="session-456",
            file_name="test.csv",
            file_path="/path/to/file.csv",
            file_type="output"
        )

        mock_client.storage.from_.assert_called_with(BucketNames.CSV_OUTPUT)

    @patch('shared.database.storage.get_supabase_client')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data=b'csv,data')
    def test_handles_already_exists_gracefully(self, mock_file, mock_exists, mock_get_client):
        """Test file already exists is handled gracefully."""
        mock_exists.return_value = True
        mock_client = Mock()
        mock_storage = Mock()
        mock_storage.list.side_effect = Exception("already exists")
        mock_storage.upload.side_effect = Exception("already exists in bucket")
        mock_client.storage.from_.return_value = mock_storage
        mock_get_client.return_value = mock_client

        result = save_csv_file_content(
            file_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            session_id="session-456",
            file_name="test.csv",
            file_path="/path/to/file.csv",
            file_type="input"
        )

        assert result is True

    @patch('shared.database.storage.get_supabase_client')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data=b'csv,data')
    def test_raises_storage_error_on_upload_failure(self, mock_file, mock_exists, mock_get_client):
        """Test StorageError on upload failure."""
        mock_exists.return_value = True
        mock_client = Mock()
        mock_storage = Mock()
        mock_storage.list.return_value = []
        mock_storage.upload.side_effect = Exception("Network error")
        mock_client.storage.from_.return_value = mock_storage
        mock_get_client.return_value = mock_client

        with pytest.raises(StorageError) as exc_info:
            save_csv_file_content(
                file_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                session_id="session-456",
                file_name="test.csv",
                file_path="/path/to/file.csv",
                file_type="input"
            )

        assert "Error uploading file" in str(exc_info.value)

    @patch('shared.database.storage.get_supabase_client')
    @patch('os.path.exists')
    def test_raises_storage_error_on_read_failure(self, mock_exists, mock_get_client):
        """Test StorageError when file cannot be read."""
        mock_exists.return_value = True
        mock_get_client.return_value = Mock()

        with patch('builtins.open', side_effect=IOError("Permission denied")):
            with pytest.raises(StorageError) as exc_info:
                save_csv_file_content(
                    file_id="file-123",
                    session_id="session-456",
                    file_name="test.csv",
                    file_path="/path/to/file.csv",
                    file_type="input"
                )

        assert "Could not read file" in str(exc_info.value)


class TestDeleteCsvFileContent:
    """Tests for delete_csv_file_content function."""

    @patch('shared.database.storage.get_supabase_client')
    def test_deletes_file_successfully(self, mock_get_client):
        """Test successful file deletion."""
        mock_client = Mock()
        mock_storage = Mock()
        mock_storage.remove.return_value = None
        mock_client.storage.from_.return_value = mock_storage
        mock_get_client.return_value = mock_client

        result = delete_csv_file_content(
            session_id="session-456",
            file_name="test.csv",
            file_type="input"
        )

        assert result is True
        mock_storage.remove.assert_called_once()

    @patch('shared.database.storage.get_supabase_client')
    def test_uses_correct_bucket_for_deletion(self, mock_get_client):
        """Test correct bucket is used for deletion."""
        mock_client = Mock()
        mock_storage = Mock()
        mock_client.storage.from_.return_value = mock_storage
        mock_get_client.return_value = mock_client

        delete_csv_file_content(
            session_id="session-456",
            file_name="test.csv",
            file_type="output"
        )

        mock_client.storage.from_.assert_called_with(BucketNames.CSV_OUTPUT)

    @patch('shared.database.storage.get_supabase_client')
    def test_raises_storage_error_on_deletion_failure(self, mock_get_client):
        """Test StorageError on deletion failure."""
        mock_client = Mock()
        mock_storage = Mock()
        mock_storage.remove.side_effect = Exception("File not found")
        mock_client.storage.from_.return_value = mock_storage
        mock_get_client.return_value = mock_client

        with pytest.raises(StorageError) as exc_info:
            delete_csv_file_content(
                session_id="session-456",
                file_name="test.csv",
                file_type="input"
            )

        assert "Error deleting file" in str(exc_info.value)


class TestGetCsvFileUrl:
    """Tests for get_csv_file_url function."""

    @patch('shared.database.storage.get_supabase_client')
    def test_returns_signed_url(self, mock_get_client):
        """Test signed URL is returned."""
        mock_client = Mock()
        mock_storage = Mock()
        mock_storage.create_signed_url.return_value = {'signedURL': 'https://example.com/signed-url'}
        mock_client.storage.from_.return_value = mock_storage
        mock_get_client.return_value = mock_client

        result = get_csv_file_url(
            session_id="session-456",
            file_name="test.csv",
            file_type="input"
        )

        assert result == 'https://example.com/signed-url'

    @patch('shared.database.storage.get_supabase_client')
    def test_uses_default_expiry(self, mock_get_client):
        """Test default expiry is used."""
        mock_client = Mock()
        mock_storage = Mock()
        mock_storage.create_signed_url.return_value = {'signedURL': 'url'}
        mock_client.storage.from_.return_value = mock_storage
        mock_get_client.return_value = mock_client

        get_csv_file_url(
            session_id="session-456",
            file_name="test.csv",
            file_type="input"
        )

        mock_storage.create_signed_url.assert_called_with(
            path="session-456/test.csv",
            expires_in=3600
        )

    @patch('shared.database.storage.get_supabase_client')
    def test_uses_custom_expiry(self, mock_get_client):
        """Test custom expiry is used."""
        mock_client = Mock()
        mock_storage = Mock()
        mock_storage.create_signed_url.return_value = {'signedURL': 'url'}
        mock_client.storage.from_.return_value = mock_storage
        mock_get_client.return_value = mock_client

        get_csv_file_url(
            session_id="session-456",
            file_name="test.csv",
            file_type="input",
            expiry=7200
        )

        mock_storage.create_signed_url.assert_called_with(
            path="session-456/test.csv",
            expires_in=7200
        )

    @patch('shared.database.storage.get_supabase_client')
    def test_returns_none_when_response_empty(self, mock_get_client):
        """Test None is returned when response is empty."""
        mock_client = Mock()
        mock_storage = Mock()
        mock_storage.create_signed_url.return_value = None
        mock_client.storage.from_.return_value = mock_storage
        mock_get_client.return_value = mock_client

        result = get_csv_file_url(
            session_id="session-456",
            file_name="test.csv",
            file_type="input"
        )

        assert result is None

    @patch('shared.database.storage.get_supabase_client')
    def test_raises_storage_error_on_url_generation_failure(self, mock_get_client):
        """Test StorageError on URL generation failure."""
        mock_client = Mock()
        mock_storage = Mock()
        mock_storage.create_signed_url.side_effect = Exception("URL generation failed")
        mock_client.storage.from_.return_value = mock_storage
        mock_get_client.return_value = mock_client

        with pytest.raises(StorageError) as exc_info:
            get_csv_file_url(
                session_id="session-456",
                file_name="test.csv",
                file_type="input"
            )

        assert "Error generating signed URL" in str(exc_info.value)


class TestBucketSelection:
    """Tests for bucket selection logic."""

    def test_input_bucket_selected_for_input_type(self):
        """Test input bucket is selected for input file type."""
        bucket = BucketNames.get_bucket_for_type("input")
        assert bucket == BucketNames.CSV_INPUT

    def test_output_bucket_selected_for_output_type(self):
        """Test output bucket is selected for output file type."""
        bucket = BucketNames.get_bucket_for_type("output")
        assert bucket == BucketNames.CSV_OUTPUT

    def test_input_bucket_is_default(self):
        """Test input bucket is default for unknown types."""
        bucket = BucketNames.get_bucket_for_type("unknown")
        assert bucket == BucketNames.CSV_INPUT

