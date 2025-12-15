"""Tests for shared.database.operations module (backward compatibility facade).

These tests verify that all symbols are correctly re-exported from the
refactored modules and that the public API remains unchanged.
"""

import pytest


class TestConfigExports:
    """Tests for config module exports."""

    def test_database_config_exported(self):
        """Test DatabaseConfig is exported."""
        from shared.database.operations import DatabaseConfig
        assert DatabaseConfig is not None
        assert hasattr(DatabaseConfig, 'DEFAULT_RETRY_ATTEMPTS')

    def test_domain_defaults_exported(self):
        """Test DomainDefaults is exported."""
        from shared.database.operations import DomainDefaults
        assert DomainDefaults is not None
        assert hasattr(DomainDefaults, 'SKALIERUNG')

    def test_table_names_exported(self):
        """Test TableNames is exported."""
        from shared.database.operations import TableNames
        assert TableNames is not None
        assert hasattr(TableNames, 'SESSIONS')

    def test_bucket_names_exported(self):
        """Test BucketNames is exported."""
        from shared.database.operations import BucketNames
        assert BucketNames is not None
        assert hasattr(BucketNames, 'CSV_INPUT')


class TestExceptionExports:
    """Tests for exception module exports."""

    def test_database_error_exported(self):
        """Test DatabaseError is exported."""
        from shared.database.operations import DatabaseError
        assert issubclass(DatabaseError, Exception)

    def test_session_not_found_error_exported(self):
        """Test SessionNotFoundError is exported."""
        from shared.database.operations import SessionNotFoundError
        from shared.database.operations import DatabaseError
        assert issubclass(SessionNotFoundError, DatabaseError)

    def test_validation_error_exported(self):
        """Test ValidationError is exported."""
        from shared.database.operations import ValidationError
        from shared.database.operations import DatabaseError
        assert issubclass(ValidationError, DatabaseError)

    def test_storage_error_exported(self):
        """Test StorageError is exported."""
        from shared.database.operations import StorageError
        from shared.database.operations import DatabaseError
        assert issubclass(StorageError, DatabaseError)

    def test_configuration_error_exported(self):
        """Test ConfigurationError is exported."""
        from shared.database.operations import ConfigurationError
        from shared.database.operations import DatabaseError
        assert issubclass(ConfigurationError, DatabaseError)


class TestValidatorExports:
    """Tests for validator function exports."""

    def test_validate_session_id_exported(self):
        """Test validate_session_id is exported."""
        from shared.database.operations import validate_session_id
        assert callable(validate_session_id)
        # Test it works
        assert validate_session_id("b2be65df-ce96-4305-b4c7-6530c7bc7096") is True

    def test_validate_file_info_exported(self):
        """Test validate_file_info is exported."""
        from shared.database.operations import validate_file_info
        assert callable(validate_file_info)
        # Test it works
        assert validate_file_info({"fileName": "test.csv"}) is True

    def test_validate_time_info_exported(self):
        """Test validate_time_info is exported."""
        from shared.database.operations import validate_time_info
        assert callable(validate_time_info)
        # Test it works
        assert validate_time_info({}) is True

    def test_sanitize_filename_exported(self):
        """Test sanitize_filename is exported."""
        from shared.database.operations import sanitize_filename
        assert callable(sanitize_filename)
        # Test it works
        assert sanitize_filename("test.csv") == "test.csv"

    def test_underscore_sanitize_filename_exported(self):
        """Test _sanitize_filename alias is exported."""
        from shared.database.operations import _sanitize_filename
        assert callable(_sanitize_filename)


class TestSessionExports:
    """Tests for session management function exports."""

    def test_get_supabase_client_exported(self):
        """Test get_supabase_client is exported."""
        from shared.database.operations import get_supabase_client
        assert callable(get_supabase_client)

    def test_retry_database_operation_exported(self):
        """Test retry_database_operation is exported."""
        from shared.database.operations import retry_database_operation
        assert callable(retry_database_operation)

    def test_create_or_get_session_uuid_exported(self):
        """Test create_or_get_session_uuid is exported."""
        from shared.database.operations import create_or_get_session_uuid
        assert callable(create_or_get_session_uuid)

    def test_get_string_id_from_uuid_exported(self):
        """Test get_string_id_from_uuid is exported."""
        from shared.database.operations import get_string_id_from_uuid
        assert callable(get_string_id_from_uuid)

    def test_get_session_uuid_exported(self):
        """Test get_session_uuid is exported."""
        from shared.database.operations import get_session_uuid
        assert callable(get_session_uuid)

    def test_underscore_get_session_uuid_exported(self):
        """Test _get_session_uuid alias is exported."""
        from shared.database.operations import _get_session_uuid
        assert callable(_get_session_uuid)


class TestPersistenceExports:
    """Tests for persistence function exports."""

    def test_save_time_info_exported(self):
        """Test save_time_info is exported."""
        from shared.database.operations import save_time_info
        assert callable(save_time_info)

    def test_save_zeitschritte_exported(self):
        """Test save_zeitschritte is exported."""
        from shared.database.operations import save_zeitschritte
        assert callable(save_zeitschritte)

    def test_save_file_info_exported(self):
        """Test save_file_info is exported."""
        from shared.database.operations import save_file_info
        assert callable(save_file_info)

    def test_transform_time_info_to_jsonb_exported(self):
        """Test transform_time_info_to_jsonb is exported."""
        from shared.database.operations import transform_time_info_to_jsonb
        assert callable(transform_time_info_to_jsonb)

    def test_underscore_transform_time_info_exported(self):
        """Test _transform_time_info_to_jsonb alias is exported."""
        from shared.database.operations import _transform_time_info_to_jsonb
        assert callable(_transform_time_info_to_jsonb)


class TestStorageExports:
    """Tests for storage function exports."""

    def test_save_csv_file_content_exported(self):
        """Test save_csv_file_content is exported."""
        from shared.database.operations import save_csv_file_content
        assert callable(save_csv_file_content)

    def test_delete_csv_file_content_exported(self):
        """Test delete_csv_file_content is exported."""
        from shared.database.operations import delete_csv_file_content
        assert callable(delete_csv_file_content)

    def test_get_csv_file_url_exported(self):
        """Test get_csv_file_url is exported."""
        from shared.database.operations import get_csv_file_url
        assert callable(get_csv_file_url)


class TestBatchExports:
    """Tests for batch operation function exports."""

    def test_prepare_file_batch_data_exported(self):
        """Test prepare_file_batch_data is exported."""
        from shared.database.operations import prepare_file_batch_data
        assert callable(prepare_file_batch_data)

    def test_batch_upsert_files_exported(self):
        """Test batch_upsert_files is exported."""
        from shared.database.operations import batch_upsert_files
        assert callable(batch_upsert_files)

    def test_underscore_prepare_file_batch_data_exported(self):
        """Test _prepare_file_batch_data alias is exported."""
        from shared.database.operations import _prepare_file_batch_data
        assert callable(_prepare_file_batch_data)

    def test_underscore_batch_upsert_files_exported(self):
        """Test _batch_upsert_files alias is exported."""
        from shared.database.operations import _batch_upsert_files
        assert callable(_batch_upsert_files)


class TestLifecycleExports:
    """Tests for lifecycle function exports."""

    def test_load_session_metadata_exported(self):
        """Test load_session_metadata is exported."""
        from shared.database.operations import load_session_metadata
        assert callable(load_session_metadata)

    def test_save_metadata_to_database_exported(self):
        """Test save_metadata_to_database is exported."""
        from shared.database.operations import save_metadata_to_database
        assert callable(save_metadata_to_database)

    def test_save_files_to_database_exported(self):
        """Test save_files_to_database is exported."""
        from shared.database.operations import save_files_to_database
        assert callable(save_files_to_database)

    def test_finalize_session_exported(self):
        """Test finalize_session is exported."""
        from shared.database.operations import finalize_session
        assert callable(finalize_session)

    def test_update_session_name_exported(self):
        """Test update_session_name is exported."""
        from shared.database.operations import update_session_name
        assert callable(update_session_name)

    def test_save_session_to_supabase_exported(self):
        """Test save_session_to_supabase is exported."""
        from shared.database.operations import save_session_to_supabase
        assert callable(save_session_to_supabase)

    def test_underscore_load_session_metadata_exported(self):
        """Test _load_session_metadata alias is exported."""
        from shared.database.operations import _load_session_metadata
        assert callable(_load_session_metadata)

    def test_underscore_save_metadata_to_database_exported(self):
        """Test _save_metadata_to_database alias is exported."""
        from shared.database.operations import _save_metadata_to_database
        assert callable(_save_metadata_to_database)

    def test_underscore_save_files_to_database_exported(self):
        """Test _save_files_to_database alias is exported."""
        from shared.database.operations import _save_files_to_database
        assert callable(_save_files_to_database)

    def test_underscore_finalize_session_exported(self):
        """Test _finalize_session alias is exported."""
        from shared.database.operations import _finalize_session
        assert callable(_finalize_session)


class TestAllExports:
    """Tests for __all__ export list."""

    def test_all_list_exists(self):
        """Test __all__ list is defined."""
        from shared.database import operations
        assert hasattr(operations, '__all__')
        assert isinstance(operations.__all__, list)

    def test_all_contains_key_functions(self):
        """Test __all__ contains key public functions."""
        from shared.database import operations
        key_exports = [
            'get_supabase_client',
            'create_or_get_session_uuid',
            'save_session_to_supabase',
            'DatabaseError',
            'ValidationError',
            'SessionNotFoundError',
            'DatabaseConfig',
            'TableNames',
            'BucketNames'
        ]
        for export in key_exports:
            assert export in operations.__all__, f"{export} not in __all__"


class TestImportCompatibility:
    """Tests for import path compatibility."""

    def test_direct_import_from_operations(self):
        """Test direct import from operations works."""
        from shared.database.operations import (
            get_supabase_client,
            create_or_get_session_uuid,
            save_session_to_supabase,
            DatabaseError,
            ValidationError
        )
        # All should be importable without error
        assert all([
            get_supabase_client,
            create_or_get_session_uuid,
            save_session_to_supabase,
            DatabaseError,
            ValidationError
        ])

    def test_import_from_package_init(self):
        """Test import from package __init__ works."""
        from shared.database import (
            DatabaseError,
            SessionNotFoundError,
            ValidationError,
            StorageError,
            ConfigurationError,
            DatabaseConfig,
            DomainDefaults,
            TableNames,
            BucketNames
        )
        # All should be importable from package level
        assert all([
            DatabaseError,
            SessionNotFoundError,
            ValidationError,
            StorageError,
            ConfigurationError,
            DatabaseConfig,
            DomainDefaults,
            TableNames,
            BucketNames
        ])

