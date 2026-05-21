"""Verify upload metadata stores user_id field for IDOR check (foundation)."""
import json
import os
import tempfile
import pytest
from unittest.mock import patch


def test_save_upload_metadata_persists_user_id():
    """save_upload_metadata must persist user_id alongside total_chunks/parameters."""
    from domains.processing.services.local_chunk_service import LocalChunkService

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch('domains.processing.services.local_chunk_service.CHUNK_DIR', tmp_dir):
            service = LocalChunkService()
            upload_id = "test-upload-meta-1"
            user_id = "user-uuid-abc-123"

            ok = service.save_upload_metadata(
                upload_id=upload_id,
                total_chunks=3,
                parameters={'delimiter': ';', 'timezone': 'UTC'},
                user_id=user_id,
            )
            assert ok is True

            # Read raw metadata file to confirm user_id field present
            chunk_dir = service.get_chunk_dir(upload_id)
            metadata_path = os.path.join(chunk_dir, '_metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            assert metadata['user_id'] == user_id
            assert metadata['total_chunks'] == 3
            assert metadata['parameters']['delimiter'] == ';'


def test_get_upload_metadata_returns_user_id():
    """get_upload_metadata must return user_id field when present."""
    from domains.processing.services.local_chunk_service import LocalChunkService

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch('domains.processing.services.local_chunk_service.CHUNK_DIR', tmp_dir):
            service = LocalChunkService()
            upload_id = "test-upload-meta-2"
            user_id = "user-uuid-xyz-456"

            service.save_upload_metadata(
                upload_id=upload_id,
                total_chunks=2,
                parameters={'delimiter': ','},
                user_id=user_id,
            )

            metadata = service.get_upload_metadata(upload_id)
            assert metadata is not None
            assert metadata['user_id'] == user_id
