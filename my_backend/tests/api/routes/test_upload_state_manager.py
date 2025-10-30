"""Tests for UploadStateManager class."""
import time
import threading
import pytest
from api.routes.load_data import UploadStateManager


class TestUploadStateManagerInit:
    """Test UploadStateManager initialization."""
    
    def test_init_creates_empty_storages(self):
        """Should initialize with empty storage dictionaries."""
        manager = UploadStateManager()
        assert len(manager.get_all_upload_ids()) == 0
        with manager._temp_lock:
            assert len(manager._temp_files) == 0
    
    def test_init_creates_locks(self):
        """Should initialize with threading locks."""
        manager = UploadStateManager()
        assert type(manager._chunk_lock).__name__ == 'lock'
        assert type(manager._temp_lock).__name__ == 'lock'


class TestCreateUpload:
    """Test create_upload method."""
    
    def test_create_upload_basic(self):
        """Should create new upload with proper structure."""
        manager = UploadStateManager()
        params = {'delimiter': ',', 'timezone': 'UTC'}
        
        manager.create_upload('test-1', total_chunks=5, parameters=params)
        
        upload = manager.get_upload('test-1')
        assert upload is not None
        assert upload['total_chunks'] == 5
        assert upload['received_chunks'] == 0
        assert upload['parameters'] == params
        assert isinstance(upload['chunks'], dict)
        assert len(upload['chunks']) == 0
        assert 'last_activity' in upload
    
    def test_create_upload_sets_timestamp(self):
        """Should set last_activity timestamp."""
        manager = UploadStateManager()
        before_time = time.time()
        
        manager.create_upload('test-2', total_chunks=3, parameters={})
        
        upload = manager.get_upload('test-2')
        after_time = time.time()
        assert before_time <= upload['last_activity'] <= after_time


class TestUploadExists:
    """Test upload_exists method."""
    
    def test_upload_exists_returns_true(self):
        """Should return True for existing upload."""
        manager = UploadStateManager()
        manager.create_upload('test-3', total_chunks=2, parameters={})
        assert manager.upload_exists('test-3') is True
    
    def test_upload_exists_returns_false(self):
        """Should return False for non-existent upload."""
        manager = UploadStateManager()
        assert manager.upload_exists('non-existent') is False


class TestStoreChunk:
    """Test store_chunk method."""
    
    def test_store_chunk_adds_chunk(self):
        """Should store chunk at specified index."""
        manager = UploadStateManager()
        manager.create_upload('test-4', total_chunks=3, parameters={})
        
        chunk_data = b'test data'
        manager.store_chunk('test-4', chunk_index=0, chunk_data=chunk_data)
        
        chunks = manager.get_chunks('test-4')
        assert chunks is not None
        assert chunks[0] == chunk_data
    
    def test_store_chunk_increments_received_count(self):
        """Should increment received_chunks counter."""
        manager = UploadStateManager()
        manager.create_upload('test-5', total_chunks=3, parameters={})
        
        manager.store_chunk('test-5', chunk_index=0, chunk_data=b'chunk0')
        assert manager.get_chunk_progress('test-5')[0] == 1
        
        manager.store_chunk('test-5', chunk_index=1, chunk_data=b'chunk1')
        assert manager.get_chunk_progress('test-5')[0] == 2
    
    def test_store_chunk_updates_timestamp(self):
        """Should update last_activity timestamp."""
        manager = UploadStateManager()
        manager.create_upload('test-6', total_chunks=2, parameters={})
        
        first_upload = manager.get_upload('test-6')
        first_timestamp = first_upload['last_activity']
        
        time.sleep(0.01)
        manager.store_chunk('test-6', chunk_index=0, chunk_data=b'data')
        
        updated_upload = manager.get_upload('test-6')
        assert updated_upload['last_activity'] > first_timestamp


class TestGetChunkProgress:
    """Test get_chunk_progress method."""
    
    def test_get_chunk_progress_returns_counts(self):
        """Should return tuple of (received, total)."""
        manager = UploadStateManager()
        manager.create_upload('test-7', total_chunks=5, parameters={})
        
        manager.store_chunk('test-7', 0, b'chunk0')
        manager.store_chunk('test-7', 1, b'chunk1')
        
        received, total = manager.get_chunk_progress('test-7')
        assert received == 2
        assert total == 5
    
    def test_get_chunk_progress_non_existent(self):
        """Should return (0, 0) for non-existent upload."""
        manager = UploadStateManager()
        received, total = manager.get_chunk_progress('non-existent')
        assert received == 0
        assert total == 0


class TestIsUploadComplete:
    """Test is_upload_complete method."""
    
    def test_is_upload_complete_returns_true(self):
        """Should return True when all chunks received."""
        manager = UploadStateManager()
        manager.create_upload('test-8', total_chunks=2, parameters={})
        
        manager.store_chunk('test-8', 0, b'chunk0')
        manager.store_chunk('test-8', 1, b'chunk1')
        
        assert manager.is_upload_complete('test-8') is True
    
    def test_is_upload_complete_returns_false(self):
        """Should return False when chunks missing."""
        manager = UploadStateManager()
        manager.create_upload('test-9', total_chunks=3, parameters={})
        
        manager.store_chunk('test-9', 0, b'chunk0')
        
        assert manager.is_upload_complete('test-9') is False
    
    def test_is_upload_complete_non_existent(self):
        """Should return False for non-existent upload."""
        manager = UploadStateManager()
        assert manager.is_upload_complete('non-existent') is False


class TestGetChunks:
    """Test get_chunks method."""
    
    def test_get_chunks_returns_dict(self):
        """Should return chunks dictionary."""
        manager = UploadStateManager()
        manager.create_upload('test-10', total_chunks=2, parameters={})
        
        manager.store_chunk('test-10', 0, b'chunk0')
        manager.store_chunk('test-10', 1, b'chunk1')
        
        chunks = manager.get_chunks('test-10')
        assert chunks is not None
        assert chunks[0] == b'chunk0'
        assert chunks[1] == b'chunk1'
    
    def test_get_chunks_non_existent(self):
        """Should return None for non-existent upload."""
        manager = UploadStateManager()
        assert manager.get_chunks('non-existent') is None


class TestGetParameters:
    """Test get_parameters method."""
    
    def test_get_parameters_returns_params(self):
        """Should return stored parameters."""
        manager = UploadStateManager()
        params = {'delimiter': ';', 'timezone': 'Europe/Berlin'}
        manager.create_upload('test-11', total_chunks=1, parameters=params)
        
        retrieved = manager.get_parameters('test-11')
        assert retrieved == params
    
    def test_get_parameters_non_existent(self):
        """Should return None for non-existent upload."""
        manager = UploadStateManager()
        assert manager.get_parameters('non-existent') is None


class TestDeleteUpload:
    """Test delete_upload method."""
    
    def test_delete_upload_removes_entry(self):
        """Should remove upload from storage."""
        manager = UploadStateManager()
        manager.create_upload('test-12', total_chunks=1, parameters={})
        
        assert manager.upload_exists('test-12') is True
        manager.delete_upload('test-12')
        assert manager.upload_exists('test-12') is False
    
    def test_delete_upload_non_existent(self):
        """Should not raise error for non-existent upload."""
        manager = UploadStateManager()
        # Should not raise exception
        manager.delete_upload('non-existent')


class TestCleanupExpiredUploads:
    """Test cleanup_expired_uploads method."""
    
    def test_cleanup_removes_expired(self):
        """Should remove uploads older than expiry time."""
        manager = UploadStateManager()
        current_time = time.time()
        expiry_seconds = 3600  # 1 hour
        
        # Create expired upload
        manager.create_upload('expired', total_chunks=1, parameters={})
        with manager._chunk_lock:
            manager._chunk_storage['expired']['last_activity'] = current_time - expiry_seconds - 100
        
        # Create active upload
        manager.create_upload('active', total_chunks=1, parameters={})
        
        manager.cleanup_expired_uploads(expiry_seconds=expiry_seconds)
        
        assert manager.upload_exists('expired') is False
        assert manager.upload_exists('active') is True
    
    def test_cleanup_keeps_active_uploads(self):
        """Should keep uploads within expiry window."""
        manager = UploadStateManager()
        expiry_seconds = 3600
        
        manager.create_upload('active-1', total_chunks=1, parameters={})
        manager.create_upload('active-2', total_chunks=1, parameters={})
        
        manager.cleanup_expired_uploads(expiry_seconds=expiry_seconds)
        
        assert manager.upload_exists('active-1') is True
        assert manager.upload_exists('active-2') is True


class TestTempFileOperations:
    """Test temporary file storage operations."""
    
    def test_store_temp_file(self):
        """Should store temporary file information."""
        manager = UploadStateManager()
        manager.store_temp_file('temp-1', file_path='/tmp/file.csv', file_name='file.csv')
        
        temp_data = manager.get_temp_file('temp-1')
        assert temp_data is not None
        assert temp_data['path'] == '/tmp/file.csv'
        assert temp_data['fileName'] == 'file.csv'
    
    def test_get_temp_file_non_existent(self):
        """Should return None for non-existent temp file."""
        manager = UploadStateManager()
        assert manager.get_temp_file('non-existent') is None
    
    def test_delete_temp_file(self):
        """Should remove temporary file entry."""
        manager = UploadStateManager()
        manager.store_temp_file('temp-2', file_path='/tmp/test.csv', file_name='test.csv')
        
        assert manager.get_temp_file('temp-2') is not None
        manager.delete_temp_file('temp-2')
        assert manager.get_temp_file('temp-2') is None


class TestClearAll:
    """Test clear_all method."""
    
    def test_clear_all_removes_everything(self):
        """Should clear both chunk and temp storage."""
        manager = UploadStateManager()
        manager.create_upload('upload-1', total_chunks=1, parameters={})
        manager.store_temp_file('temp-1', file_path='/tmp/file.csv', file_name='file.csv')
        
        manager.clear_all()
        
        assert len(manager.get_all_upload_ids()) == 0
        with manager._temp_lock:
            assert len(manager._temp_files) == 0
    
    def test_clear_all_chunk_only(self):
        """Should clear only chunk storage when specified."""
        manager = UploadStateManager()
        manager.create_upload('upload-1', total_chunks=1, parameters={})
        manager.store_temp_file('temp-1', file_path='/tmp/file.csv', file_name='file.csv')
        
        manager.clear_all(storage_type='chunk')
        
        assert len(manager.get_all_upload_ids()) == 0
        assert manager.get_temp_file('temp-1') is not None
    
    def test_clear_all_temp_only(self):
        """Should clear only temp storage when specified."""
        manager = UploadStateManager()
        manager.create_upload('upload-1', total_chunks=1, parameters={})
        manager.store_temp_file('temp-1', file_path='/tmp/file.csv', file_name='file.csv')
        
        manager.clear_all(storage_type='temp')
        
        assert len(manager.get_all_upload_ids()) == 1
        assert manager.get_temp_file('temp-1') is None


class TestThreadSafety:
    """Test thread-safety of UploadStateManager."""
    
    def test_concurrent_chunk_storage(self):
        """Should handle concurrent chunk storage safely."""
        manager = UploadStateManager()
        manager.create_upload('concurrent-1', total_chunks=100, parameters={})
        
        def store_chunks(start_idx, count):
            for i in range(start_idx, start_idx + count):
                manager.store_chunk('concurrent-1', i, f'chunk{i}'.encode())
        
        # Create 10 threads, each storing 10 chunks
        threads = []
        for t in range(10):
            thread = threading.Thread(target=store_chunks, args=(t * 10, 10))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify all chunks were stored
        received, total = manager.get_chunk_progress('concurrent-1')
        assert received == 100
        assert total == 100
        assert manager.is_upload_complete('concurrent-1') is True
    
    def test_concurrent_upload_creation(self):
        """Should handle concurrent upload creation safely."""
        manager = UploadStateManager()
        
        def create_uploads(start_idx, count):
            for i in range(start_idx, start_idx + count):
                manager.create_upload(f'upload-{i}', total_chunks=1, parameters={'idx': i})
        
        # Create 10 threads, each creating 10 uploads
        threads = []
        for t in range(10):
            thread = threading.Thread(target=create_uploads, args=(t * 10, 10))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify all uploads were created
        upload_ids = manager.get_all_upload_ids()
        assert len(upload_ids) == 100
    
    def test_concurrent_cleanup_and_access(self):
        """Should handle concurrent cleanup and access safely."""
        manager = UploadStateManager()
        expiry_seconds = 1
        
        # Create some uploads
        for i in range(20):
            manager.create_upload(f'upload-{i}', total_chunks=1, parameters={})
        
        # Make half of them expired
        current_time = time.time()
        with manager._chunk_lock:
            for i in range(10):
                manager._chunk_storage[f'upload-{i}']['last_activity'] = current_time - expiry_seconds - 100
        
        def cleanup_repeatedly():
            for _ in range(5):
                manager.cleanup_expired_uploads(expiry_seconds=expiry_seconds)
                time.sleep(0.01)
        
        def access_uploads():
            for _ in range(5):
                for i in range(20):
                    manager.get_upload(f'upload-{i}')
                time.sleep(0.01)
        
        # Run cleanup and access concurrently
        cleanup_thread = threading.Thread(target=cleanup_repeatedly)
        access_thread = threading.Thread(target=access_uploads)
        
        cleanup_thread.start()
        access_thread.start()
        
        cleanup_thread.join()
        access_thread.join()
        
        # Verify expired uploads were cleaned up
        remaining_ids = manager.get_all_upload_ids()
        assert len(remaining_ids) == 10
        
        # Verify active uploads still exist
        for i in range(10, 20):
            assert manager.upload_exists(f'upload-{i}') is True


class TestUploadStateManagerIntegration:
    """Integration tests for UploadStateManager."""
    
    def test_full_upload_workflow(self):
        """Should handle complete upload workflow."""
        manager = UploadStateManager()
        params = {'delimiter': ',', 'timezone': 'UTC', 'has_header': 'ja'}
        
        # Create upload
        manager.create_upload('workflow-1', total_chunks=3, parameters=params)
        assert manager.upload_exists('workflow-1')
        
        # Store chunks
        for i in range(3):
            manager.store_chunk('workflow-1', i, f'chunk{i}'.encode())
        
        # Verify progress
        received, total = manager.get_chunk_progress('workflow-1')
        assert received == 3
        assert total == 3
        assert manager.is_upload_complete('workflow-1')
        
        # Retrieve data
        chunks = manager.get_chunks('workflow-1')
        assert len(chunks) == 3
        
        params_retrieved = manager.get_parameters('workflow-1')
        assert params_retrieved == params
        
        # Clean up
        manager.delete_upload('workflow-1')
        assert not manager.upload_exists('workflow-1')
    
    def test_mixed_operations(self):
        """Should handle mixed chunk and temp file operations."""
        manager = UploadStateManager()
        
        # Create uploads
        manager.create_upload('mixed-1', total_chunks=2, parameters={'test': 'value'})
        manager.create_upload('mixed-2', total_chunks=1, parameters={})
        
        # Store temp files
        manager.store_temp_file('temp-1', file_path='/tmp/file1.csv', file_name='file1.csv')
        manager.store_temp_file('temp-2', file_path='/tmp/file2.csv', file_name='file2.csv')
        
        # Verify all exist
        assert len(manager.get_all_upload_ids()) == 2
        assert manager.get_temp_file('temp-1') is not None
        assert manager.get_temp_file('temp-2') is not None
        
        # Clear only chunks
        manager.clear_all(storage_type='chunk')
        assert len(manager.get_all_upload_ids()) == 0
        assert manager.get_temp_file('temp-1') is not None
        
        # Clear only temps
        manager.clear_all(storage_type='temp')
        assert manager.get_temp_file('temp-1') is None
        assert manager.get_temp_file('temp-2') is None
