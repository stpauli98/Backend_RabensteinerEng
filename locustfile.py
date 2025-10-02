#!/usr/bin/env python3
"""
Locust Load Testing Suite za Rabensteiner Engineering Backend
==============================================================

Test scenarios:
- Chunked CSV file uploads (simulate 2-3MB files)
- Real-time Socket.IO progress tracking
- Data processing with different methods (mean, nearest, intrpl)
- Concurrent user simulation

Usage:
    # Web UI mode
    locust -f locustfile.py --host=http://localhost:8080

    # Headless mode
    locust -f locustfile.py --host=http://localhost:8080 \
           --users 50 --spawn-rate 5 --run-time 10m --headless

    # Distributed mode (master)
    locust -f locustfile.py --master --host=http://localhost:8080

    # Distributed mode (worker)
    locust -f locustfile.py --worker --master-host=192.168.1.10
"""

import time
import json
import random
import logging
from io import BytesIO
from datetime import datetime, timedelta

from locust import HttpUser, task, between, events
from locust.exception import StopUser
import socketio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CSV Test Data Generator
# ============================================================================

def generate_csv_content(rows=8000, timestep_minutes=3):
    """
    Generate realistic CSV test data similar to 11.csv

    Args:
        rows: Number of data rows (default: 8000 for ~2MB file)
        timestep_minutes: Time between measurements in minutes

    Returns:
        bytes: CSV content as bytes
    """
    start_time = datetime(2021, 12, 31, 23, 0, 0)

    lines = ["UTC;Hi\n"]

    for i in range(rows):
        current_time = start_time + timedelta(minutes=i * timestep_minutes)
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        # Simulate realistic water level data (1.5 - 2.5 meters)
        value = round(1.5 + random.random(), 2)
        lines.append(f"{timestamp};{value}\n")

    return ''.join(lines).encode('utf-8')


# ============================================================================
# Socket.IO Client Integration
# ============================================================================

class SocketIOClient:
    """Socket.IO client for tracking real-time progress updates"""

    def __init__(self, base_url):
        self.sio = socketio.Client(logger=False, engineio_logger=False)
        self.base_url = base_url
        self.connected = False
        self.progress_updates = []
        self.current_room = None

        # Register event handlers
        self.sio.on('connect', self.on_connect)
        self.sio.on('disconnect', self.on_disconnect)
        self.sio.on('progress', self.on_progress)

    def connect(self):
        """Connect to Socket.IO server"""
        try:
            self.sio.connect(self.base_url, transports=['websocket'])
            self.connected = True
            logger.debug(f"Socket.IO connected to {self.base_url}")
        except Exception as e:
            logger.error(f"Socket.IO connection failed: {e}")
            self.connected = False

    def disconnect(self):
        """Disconnect from Socket.IO server"""
        if self.connected:
            try:
                self.sio.disconnect()
                self.connected = False
                logger.debug("Socket.IO disconnected")
            except Exception as e:
                logger.error(f"Socket.IO disconnect error: {e}")

    def join_room(self, room_id):
        """Join a Socket.IO room for progress tracking"""
        if self.connected:
            try:
                self.sio.emit('join', {'room': room_id})
                self.current_room = room_id
                logger.debug(f"Joined Socket.IO room: {room_id}")
            except Exception as e:
                logger.error(f"Failed to join room {room_id}: {e}")

    def on_connect(self):
        """Handle connection event"""
        logger.debug("Socket.IO client connected")

    def on_disconnect(self):
        """Handle disconnection event"""
        logger.debug("Socket.IO client disconnected")

    def on_progress(self, data):
        """Handle progress update events"""
        self.progress_updates.append({
            'timestamp': time.time(),
            'data': data
        })
        logger.debug(f"Progress update: {data.get('stage')} - {data.get('progress', 0)}%")

    def get_progress_count(self):
        """Get number of progress updates received"""
        return len(self.progress_updates)

    def clear_progress(self):
        """Clear progress history"""
        self.progress_updates = []


# ============================================================================
# Main Locust User Class
# ============================================================================

class DataAdjustmentUser(HttpUser):
    """
    Simulates a user uploading and processing CSV data files

    Task weights:
    - upload_csv_chunked: 3 (most common operation)
    - set_processing_params: 2 (parameter configuration)
    - complete_processing: 1 (final processing step)
    """

    # Wait 1-3 seconds between tasks (realistic user behavior)
    wait_time = between(1, 3)

    # User instance variables
    upload_id = None
    filename = None
    socketio_client = None
    test_data_size = "medium"  # Options: small, medium, large

    def on_start(self):
        """
        Called when a user starts - setup Socket.IO connection
        """
        # Initialize Socket.IO client
        self.socketio_client = SocketIOClient(self.host)
        self.socketio_client.connect()

        # Generate unique identifiers
        timestamp = int(time.time() * 1000)
        self.upload_id = f"load_test_{timestamp}_{random.randint(1000, 9999)}"
        self.filename = f"test_data_{timestamp}.csv"

        # Configure test data size
        size_config = {
            'small': 2000,    # ~500KB
            'medium': 8000,   # ~2MB (like 11.csv)
            'large': 16000    # ~4MB
        }
        self.csv_rows = size_config.get(self.test_data_size, 8000)

        logger.info(f"User started: upload_id={self.upload_id}, data_size={self.test_data_size}")

    def on_stop(self):
        """
        Called when a user stops - cleanup resources
        """
        if self.socketio_client:
            self.socketio_client.disconnect()

        logger.info(f"User stopped: upload_id={self.upload_id}")

    @task(3)
    def upload_csv_chunked(self):
        """
        Task: Upload CSV file in chunks and analyze
        Weight: 3 (highest - most common operation)
        """
        # Generate test CSV data
        csv_content = generate_csv_content(rows=self.csv_rows, timestep_minutes=3)

        # Simulate chunking (5MB chunk size)
        chunk_size = 5 * 1024 * 1024  # 5MB
        total_size = len(csv_content)

        if total_size <= chunk_size:
            # Single chunk upload
            total_chunks = 1
            chunks = [csv_content]
        else:
            # Multiple chunks
            total_chunks = (total_size + chunk_size - 1) // chunk_size
            chunks = [
                csv_content[i * chunk_size:(i + 1) * chunk_size]
                for i in range(total_chunks)
            ]

        # Join Socket.IO room for progress tracking
        room_id = f"adj_{self.upload_id}"
        if self.socketio_client and self.socketio_client.connected:
            self.socketio_client.join_room(room_id)
            self.socketio_client.clear_progress()

        # Upload chunks
        start_time = time.time()

        for chunk_index, chunk_data in enumerate(chunks):
            files = {'files[]': (self.filename, BytesIO(chunk_data), 'text/csv')}
            data = {
                'uploadId': self.upload_id,
                'chunkIndex': chunk_index,
                'totalChunks': total_chunks,
                'filename': self.filename
            }

            with self.client.post(
                "/api/adjustmentsOfData/upload-chunk",
                files=files,
                data=data,
                catch_response=True,
                name="/api/adjustmentsOfData/upload-chunk [chunked]"
            ) as response:
                if response.status_code == 200:
                    result = response.json()

                    # Validate response structure
                    if not result.get('success'):
                        response.failure(f"Upload chunk {chunk_index} returned success=False")

                    # Store upload_id from response
                    if 'uploadId' in result:
                        self.upload_id = result['uploadId']

                    response.success()
                else:
                    response.failure(f"Upload chunk {chunk_index} failed: {response.status_code}")

        upload_duration = time.time() - start_time

        # Validate upload performance (should be < 5s for 2MB file)
        if upload_duration > 5.0:
            events.request.fire(
                request_type="VALIDATION",
                name="Upload Performance Check",
                response_time=upload_duration * 1000,
                response_length=0,
                exception=Exception(f"Upload too slow: {upload_duration:.2f}s > 5s threshold")
            )

        # Check Socket.IO progress updates
        time.sleep(0.5)  # Wait for progress updates
        if self.socketio_client:
            progress_count = self.socketio_client.get_progress_count()
            if progress_count == 0:
                logger.warning(f"No Socket.IO progress updates received for upload {self.upload_id}")

    @task(2)
    def set_processing_params(self):
        """
        Task: Set processing parameters (timestep, offset, method)
        Weight: 2 (parameter configuration)
        """
        if not self.upload_id or not self.filename:
            # Skip if no upload has been done yet
            return

        # Random processing configuration
        methods = ['mean', 'nearest', 'nearest (mean)', 'intrpl']
        selected_method = random.choice(methods)

        payload = {
            "upload_id": self.upload_id,
            "timeStepSize": 1,  # Convert 3-min to 1-min
            "offset": 0,
            "methods": {
                self.filename: {"method": selected_method}
            }
        }

        with self.client.post(
            "/api/adjustmentsOfData/adjust-data-chunk",
            json=payload,
            headers={'Content-Type': 'application/json'},
            catch_response=True,
            name="/api/adjustmentsOfData/adjust-data-chunk [params]"
        ) as response:
            if response.status_code == 200:
                result = response.json()

                # Validate that methods were accepted
                if result.get('methodsRequired'):
                    logger.debug(f"Methods required for {self.filename} (expected)")

                response.success()
            else:
                response.failure(f"Parameter setup failed: {response.status_code}")

    @task(1)
    def complete_processing(self):
        """
        Task: Trigger final data processing
        Weight: 1 (less common - processing step)
        """
        if not self.upload_id:
            # Skip if no upload has been done yet
            return

        payload = {
            "uploadId": self.upload_id
        }

        # Join Socket.IO room for progress tracking
        room_id = f"adj_{self.upload_id}"
        if self.socketio_client and self.socketio_client.connected:
            self.socketio_client.join_room(room_id)
            self.socketio_client.clear_progress()

        start_time = time.time()

        with self.client.post(
            "/api/adjustmentsOfData/adjustdata/complete",
            json=payload,
            headers={'Content-Type': 'application/json'},
            catch_response=True,
            name="/api/adjustmentsOfData/adjustdata/complete"
        ) as response:
            processing_duration = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                if result.get('success'):
                    # Validate processing performance
                    # Optimized backend should process 2MB file in <25s (down from 23s baseline)
                    if processing_duration > 30.0:
                        response.failure(
                            f"Processing too slow: {processing_duration:.2f}s > 30s threshold"
                        )
                    else:
                        response.success()

                    # Check for processed data
                    data = result.get('data', {})
                    processed_files = data.get('result', [])

                    if not processed_files:
                        logger.warning(f"No processed files in response for {self.upload_id}")
                    else:
                        total_records = sum(
                            len(pf.get('records', []))
                            for pf in processed_files
                        )
                        logger.info(
                            f"Processed {len(processed_files)} file(s), "
                            f"{total_records} total records in {processing_duration:.2f}s"
                        )
                else:
                    response.failure("Processing returned success=False")
            else:
                response.failure(f"Processing failed: {response.status_code}")

        # Check Socket.IO progress updates
        time.sleep(0.5)  # Wait for final progress updates
        if self.socketio_client:
            progress_count = self.socketio_client.get_progress_count()
            logger.debug(f"Received {progress_count} progress updates during processing")

            # Store processed data for save test
            if result.get('success'):
                data = result.get('data', {})
                dataframe = data.get('dataframe', [])
                if dataframe:
                    self.processed_data = dataframe

    @task(1)
    def prepare_save_data(self):
        """
        Task: Test /prepare-save endpoint (prepare CSV for download)
        Weight: 1 (less common)
        """
        if not hasattr(self, 'processed_data') or not self.processed_data:
            # Need processed data first
            return

        # Sample the data (send first 100 records to avoid large payloads)
        sample_data = self.processed_data[:100] if len(self.processed_data) > 100 else self.processed_data

        # Convert to CSV-ready format
        csv_data = []
        for record in sample_data:
            # Extract relevant fields
            utc = record.get('UTC', '')
            value = record.get('Hi', record.get('value', ''))
            filename = record.get('filename', '')
            csv_data.append([utc, value, filename])

        payload = {
            'data': csv_data
        }

        with self.client.post(
            "/api/adjustmentsOfData/prepare-save",
            json=payload,
            headers={'Content-Type': 'application/json'},
            catch_response=True,
            name="/api/adjustmentsOfData/prepare-save"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                file_id = result.get('fileId')

                if file_id:
                    response.success()
                    logger.info(f"File prepared for download: {file_id}")
                    self.file_id = file_id  # Store for download test
                else:
                    response.failure("No fileId in response")
            else:
                response.failure(f"Prepare save failed: {response.status_code}")

    @task(1)
    def download_prepared_file(self):
        """
        Task: Test /download/<file_id> endpoint (download prepared CSV)
        Weight: 1 (less common)
        """
        if not hasattr(self, 'file_id') or not self.file_id:
            # Need file_id from prepare-save first
            return

        with self.client.get(
            f"/api/adjustmentsOfData/download/{self.file_id}",
            catch_response=True,
            name="/api/adjustmentsOfData/download/<file_id>"
        ) as response:
            if response.status_code == 200:
                # Check Content-Type header
                content_type = response.headers.get('Content-Type', '')
                if 'text/csv' in content_type or 'application/octet-stream' in content_type:
                    # Validate file size
                    content_length = len(response.content)
                    if content_length > 0:
                        response.success()
                        logger.info(f"Downloaded file {self.file_id}: {content_length} bytes")
                    else:
                        response.failure("Downloaded file is empty")
                else:
                    response.failure(f"Invalid content type: {content_type}")
            elif response.status_code == 404:
                response.failure("File not found (may have expired)")
            else:
                response.failure(f"Download failed: {response.status_code}")


# ============================================================================
# Event Listeners for Custom Metrics
# ============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts"""
    logger.info("=" * 60)
    logger.info("LOAD TEST STARTED")
    logger.info(f"Target host: {environment.host}")
    logger.info("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops"""
    logger.info("=" * 60)
    logger.info("LOAD TEST STOPPED")
    logger.info("=" * 60)

    # Print summary statistics
    stats = environment.stats
    logger.info(f"Total requests: {stats.total.num_requests}")
    logger.info(f"Total failures: {stats.total.num_failures}")
    logger.info(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    logger.info(f"Max response time: {stats.total.max_response_time:.2f}ms")
    logger.info(f"Requests/sec: {stats.total.total_rps:.2f}")


# ============================================================================
# Custom Command Line Arguments
# ============================================================================

@events.init_command_line_parser.add_listener
def add_custom_arguments(parser):
    """Add custom command-line arguments"""
    parser.add_argument(
        "--data-size",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="Size of test CSV data (small=500KB, medium=2MB, large=4MB)"
    )


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize custom settings from command-line arguments"""
    if hasattr(environment.parsed_options, 'data_size'):
        DataAdjustmentUser.test_data_size = environment.parsed_options.data_size
        logger.info(f"Test data size configured: {environment.parsed_options.data_size}")
