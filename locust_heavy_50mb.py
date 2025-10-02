#!/usr/bin/env python3
"""
Locust Heavy Load Test - 50MB CSV Files
========================================

Tests backend performance with large 50MB CSV files (~2.1M rows).
This simulates real-world production load with large datasets.

Usage:
    python3 -m locust -f locust_heavy_50mb.py --host=http://localhost:8080 \
           --users 5 --spawn-rate 1 --run-time 5m --headless
"""

import time
import json
import random
import logging
from io import BytesIO
from datetime import datetime, timedelta

from locust import HttpUser, task, between, events, SequentialTaskSet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 50MB CSV Test Data Generator
# ============================================================================

def generate_large_csv_content(rows=2_097_152, timestep_minutes=3):
    """
    Generate 50MB CSV test data (~2.1M rows)

    Format: 'YYYY-MM-DD HH:MM:SS;1.23\n'
    Average line: ~25 bytes → 2,097,152 rows = ~50MB
    """
    logger.info(f"Generating CSV with {rows:,} rows (~50MB)...")
    start_gen = time.time()

    start_time = datetime(2021, 1, 1, 0, 0, 0)
    lines = ["UTC;Hi\n"]

    # Pre-allocate list for better performance
    lines_capacity = rows + 1

    for i in range(rows):
        current_time = start_time + timedelta(minutes=i * timestep_minutes)
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        value = round(1.5 + random.random(), 2)
        lines.append(f"{timestamp};{value}\n")

        # Progress logging every 200k rows
        if (i + 1) % 200_000 == 0:
            logger.info(f"  Generated {i + 1:,}/{rows:,} rows...")

    csv_content = ''.join(lines).encode('utf-8')
    generation_time = time.time() - start_gen
    actual_size_mb = len(csv_content) / 1024 / 1024

    logger.info(
        f"✅ CSV generation complete: {rows:,} rows, "
        f"{actual_size_mb:.2f}MB in {generation_time:.2f}s"
    )

    return csv_content


# ============================================================================
# Heavy Load Sequential Workflow
# ============================================================================

class HeavyLoadWorkflow(SequentialTaskSet):
    """Sequential workflow for testing with 50MB files"""

    def on_start(self):
        """Initialize with 50MB CSV data"""
        self.upload_id = f"heavy_test_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        self.filename = "heavy_50mb_test.csv"

        # Generate 50MB CSV (~2.1M rows)
        self.csv_content = generate_large_csv_content(rows=2_097_152, timestep_minutes=3)

        self.chunk_size = 5 * 1024 * 1024  # 5MB chunks (optimal for backend)
        self.file_id = None
        self.processed_data = None

        logger.info(f"🔥 Starting HEAVY load test: {self.upload_id}")
        logger.info(f"   File size: {len(self.csv_content) / 1024 / 1024:.2f}MB")
        logger.info(f"   Chunk size: {self.chunk_size / 1024 / 1024:.0f}MB")

    @task
    def step_1_upload_large_chunks(self):
        """STEP 1: Upload 50MB CSV in 5MB chunks"""
        logger.info(f"[STEP 1] Uploading 50MB file: {self.filename}")

        # Split into 5MB chunks
        chunks = []
        for i in range(0, len(self.csv_content), self.chunk_size):
            chunks.append(self.csv_content[i:i + self.chunk_size])

        total_chunks = len(chunks)
        logger.info(f"  Splitting into {total_chunks} chunks of {self.chunk_size / 1024 / 1024:.0f}MB")

        upload_start = time.time()

        # Upload each chunk with progress tracking
        for chunk_index, chunk_data in enumerate(chunks):
            chunk_start = time.time()

            files = {
                'files[]': (self.filename, BytesIO(chunk_data), 'text/csv')
            }

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
                name="/api/adjustmentsOfData/upload-chunk [50MB]",
                timeout=60  # 60s timeout for large chunks
            ) as response:
                chunk_duration = time.time() - chunk_start

                if response.status_code == 200:
                    result = response.json()

                    logger.info(
                        f"  ✅ Chunk {chunk_index + 1}/{total_chunks} uploaded "
                        f"({len(chunk_data) / 1024 / 1024:.1f}MB in {chunk_duration:.2f}s)"
                    )

                    if result.get('status') == 'complete':
                        upload_duration = time.time() - upload_start
                        logger.info(
                            f"[STEP 1] ✅ Upload COMPLETE: {self.filename} "
                            f"({len(self.csv_content) / 1024 / 1024:.2f}MB in {upload_duration:.2f}s)"
                        )
                        response.success()
                    else:
                        response.success()
                else:
                    response.failure(f"Chunk {chunk_index} upload failed: {response.status_code}")
                    return

        time.sleep(1)  # Allow backend to finish processing

    @task
    def step_2_set_parameters_for_large_file(self):
        """STEP 2: Set processing parameters for 50MB file"""
        logger.info(f"[STEP 2] Setting parameters for 50MB file")

        payload = {
            "upload_id": self.upload_id,
            "timeStepSize": 5,  # Downsample from 3min to 5min
            "offset": 0,
            "methods": {
                self.filename: {"method": "mean", "intrpl_max": 120}
            }
        }

        with self.client.post(
            "/api/adjustmentsOfData/adjust-data-chunk",
            json=payload,
            headers={'Content-Type': 'application/json'},
            catch_response=True,
            name="/api/adjustmentsOfData/adjust-data-chunk [50MB]"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[STEP 2] ✅ Parameters set for 50MB file")
                response.success()
            else:
                response.failure(f"Parameter setup failed: {response.status_code}")

        time.sleep(1)

    @task
    def step_3_process_large_file(self):
        """STEP 3: Process 50MB file (this will take time!)"""
        logger.info(f"[STEP 3] Processing 50MB file (this may take 30-60s)...")

        payload = {
            "uploadId": self.upload_id
        }

        start_time = time.time()

        with self.client.post(
            "/api/adjustmentsOfData/adjustdata/complete",
            json=payload,
            headers={'Content-Type': 'application/json'},
            catch_response=True,
            name="/api/adjustmentsOfData/adjustdata/complete [50MB]",
            timeout=120  # 2min timeout for processing
        ) as response:
            processing_duration = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                if result.get('success'):
                    data = result.get('data', {})
                    dataframe = data.get('dataframe', [])

                    if dataframe:
                        self.processed_data = dataframe
                        records_count = len(dataframe)

                        logger.info(
                            f"[STEP 3] ✅ Processing COMPLETE: "
                            f"{records_count:,} records in {processing_duration:.2f}s "
                            f"({records_count / processing_duration:.0f} rec/s)"
                        )
                        response.success()
                    else:
                        response.failure("No processed data in response")
                else:
                    response.failure("Processing returned success=False")
            else:
                response.failure(f"Processing failed: {response.status_code}")

        time.sleep(1)

    @task
    def step_4_prepare_large_dataset(self):
        """STEP 4: Prepare processed data for download"""
        if not self.processed_data:
            logger.warning("[STEP 4] ⚠️ No processed data, skipping")
            return

        logger.info(f"[STEP 4] Preparing {len(self.processed_data):,} records")

        # Sample 500 records for download preparation
        sample_size = min(500, len(self.processed_data))
        sample_data = self.processed_data[:sample_size]

        csv_data = []
        for record in sample_data:
            utc = record.get('UTC', '')
            value = record.get('Hi', record.get('value', ''))
            filename = record.get('filename', '')
            csv_data.append([utc, value, filename])

        payload = {'data': csv_data}

        with self.client.post(
            "/api/adjustmentsOfData/prepare-save",
            json=payload,
            headers={'Content-Type': 'application/json'},
            catch_response=True,
            name="/api/adjustmentsOfData/prepare-save [50MB]"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                file_id = result.get('fileId')

                if file_id:
                    self.file_id = file_id
                    logger.info(f"[STEP 4] ✅ File prepared: {file_id}")
                    response.success()
                else:
                    response.failure("No fileId in response")
            else:
                response.failure(f"Prepare save failed: {response.status_code}")

        time.sleep(1)

    @task
    def step_5_download_result(self):
        """STEP 5: Download prepared file"""
        if not self.file_id:
            logger.warning("[STEP 5] ⚠️ No file_id, skipping")
            return

        logger.info(f"[STEP 5] Downloading: {self.file_id}")

        with self.client.get(
            f"/api/adjustmentsOfData/download/{self.file_id}",
            catch_response=True,
            name="/api/adjustmentsOfData/download/<file_id> [50MB]"
        ) as response:
            if response.status_code == 200:
                content_length = len(response.content)
                logger.info(f"[STEP 5] ✅ Downloaded: {content_length:,} bytes")
                response.success()
            else:
                response.failure(f"Download failed: {response.status_code}")

        logger.info(f"🎉 50MB file test COMPLETE for {self.upload_id}")


# ============================================================================
# Locust User
# ============================================================================

class HeavyLoadUser(HttpUser):
    """User for heavy load testing with 50MB files"""
    wait_time = between(5, 10)  # Longer wait between workflows
    tasks = [HeavyLoadWorkflow]

    def on_start(self):
        logger.info(f"Heavy load user started: {id(self)}")

    def on_stop(self):
        logger.info(f"Heavy load user stopped: {id(self)}")


# ============================================================================
# Event Listeners
# ============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info("=" * 80)
    logger.info("🔥 HEAVY LOAD TEST STARTED - 50MB CSV FILES")
    logger.info(f"Target host: {environment.host}")
    logger.info(f"File size: ~50MB (~2.1M rows)")
    logger.info(f"Chunk size: 5MB")
    logger.info("=" * 80)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    logger.info("=" * 80)
    logger.info("🔥 HEAVY LOAD TEST STOPPED")
    logger.info("=" * 80)

    stats = environment.stats
    logger.info(f"Total requests: {stats.total.num_requests}")
    logger.info(f"Total failures: {stats.total.num_failures}")
    logger.info(f"Failure rate: {stats.total.fail_ratio * 100:.2f}%")
    logger.info(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    logger.info(f"Max response time: {stats.total.max_response_time:.2f}ms")
    logger.info(f"Requests/sec: {stats.total.total_rps:.2f}")
