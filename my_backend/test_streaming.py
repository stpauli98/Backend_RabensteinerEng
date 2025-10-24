#!/usr/bin/env python3
"""
Test script for streaming response functionality
Tests the complete flow: upload -> analyze -> process with streaming
"""

import requests
import socketio
import json
import time
import sys
from pathlib import Path

API_URL = "http://localhost:8080"
SOCKET_URL = "http://localhost:8080"

TEST_FILES = [
    "temp_training_data/0057f6f4-aedf-4c63-8090-648d4df930bd_Leistung.csv",
    "temp_training_data/0057f6f4-aedf-4c63-8090-648d4df930bd_Temp.csv"
]

class StreamingTester:
    def __init__(self):
        self.sio = socketio.Client()
        self.results = {
            'info_df': [],
            'dataframe': []
        }
        self.files_received = 0
        self.total_files = 0
        self.upload_id = None

        self.setup_listeners()

    def setup_listeners(self):
        """Setup SocketIO event listeners"""

        @self.sio.on('connect')
        def on_connect():
            print("✅ SocketIO connected")

        @self.sio.on('disconnect')
        def on_disconnect():
            print("❌ SocketIO disconnected")

        @self.sio.on('processing_progress')
        def on_progress(data):
            if data.get('uploadId') == self.upload_id:
                print(f"📊 Progress: {data.get('progress')}% - {data.get('message')}")

        @self.sio.on('file_result')
        def on_file_result(data):
            if data.get('uploadId') == self.upload_id:
                print(f"\n🔵 FILE_RESULT received for: {data.get('filename')}")
                print(f"   - Chunked: {data.get('chunked', False)}")
                print(f"   - File {data.get('fileIndex', 0) + 1}/{data.get('totalFiles', 0)}")

                if data.get('info_record'):
                    self.results['info_df'].append(data['info_record'])
                    print(f"   - Info record added ✓")

                if not data.get('chunked') and data.get('dataframe_chunk'):
                    self.results['dataframe'].extend(data['dataframe_chunk'])
                    print(f"   - Added {len(data['dataframe_chunk'])} rows")

                self.files_received += 1

        @self.sio.on('dataframe_chunk')
        def on_dataframe_chunk(data):
            if data.get('uploadId') == self.upload_id:
                chunk_idx = data.get('chunkIndex', 0)
                total_chunks = data.get('totalChunks', 0)
                print(f"🟢 DATAFRAME_CHUNK: {chunk_idx + 1}/{total_chunks} for {data.get('filename')}")

                if data.get('chunk'):
                    self.results['dataframe'].extend(data['chunk'])
                    print(f"   - Added {len(data['chunk'])} rows")

        @self.sio.on('file_error')
        def on_file_error(data):
            if data.get('uploadId') == self.upload_id:
                print(f"🔴 FILE_ERROR: {data.get('filename')} - {data.get('error')}")
                self.files_received += 1

    def upload_file(self, file_path):
        """Upload a file in chunks"""
        print(f"\n📤 Uploading: {file_path}")

        import uuid
        upload_id = f"test_{uuid.uuid4()}"

        with open(file_path, 'r') as f:
            content = f.read()

        form_data = {
            'uploadId': upload_id,
            'chunkIndex': '0',
            'totalChunks': '1',
            'filename': Path(file_path).name
        }

        files = {'files[]': ('file.csv', content, 'text/csv')}

        response = requests.post(
            f"{API_URL}/api/adjustmentsOfData/upload-chunk",
            data=form_data,
            files=files
        )

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Upload successful: {data.get('message')}")
            return upload_id, data.get('data', {}).get('info_df', [{}])[0]
        else:
            print(f"❌ Upload failed: {response.text}")
            return None, None

    def process_data(self, upload_ids, file_info):
        """Process uploaded data"""
        print(f"\n🔄 Processing data...")

        self.upload_id = list(upload_ids.values())[0]
        self.total_files = len(upload_ids)

        self.sio.emit('join', {'uploadId': self.upload_id})
        print(f"📡 Joined SocketIO room: {self.upload_id}")

        first_info = list(file_info.values())[0]

        print(f"\n📝 Step 1: Sending parameters via /adjust-data-chunk...")

        adjust_payload = {
            'upload_id': self.upload_id,
            'startTime': None,
            'endTime': None,
            'timeStepSize': first_info.get('Zeitschrittweite [min]', 15),
            'offset': first_info.get('Offset [min]', 0),
            'methods': {}
        }

        adjust_response = requests.post(
            f"{API_URL}/api/adjustmentsOfData/adjust-data-chunk",
            json=adjust_payload,
            headers={'Content-Type': 'application/json'}
        )

        if adjust_response.status_code != 200:
            print(f"❌ Parameter adjustment failed: {adjust_response.text}")
            return False

        print(f"✅ Parameters set successfully")

        print(f"\n📝 Step 2: Completing processing...")

        payload = {
            'uploadId': self.upload_id,
            'totalChunks': 1,
            'files': list(upload_ids.keys()),
            'startTime': None,
            'endTime': None,
            'timeStepSize': first_info.get('Zeitschrittweite [min]', 15),
            'offset': first_info.get('Offset [min]', 0),
            'methods': {}
        }

        print(f"📝 Complete request payload: {json.dumps(payload, indent=2)}")

        response = requests.post(
            f"{API_URL}/api/adjustmentsOfData/adjustdata/complete",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Complete request successful")
            print(f"📊 Response: {json.dumps(data, indent=2)}")

            if data.get('streaming'):
                print(f"\n🌊 STREAMING MODE ACTIVE!")
                print(f"   - Total files: {data.get('totalFiles')}")
                print(f"   - Waiting for streaming results...")

                timeout = 30
                start_time = time.time()

                while self.files_received < self.total_files:
                    if time.time() - start_time > timeout:
                        print(f"\n⏰ Timeout waiting for streaming results")
                        break
                    time.sleep(0.1)

                if self.files_received == self.total_files:
                    print(f"\n✅ All {self.total_files} files received via streaming!")
                    return True
            else:
                print(f"\n📦 Traditional response (not streaming)")
                self.results['info_df'] = data.get('data', {}).get('info_df', [])
                self.results['dataframe'] = data.get('data', {}).get('dataframe', [])
                return True
        else:
            print(f"\n❌ Complete request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    def run_test(self):
        """Run complete test"""
        print("=" * 60)
        print("🧪 STREAMING RESPONSE TEST")
        print("=" * 60)

        try:
            print("\n📡 Connecting to SocketIO...")
            self.sio.connect(SOCKET_URL)

            upload_ids = {}
            file_info = {}

            for test_file in TEST_FILES:
                upload_id, info = self.upload_file(test_file)
                if upload_id:
                    upload_ids[Path(test_file).name] = upload_id
                    file_info[Path(test_file).name] = info

            if not upload_ids:
                print("\n❌ No files uploaded successfully")
                return False

            success = self.process_data(upload_ids, file_info)

            print("\n" + "=" * 60)
            print("📊 RESULTS SUMMARY")
            print("=" * 60)
            print(f"✓ Info records: {len(self.results['info_df'])}")
            print(f"✓ Dataframe rows: {len(self.results['dataframe'])}")

            if self.results['info_df']:
                print("\n📋 Info Records:")
                for i, info in enumerate(self.results['info_df']):
                    print(f"   {i + 1}. {info.get('Name der Datei', 'N/A')}")
                    print(f"      - Rows: {info.get('Anzahl der Datenpunkte', 0)}")
                    print(f"      - Timestep: {info.get('Zeitschrittweite [min]', 0)} min")

            if self.results['dataframe']:
                print(f"\n📈 Dataframe sample (first 3 rows):")
                for i, row in enumerate(self.results['dataframe'][:3]):
                    print(f"   {i + 1}. {row}")

            print("\n" + "=" * 60)

            if success:
                print("✅ TEST PASSED")
            else:
                print("❌ TEST FAILED")

            print("=" * 60)

            return success

        except Exception as e:
            print(f"\n❌ Test error: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            if self.sio.connected:
                self.sio.disconnect()
            print("\n🔌 Disconnected from SocketIO")

if __name__ == "__main__":
    tester = StreamingTester()
    success = tester.run_test()
    sys.exit(0 if success else 1)
