import requests
import json
import os
import tempfile
import subprocess
import re
import sys
import pandas as pd

# Configuration
BACKEND_URL = "http://127.0.0.1:8080"
ORIGINAL_SCRIPT_PATH = "training_backend_test_2.py"

def fetch_data(endpoint, session_id):
    """Fetches JSON data from the backend."""
    url = f"{BACKEND_URL}{endpoint.replace('<session_id>', session_id)}"
    print(f"Fetching data from: {url}")
    response = requests.get(url)
    response.raise_for_status() # Raise an exception for HTTP errors
    return response.json()

def download_file(session_id, file_type, file_name, temp_dir):
    """Downloads a file from the backend and saves it to a temporary directory."""
    url = f"{BACKEND_URL}/api/training/file/download/{session_id}/{file_type}/{file_name}"
    print(f"Downloading file from: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    temp_file_path = os.path.join(temp_dir, file_name)
    with open(temp_file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"File saved to: {temp_file_path}")
    return temp_file_path

def run_training_script(session_id):
    """
    Fetches data from backend, modifies the training script, runs it, and cleans up.
    """
    temp_dir = None
    modified_script_path = None
    try:
        # 1. Get UUID for the session_id
        print(f"Getting UUID for session ID: {session_id}")
        uuid_response = fetch_data(f"/api/training/get-session-uuid/<session_id>", session_id)
        if not uuid_response.get("success"):
            raise Exception(f"Failed to get UUID for session: {uuid_response.get('error', 'Unknown error')}")
        uuid_session_id = uuid_response["sessionUuid"]
        print(f"Resolved UUID for {session_id}: {uuid_session_id}")

        # 2. Fetch time_info and zeitschritte
        print("Fetching time_info and zeitschritte...")
        # Using the test-data-loading endpoint as it provides both
        data_loading_response = fetch_data(f"/api/training/test-data-loading/<session_id>", session_id)
        if not data_loading_response.get("success"):
            raise Exception(f"Failed to fetch time_info/zeitschritte: {data_loading_response.get('error', 'Unknown error')}")
        
        time_info = data_loading_response["data"]["time_info"]
        zeitschritte = data_loading_response["data"]["zeitschritte"]
        print("Time info and Zeitschritte fetched.")

        # 3. Fetch file metadata
        print("Fetching file metadata...")
        files_metadata_response = fetch_data(f"/api/training/get-all-files-metadata/<session_id>", session_id)
        if not files_metadata_response.get("success"):
            raise Exception(f"Failed to fetch file metadata: {files_metadata_response.get('error', 'Unknown error')}")
        files_metadata = files_metadata_response["files"]
        print(f"Fetched {len(files_metadata)} file metadata entries.")

        # 4. Create a temporary directory for downloaded files and modified script
        temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {temp_dir}")

        # 5. Download files and map original names to temp paths
        downloaded_files_by_type = {
            'input': [],
            'output': []
        }

        print(f"DEBUG: files_metadata received by middleman_runner: {files_metadata}")
        for file_meta in files_metadata:
            print(f"DEBUG: Processing individual file_meta: {file_meta}")
            file_name = file_meta["fileName"]
            file_type = file_meta["type"]
            
            downloaded_path = download_file(uuid_session_id, file_type, file_name, temp_dir)
            downloaded_files_by_type[file_type].append(downloaded_path)
        print("All necessary files downloaded.")

        # Define the hardcoded paths in training_backend_test_2.py in order
        # Based on manual inspection of training_backend_test_2.py
        hardcoded_input_paths = [
            "data/historical/grid load/data_4/load_grid_kW_Krumpendorf.csv", # First input file
            "data/historical/solarthermics/data_4/Wert 1.csv", # Second input file (alternative path)
            "data/historical/grid load/data_4/t_out_grad_C_Krumpendorf.csv", # Third input file
            "data/historical/solarthermics/data_4/Wert 2.csv" # Fourth input file (alternative path)
        ]
        hardcoded_output_paths = [
            "data/historical/grid load/data_4/load_grid_kW_Krumpendorf.csv", # First output file
            "data/historical/solarthermics/data_4/Wert 2.csv", # Second output file (alternative path)
            "data/historical/grid load/data_4/Wert 3.csv" # Third output file (alternative path)
        ]

        # Check for file quantity mismatch
        if len(downloaded_files_by_type['input']) < len(hardcoded_input_paths):
            print(f"WARNING: Expected at least {len(hardcoded_input_paths)} input files but only downloaded {len(downloaded_files_by_type['input'])}. Script may fail.")
        if len(downloaded_files_by_type['output']) < len(hardcoded_output_paths):
            print(f"WARNING: Expected at least {len(hardcoded_output_paths)} output files but only downloaded {len(downloaded_files_by_type['output'])}. Script may fail.")

        # Perform replacements
        # Replace input file paths
        for i, old_path in enumerate(hardcoded_input_paths):
            if i < len(downloaded_files_by_type['input']):
                new_path = downloaded_files_by_type['input'][i]
                # This regex looks for 'path = "OLD_PATH"' and replaces it.
                # It's designed to be as broad as possible to catch variations.
                # It will replace the first occurrence of the old_path.
                pattern = re.compile(r'(path\s*=\s*"' + re.escape(old_path) + r'")', re.MULTILINE)
                
                # Use subn to get the count of replacements
                modified_content, num_replacements = pattern.subn(f'path = "{new_path}"', modified_content, count=1) # Replace only first occurrence
                if num_replacements > 0:
                    print(f"Replaced input path '{old_path}' with '{new_path}' ({num_replacements} occurrence).")
                else:
                    print(f"Warning: Could not find hardcoded input path '{old_path}' for replacement.")

        # Replace output file paths
        for i, old_path in enumerate(hardcoded_output_paths):
            if i < len(downloaded_files_by_type['output']):
                new_path = downloaded_files_by_type['output'][i]
                pattern = re.compile(r'(path\s*=\s*"' + re.escape(old_path) + r'")', re.MULTILINE)
                modified_content, num_replacements = pattern.subn(f'path = "{new_path}"', modified_content, count=1) # Replace only first occurrence
                if num_replacements > 0:
                    print(f"Replaced output path '{old_path}' with '{new_path}' ({num_replacements} occurrence).")
                else:
                    print(f"Warning: Could not find hardcoded output path '{old_path}' for replacement.")
        print("File paths modified.")

        # 8. Write the modified content to a temporary file
        modified_script_path = os.path.join(temp_dir, "modified_training_script.py")
        with open(modified_script_path, 'w') as f:
            f.write(modified_content)
        print(f"Modified script saved to: {modified_script_path}")

        # 9. Execute the modified script
        print("Executing modified script...")
        print(f"DEBUG: Python interpreter used by subprocess: {sys.executable}")
        result = subprocess.run([sys.executable, modified_script_path], capture_output=True, text=True, check=False)
        print("Script Stdout:\n", result.stdout)
        if result.stderr:
            print("Script Stderr:\n", result.stderr)
        if result.returncode != 0:
            print(f"Script exited with error code: {result.returncode}")
            raise Exception("Modified script execution failed.")
        print("Modified script execution completed.")

    except requests.exceptions.RequestException as e:
        print(f"Network or HTTP error: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # 10. Clean up temporary files
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        session_id_to_run = sys.argv[1]
        run_training_script(session_id_to_run)
    else:
        print("Usage: python middleman_runner.py <session_id>")
        print("Please provide a session ID as a command-line argument.")