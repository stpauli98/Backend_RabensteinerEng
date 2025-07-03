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
ORIGINAL_SCRIPT_PATH = "/Users/posao/Documents/GitHub/Backend_RabensteinerEng/my_backend/training_backend_test_2.py"

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
        file_path_map = {}
        for file_meta in files_metadata:
            file_name = file_meta["fileName"]
            file_type = file_meta["type"] # 'input' or 'output'
            
            # The 'bezeichnung' field from your metadata seems to be the key used in the original script
            # e.g., "Netzlast [kW]", "Aussentemperatur Krumpendorf [GradC]", "a"
            original_script_name = file_meta.get("bezeichnung", os.path.splitext(file_name)[0]) # Fallback to filename without extension

            downloaded_path = download_file(uuid_session_id, file_type, file_name, temp_dir)
            file_path_map[original_script_name] = downloaded_path
        print("All necessary files downloaded.")

        # 6. Read the original script content
        with open(ORIGINAL_SCRIPT_PATH, 'r') as f:
            script_content = f.read()

        # 7. Modify the script content
        modified_content = script_content

        # Modify MTS class
        if zeitschritte:
            modified_content = re.sub(r"I_N\s*=\s*\d+", f"I_N = {zeitschritte.get('eingabe', 13)}", modified_content)
            modified_content = re.sub(r"O_N\s*=\s*\d+", f"O_N = {zeitschritte.get('ausgabe', 13)}", modified_content)
            modified_content = re.sub(r"DELT\s*=\s*\d+", f"DELT = {zeitschritte.get('zeitschrittweite', 3)}", modified_content)
            modified_content = re.sub(r"OFST\s*=\s*\d+", f"OFST = {zeitschritte.get('offset', 0)}", modified_content)
        print("MTS class modified.")

        # Modify T class (Time Information)
        if time_info and time_info.get("category_data"):
            t_class_str_parts = []
            t_class_str_parts.append("class T:")
            t_class_str_parts.append(f"    TZ = \"{time_info.get('zeitzone', 'Europe/Vienna')}\"")

            categories = {
                "jahr": "Y", "monat": "M", "woche": "W", "tag": "D", "feiertag": "H"
            }
            for category_key, class_name in categories.items():
                cat_data = time_info["category_data"].get(category_key, {})
                imp = time_info.get(category_key, False)
                spec = cat_data.get("datenform", "Aktuelle Zeit")
                th_strt = cat_data.get("zeithorizontStart", 0)
                th_end = cat_data.get("zeithorizontEnd", 0)
                scal = cat_data.get("skalierung", "nein") == "ja"
                scal_max = cat_data.get("skalierungMax", 1)
                scal_min = cat_data.get("skalierungMin", 0)
                
                t_class_str_parts.append(f"    class {class_name}:")
                t_class_str_parts.append(f"        IMP = {imp}")
                # Assuming 'detaillierteBerechnung' maps to 'LT'
                t_class_str_parts.append(f"        LT = {cat_data.get('detaillierteBerechnung', False)}") 
                t_class_str_parts.append(f"        SPEC = \"{spec}\"")
                t_class_str_parts.append(f"        TH_STRT = {th_strt}")
                t_class_str_parts.append(f"        TH_END = {th_end}")
                t_class_str_parts.append(f"        SCAL = {scal}")
                t_class_str_parts.append(f"        SCAL_MAX = {scal_max}")
                t_class_str_parts.append(f"        SCAL_MIN = {scal_min}")
                if category_key == "feiertag":
                    t_class_str_parts.append(f"        CNTRY = \"{cat_data.get('land', 'Österreich')}\"")
                # Replicate original DELT calculation
                t_class_str_parts.append(f"        DELT = (TH_END-TH_STRT)*60/(MTS.I_N-1)")
                t_class_str_parts.append("") # Empty line for formatting

            t_class_str = "\n".join(t_class_str_parts) # Use \n for literal newline in regex replacement

            # Find the exact start and end of the T class block
            t_class_start_marker = "# ZEITINFORMATION #############################################################"
            t_class_end_marker = "# AUSGABEDATEN ################################################################"
            
            # Find the content between markers
            match = re.search(f"{re.escape(t_class_start_marker)}(.*?){re.escape(t_class_end_marker)}", modified_content, re.DOTALL)

            if match:
                # Replace the content between markers with the new T class string
                modified_content = modified_content.replace(match.group(1), f"\n{t_class_str}\n")
            else:
                print("Warning: Could not find T class block for modification.")
        print("T class modified.")

        # Modify file paths (i_dat and o_dat sections)
        # This is highly dependent on the exact structure of the original script.
        # It looks for 'name = "..."' followed by 'path = "..."' and replaces the path.
        
        # Create a combined list of names to search for in the script
        # This is to handle cases where the script might use a different name than 'bezeichnung'
        # or if 'bezeichnung' is empty.
        script_names_to_paths = {}
        for file_meta in files_metadata:
            file_name = file_meta["fileName"]
            bezeichnung = file_meta.get("bezeichnung")
            downloaded_path = file_path_map.get(bezeichnung if bezeichnung else os.path.splitext(file_name)[0])

            if downloaded_path:
                # Add mapping for 'bezeichnung'
                if bezeichnung:
                    script_names_to_paths[bezeichnung] = downloaded_path
                # Add mapping for filename without extension (common fallback)
                script_names_to_paths[os.path.splitext(file_name)[0]] = downloaded_path
                # Add mapping for full filename (less common for 'name' in script, but good to have)
                script_names_to_paths[file_name] = downloaded_path

        # Iterate through the script_names_to_paths and replace paths
        for script_name, downloaded_path in script_names_to_paths.items():
            escaped_script_name = re.escape(script_name)
            # This regex looks for 'name = "SCRIPT_NAME"' followed by any characters (including newlines)
            # until it finds 'path = "..."' and captures the entire path line.
            # It's designed to be as broad as possible to catch variations.
            pattern = re.compile(r'(name\s*=\s*\"' + escaped_script_name + r'\"(?:.|\n)*?path\s*=\s*\"[^\"]*\")', re.MULTILINE)
            
            # Find all occurrences and replace
            modified_content, num_replacements = pattern.subn(lambda m: re.sub(r'path\s*=\s*\"[^\"]*\"', f'path = "{downloaded_path}"', m.group(0)), modified_content)
            if num_replacements > 0:
                print(f"Modified {num_replacements} path(s) for '{script_name}' to '{downloaded_path}'")
            # else:
            #     print(f"Warning: No path modification found for '{script_name}'.")
        print("File paths modified.")


        # 8. Write the modified content to a temporary file
        modified_script_path = os.path.join(temp_dir, "modified_training_script.py")
        with open(modified_script_path, 'w') as f:
            f.write(modified_content)
        print(f"Modified script saved to: {modified_script_path}")

        # 9. Execute the modified script
        print("Executing modified script...")
        result = subprocess.run(['python', modified_script_path], capture_output=True, text=True, check=False)
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