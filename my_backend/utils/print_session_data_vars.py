import requests
import json
import sys
import os

BACKEND_URL = "http://127.0.0.1:8080"

def fetch_data(endpoint, session_id):
    """Fetches JSON data from the backend."""
    url = f"{BACKEND_URL}{endpoint.replace('<session_id>', session_id)}"
    print(f"Fetching data from: {url}")
    response = requests.get(url)
    response.raise_for_status() # Raise an exception for HTTP errors
    return response.json()

def download_file_content(session_uuid, file_type, file_name):
    """Downloads file content from the backend."""
    url = f"{BACKEND_URL}/api/training/file/download/{session_uuid}/{file_type}/{file_name}"
    print(f"Downloading content for file: {file_name} (type: {file_type})")
    response = requests.get(url)
    response.raise_for_status()
    return response.content

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print_session_data_vars.py <session_id>")
        sys.exit(1)

    session_id_arg = sys.argv[1]

    print(f"\n--- Fetching and Assigning Data for Session: {session_id_arg} ---")

    try:
        # 1. Get UUID for the session_id
        uuid_response = fetch_data(f"/api/training/get-session-uuid/<session_id>", session_id_arg)
        if not uuid_response.get("success"):
            raise Exception(f"Failed to get UUID for session: {uuid_response.get('error', 'Unknown error')}")
        session_uuid = uuid_response["sessionUuid"]
        print(f"session_uuid = '{session_uuid}'")

        # 2. Fetch time_info and zeitschritte
        data_loading_response = fetch_data(f"/api/training/test-data-loading/<session_id>", session_id_arg)
        if not data_loading_response.get("success"):
            raise Exception(f"Failed to fetch time_info/zeitschritte: {data_loading_response.get('error', 'Unknown error')}")
        
        time_info_data = data_loading_response["data"]["time_info"]
        zeitschritte_data = data_loading_response["data"]["zeitschritte"]

        print("\n--- Time Information (time_info) ---")
        print(f"time_info_data = {json.dumps(time_info_data, indent=2)}")
        if time_info_data:
            time_info_id = time_info_data.get("id")
            time_info_session_id = time_info_data.get("session_id")
            time_info_jahr = time_info_data.get("jahr")
            time_info_woche = time_info_data.get("woche")
            time_info_monat = time_info_data.get("monat")
            time_info_feiertag = time_info_data.get("feiertag")
            time_info_tag = time_info_data.get("tag")
            time_info_zeitzone = time_info_data.get("zeitzone")
            time_info_category_data = time_info_data.get("category_data")
            print(f"time_info_id = '{time_info_id}'")
            print(f"time_info_session_id = '{time_info_session_id}'")
            print(f"time_info_jahr = {time_info_jahr}")
            print(f"time_info_woche = {time_info_woche}")
            print(f"time_info_monat = {time_info_monat}")
            print(f"time_info_feiertag = {time_info_feiertag}")
            print(f"time_info_tag = {time_info_tag}")
            print(f"time_info_zeitzone = '{time_info_zeitzone}'")
            print(f"time_info_category_data = {json.dumps(time_info_category_data, indent=2)}")

        print("\n--- Zeitschritte Information ---")
        print(f"zeitschritte_data = {json.dumps(zeitschritte_data, indent=2)}")
        if zeitschritte_data:
            zeitschritte_id = zeitschritte_data.get("id")
            zeitschritte_session_id = zeitschritte_data.get("session_id")
            zeitschritte_eingabe = zeitschritte_data.get("eingabe")
            zeitschritte_ausgabe = zeitschritte_data.get("ausgabe")
            zeitschritte_zeitschrittweite = zeitschritte_data.get("zeitschrittweite")
            zeitschritte_offset = zeitschritte_data.get("offset")
            print(f"zeitschritte_id = '{zeitschritte_id}'")
            print(f"zeitschritte_session_id = '{zeitschritte_session_id}'")
            print(f"zeitschritte_eingabe = '{zeitschritte_eingabe}'")
            print(f"zeitschritte_ausgabe = '{zeitschritte_ausgabe}'")
            print(f"zeitschritte_zeitschrittweite = '{zeitschritte_zeitschrittweite}'")
            print(f"zeitschritte_offset = '{zeitschritte_offset}'")

        # 3. Fetch file metadata
        files_metadata_response = fetch_data(f"/api/training/get-all-files-metadata/<session_id>", session_id_arg)
        if not files_metadata_response.get("success"):
            raise Exception(f"Failed to fetch file metadata: {files_metadata_response.get('error', 'Unknown error')}")
        files_metadata_list = files_metadata_response["files"]

        print(f"\n--- File Metadata ({len(files_metadata_list)} files) ---")
        for i, file_meta in enumerate(files_metadata_list):
            print(f"\nFile {i+1} Metadata:")
            print(f"file_{i+1}_meta_data = {json.dumps(file_meta, indent=2)}")
            
            file_id = file_meta.get("id")
            file_name = file_meta.get("fileName")
            file_bezeichnung = file_meta.get("bezeichnung")
            file_type = file_meta.get("type")
            file_utc_min = file_meta.get("utcMin")
            file_utc_max = file_meta.get("utcMax")
            file_zeitschrittweite = file_meta.get("zeitschrittweite")
            file_min = file_meta.get("min")
            file_max = file_meta.get("max")
            file_offsett = file_meta.get("offsett")
            file_datenpunkte = file_meta.get("datenpunkte")
            file_numerische_datenpunkte = file_meta.get("numerischeDatenpunkte")
            file_numerischer_anteil = file_meta.get("numerischerAnteil")
            file_datenform = file_meta.get("datenform")
            file_zeithorizont_start = file_meta.get("zeithorizontStart")
            file_zeithorizont_end = file_meta.get("zeithorizontEnd")
            file_zeitschrittweite_transferierten_daten = file_meta.get("zeitschrittweiteTransferiertenDaten")
            file_offset_transferierten_daten = file_meta.get("offsetTransferiertenDaten")
            file_mittelwertbildung_uber_den_zeithorizont = file_meta.get("mittelwertbildung_uber_den_zeithorizont")
            file_datenanpassung = file_meta.get("datenanpassung")
            file_zeitschrittweite_mittelwert = file_meta.get("zeitschrittweiteMittelwert")
            file_zeitschrittweite_min = file_meta.get("zeitschrittweiteMin")
            file_skalierung = file_meta.get("skalierung")
            file_skalierung_max = file_meta.get("skalierungMax")
            file_skalierung_min = file_meta.get("skalierungMin")
            file_storage_path = file_meta.get("storagePath")

            print(f"file_{i+1}_id = '{file_id}'")
            print(f"file_{i+1}_name = '{file_name}'")
            print(f"file_{i+1}_bezeichnung = '{file_bezeichnung}'")
            print(f"file_{i+1}_type = '{file_type}'")
            print(f"file_{i+1}_utc_min = '{file_utc_min}'")
            print(f"file_{i+1}_utc_max = '{file_utc_max}'")
            print(f"file_{i+1}_zeitschrittweite = '{file_zeitschrittweite}'")
            print(f"file_{i+1}_min = '{file_min}'")
            print(f"file_{i+1}_max = '{file_max}'")
            print(f"file_{i+1}_offsett = '{file_offsett}'")
            print(f"file_{i+1}_datenpunkte = '{file_datenpunkte}'")
            print(f"file_{i+1}_numerische_datenpunkte = '{file_numerische_datenpunkte}'")
            print(f"file_{i+1}_numerischer_anteil = '{file_numerischer_anteil}'")
            print(f"file_{i+1}_datenform = '{file_datenform}'")
            print(f"file_{i+1}_zeithorizont_start = '{file_zeithorizont_start}'")
            print(f"file_{i+1}_zeithorizont_end = '{file_zeithorizont_end}'")
            print(f"file_{i+1}_zeitschrittweite_transferierten_daten = '{file_zeitschrittweite_transferierten_daten}'")
            print(f"file_{i+1}_offset_transferierten_daten = '{file_offset_transferierten_daten}'")
            print(f"file_{i+1}_mittelwertbildung_uber_den_zeithorizont = '{file_mittelwertbildung_uber_den_zeithorizont}'")
            print(f"file_{i+1}_datenanpassung = '{file_datenanpassung}'")
            print(f"file_{i+1}_zeitschrittweite_mittelwert = '{file_zeitschrittweite_mittelwert}'")
            print(f"file_{i+1}_zeitschrittweite_min = '{file_zeitschrittweite_min}'")
            print(f"file_{i+1}_skalierung = '{file_skalierung}'")
            print(f"file_{i+1}_skalierung_max = '{file_skalierung_max}'")
            print(f"file_{i+1}_skalierung_min = '{file_skalierung_min}'")
            print(f"file_{i+1}_storage_path = '{file_storage_path}'")

            # 4. Download file content (first 100 chars for brevity)
            try:
                file_content = download_file_content(session_uuid, file_type, file_name)
                print(f"file_{i+1}_content_preview = '{file_content[:100].decode(errors='ignore')}...'\n")
            except Exception as e:
                print(f"  -> Failed to download content for {file_name}: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with backend: {e}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response from backend: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")