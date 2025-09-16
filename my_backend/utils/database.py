import os
import json
import logging
import uuid
import base64
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
def get_supabase_client() -> Client:
    """
    Get the Supabase client instance.
    
    Returns:
        Client: Supabase client instance
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase URL or key not found in environment variables. Database features will be disabled.")
        return None
        
    try:
        # Create client with only the supported parameters
        # Note: supabase==2.3.3 doesn't support the proxy parameter
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Error creating Supabase client: {str(e)}")
        return None

def create_or_get_session_uuid(session_id: str) -> str:
    """
    Create or get UUID for a session from the session_mappings table.

    Args:
        session_id: The string-based session ID from the frontend.

    Returns:
        str: The UUID of the session.
    """
    if not session_id:
        logger.error("session_id cannot be empty")
        return None
    
    # Check if session_id is already a UUID
    import re
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    if uuid_pattern.match(session_id):
        logger.info(f"Session ID {session_id} is already a UUID, returning as-is")
        return session_id

    supabase = get_supabase_client()
    if not supabase:
        return None

    # 1. Check if a mapping already exists
    try:
        response = supabase.table('session_mappings').select('uuid_session_id').eq('string_session_id', session_id).single().execute()
        if response.data:
            uuid_session_id = response.data['uuid_session_id']
            logger.info(f"Found existing mapping for {session_id}: {uuid_session_id}")
            return uuid_session_id
    except Exception as e:
        logger.info(f"No existing mapping found for {session_id}, will create a new one. Error: {e}")


    # 2. If no mapping exists, create a new session and a new mapping
    try:
        # Create a new session in the 'sessions' table
        session_response = supabase.table('sessions').insert({}).execute()
        
        if getattr(session_response, 'error', None):
            raise Exception(f"Error creating new session: {session_response.error}")
        
        if not session_response.data:
            raise Exception("Insert operation into 'sessions' did not return data.")

        new_uuid_session_id = session_response.data[0]['id']

        # Create a new mapping in the 'session_mappings' table
        mapping_response = supabase.table('session_mappings').insert({
            'string_session_id': session_id,
            'uuid_session_id': new_uuid_session_id
        }).execute()

        if getattr(mapping_response, 'error', None):
            raise Exception(f"Error creating session mapping: {mapping_response.error}")

        logger.info(f"Created new session and mapping for {session_id}: {new_uuid_session_id}")
        return new_uuid_session_id

    except Exception as e:
        logger.error(f"Failed to create session and mapping for {session_id}: {e}")
        return None

def get_string_id_from_uuid(uuid_session_id: str) -> str:
    """
    Get the string session ID from a UUID.

    Args:
        uuid_session_id: The UUID of the session.

    Returns:
        str: The string-based session ID.
    """
    if not uuid_session_id:
        return None

    supabase = get_supabase_client()
    if not supabase:
        return None

    try:
        response = supabase.table('session_mappings').select('string_session_id').eq('uuid_session_id', uuid_session_id).single().execute()
        if response.data:
            return response.data['string_session_id']
        return None
    except Exception as e:
        logger.error(f"Could not retrieve string_session_id for UUID {uuid_session_id}: {e}")
        return None

def save_time_info(session_id: str, time_info: dict) -> bool:
    """
    Save time information to the time_info table.
    
    Args:
        session_id: ID of the session (can be string or UUID)
        time_info: Dictionary containing time information
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
        
        # Convert session_id to UUID format if it's not already
        logger.info(f"save_time_info called with session_id: {session_id}")
        try:
            # Check if it's already a valid UUID
            uuid.UUID(session_id)
            database_session_id = session_id
            logger.info(f"Session_id {session_id} is already a valid UUID")
        except (ValueError, TypeError):
            # If not UUID, try to get or create UUID session
            logger.info(f"Converting string session_id '{session_id}' to UUID format")
            database_session_id = create_or_get_session_uuid(session_id)
            if not database_session_id:
                logger.error(f"Failed to convert session_id {session_id} to UUID")
                return False
            logger.info(f"Successfully converted '{session_id}' to UUID: '{database_session_id}'")
        
        logger.info(f"Processing time_info for session {database_session_id}")
        logger.info(f"time_info keys: {list(time_info.keys())}")
            
        # Prepare data for insertion - simplified structure
        data = {
            "session_id": database_session_id,
            # Boolean flags for active categories
            "jahr": time_info.get("jahr", False),
            "woche": time_info.get("woche", False),
            "monat": time_info.get("monat", False),
            "feiertag": time_info.get("feiertag", False),
            "tag": time_info.get("tag", False),
            
            # Global timezone setting
            "zeitzone": time_info.get("zeitzone", "UTC"),
            
            # JSONB structure for category-specific data
            "category_data": time_info.get("category_data", {})
        }
        
        # Ensure proper UTF-8 encoding for category_data
        if isinstance(data['category_data'], dict):
            # Convert any Unicode strings to proper UTF-8
            import json
            category_data_str = json.dumps(data['category_data'], ensure_ascii=False)
            data['category_data'] = json.loads(category_data_str)
        
        # Log data being sent to database
        logger.info(f"Sending time_info data to database for session {database_session_id}:")
        logger.info(f"Boolean flags: jahr={data['jahr']}, monat={data['monat']}, woche={data['woche']}, tag={data['tag']}, feiertag={data['feiertag']}")
        logger.info(f"Zeitzone: {data['zeitzone']}")
        logger.info(f"Category data keys: {list(data['category_data'].keys())}")
        
        # Prvo proverimo da li već postoji zapis za ovu sesiju
        logger.info(f"Checking for existing time_info record for session {database_session_id}")
        try:
            existing = supabase.table("time_info").select("*").eq("session_id", database_session_id).execute()
            logger.info(f"Existing record check successful: found {len(existing.data) if existing.data else 0} records")
        except Exception as e:
            logger.error(f"Error checking existing record: {str(e)}")
            return False
        
        if existing.data and len(existing.data) > 0:
            # Ako postoji, ažuriramo postojeći zapis
            logger.info(f"Found existing time_info record for session {database_session_id}, updating...")
            try:
                response = supabase.table("time_info").update(data).eq("session_id", database_session_id).execute()
                logger.info(f"Update response: {response}")
                logger.info(f"Updated existing time_info for session {database_session_id}")
            except Exception as e:
                logger.error(f"Error updating time_info: {str(e)}")
                return False
        else:
            # Ako ne postoji, dodajemo novi zapis
            logger.info(f"No existing time_info record found for session {database_session_id}, inserting new record...")
            try:
                response = supabase.table("time_info").insert(data).execute()
                logger.info(f"Insert response: {response}")
                logger.info(f"Inserted new time_info for session {database_session_id}")
            except Exception as e:
                logger.error(f"Error inserting time_info: {str(e)}")
                return False
        
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error saving time_info: {response.error}")
            return False
            
        logger.info(f"Successfully saved time_info for session {database_session_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving time_info: {str(e)}")
        return False

def save_zeitschritte(session_id: str, zeitschritte: dict) -> bool:
    """
    Save zeitschritte information to the zeitschritte table.
    
    Args:
        session_id: ID of the session (can be string or UUID)
        zeitschritte: Dictionary containing zeitschritte information
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
        
        # Convert session_id to UUID format if it's not already
        try:
            # Check if it's already a valid UUID
            uuid.UUID(session_id)
            database_session_id = session_id
        except (ValueError, TypeError):
            # If not UUID, try to get or create UUID session
            logger.info(f"Converting string session_id {session_id} to UUID format for zeitschritte")
            database_session_id = create_or_get_session_uuid(session_id)
            if not database_session_id:
                logger.error(f"Failed to convert session_id {session_id} to UUID for zeitschritte")
                return False
            logger.info(f"Using UUID session_id for zeitschritte: {database_session_id}")
            
        # Prepare data for insertion
        # Handle both 'offset' and 'offsett' from frontend (frontend sends 'offsett')
        offset_value = zeitschritte.get("offsett", zeitschritte.get("offset", ""))
        data = {
            "session_id": database_session_id,
            "eingabe": zeitschritte.get("eingabe", ""),
            "ausgabe": zeitschritte.get("ausgabe", ""),
            "zeitschrittweite": zeitschritte.get("zeitschrittweite", ""),
            "offset": offset_value  # Database column is 'offset' with single 't'
        }
        logger.info(f"Preparing zeitschritte data with offset value: '{offset_value}'")
        
        # Prvo proverimo da li već postoji zapis za ovu sesiju
        logger.info(f"Checking for existing zeitschritte record for session {database_session_id}")
        try:
            existing = supabase.table("zeitschritte").select("*").eq("session_id", database_session_id).execute()
            logger.info(f"Existing zeitschritte check successful: found {len(existing.data) if existing.data else 0} records")
        except Exception as e:
            logger.error(f"Error checking existing zeitschritte: {str(e)}")
            return False
        
        if existing.data and len(existing.data) > 0:
            # Ako postoji, ažuriramo postojeći zapis
            logger.info(f"Found existing zeitschritte record, updating...")
            try:
                response = supabase.table("zeitschritte").update(data).eq("session_id", database_session_id).execute()
                logger.info(f"Zeitschritte update response: {response}")
                logger.info(f"Updated existing zeitschritte for session {database_session_id}")
            except Exception as e:
                logger.error(f"Error updating zeitschritte: {str(e)}")
                return False
        else:
            # Ako ne postoji, dodajemo novi zapis
            logger.info(f"No existing zeitschritte record found, inserting new...")
            try:
                response = supabase.table("zeitschritte").insert(data).execute()
                logger.info(f"Zeitschritte insert response: {response}")
                logger.info(f"Inserted new zeitschritte for session {database_session_id}")
            except Exception as e:
                logger.error(f"Error inserting zeitschritte: {str(e)}")
                return False
        
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error saving zeitschritte: {response.error}")
            return False
            
        logger.info(f"Successfully saved zeitschritte for session {database_session_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving zeitschritte: {str(e)}")
        return False

def save_file_info(session_id: str, file_info: dict) -> tuple:
    """
    Save file information to the files table.
    
    Args:
        session_id: ID of the session (can be string or UUID)
        file_info: Dictionary containing file information
        
    Returns:
        tuple: (success, valid_uuid) where success is True if successful, False otherwise,
               and valid_uuid is the UUID used for the file (either validated or newly generated)
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False, None
        
        # Convert session_id to UUID format if it's not already
        try:
            # Check if it's already a valid UUID
            uuid.UUID(session_id)
            database_session_id = session_id
        except (ValueError, TypeError):
            # If not UUID, try to get or create UUID session
            logger.info(f"Converting string session_id {session_id} to UUID format for file info")
            database_session_id = create_or_get_session_uuid(session_id)
            if not database_session_id:
                logger.error(f"Failed to convert session_id {session_id} to UUID for file info")
                return False, None
            logger.info(f"Using UUID session_id for file info: {database_session_id}")
            
        # Proveri da li je ID u UUID formatu, ako nije generiši novi UUID
        file_id = file_info.get("id")
        try:
            # Pokušaj konverziju u UUID format
            uuid_obj = uuid.UUID(file_id)
            valid_uuid = str(uuid_obj)
        except (ValueError, TypeError, AttributeError):
            # Ako konverzija ne uspe, generiši novi UUID
            valid_uuid = str(uuid.uuid4())
            logger.info(f"Generated new UUID {valid_uuid} for file {file_info.get('fileName')}")
            
        # Prepare data for insertion/update - strogo prema shemi tabele files
        # Prepare data according to files table schema
        
        # Generate storage path if not provided in file_info
        storage_path = file_info.get("storagePath", "")
        if not storage_path:
            file_name = file_info.get("fileName", "")
            if file_name:
                storage_path = f"{database_session_id}/{file_name}"
                logger.info(f"Generated storage_path: {storage_path} for file {file_name}")
            else:
                logger.warning("No fileName provided, storage_path will be empty")
        
        data = {
            "id": valid_uuid,
            "session_id": database_session_id,
            "file_name": file_info.get("fileName", ""),
            "bezeichnung": file_info.get("bezeichnung", ""),
            "min": str(file_info.get("min", "")),
            "max": str(file_info.get("max", "")),
            "offset": str(file_info.get("offset", "")),
            "datenpunkte": str(file_info.get("datenpunkte", "")),
            "numerische_datenpunkte": str(file_info.get("numerischeDatenpunkte", "")),
            "numerischer_anteil": str(file_info.get("numerischerAnteil", "")),
            "datenform": file_info.get("datenform", ""),
            "datenanpassung": file_info.get("datenanpassung", ""),
            "zeitschrittweite": str(file_info.get("zeitschrittweite", "")),
            "zeitschrittweite_mittelwert": str(file_info.get("zeitschrittweiteAvgValue", file_info.get("zeitschrittweiteMittelwert", ""))),
            "zeitschrittweite_min": str(file_info.get("zeitschrittweiteMinValue", file_info.get("zeitschrittweiteMin", ""))),
            "skalierung": file_info.get("skalierung", "nein"),
            "skalierung_max": str(file_info.get("skalierungMax", "")),
            "skalierung_min": str(file_info.get("skalierungMin", "")),
            "zeithorizont_start": file_info.get("zeithorizontStart", ""),
            "zeithorizont_end": file_info.get("zeithorizontEnd", ""),
            "zeitschrittweite_transferierten_daten": str(file_info.get("zeitschrittweiteTransferiertenDaten", "")),
            "offset_transferierten_daten": str(file_info.get("offsetTransferiertenDaten", "")),
            "mittelwertbildung_uber_den_zeithorizont": file_info.get("mittelwertbildungÜberDenZeithorizont", "nein"),
            "storage_path": storage_path,
            "type": file_info.get("type", "")
        }
        
        # Posebno rukovanje timestamp poljima
        utc_min = file_info.get("utcMin")
        utc_max = file_info.get("utcMax")
        if utc_min:
            try:
                # Parsiramo datetime objekt
                dt_obj = datetime.fromisoformat(utc_min)
                # Pretvaramo ga u string format koji PostgreSQL razumije
                data["utc_min"] = dt_obj.isoformat(sep=' ', timespec='seconds')
                logger.info(f"Successfully parsed utc_min: {data['utc_min']}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid utcMin format: {utc_min}, error: {str(e)}")
        else:
            logger.info("No utc_min provided")
            
        if utc_max:
            try:
                # Parsiramo datetime objekt
                dt_obj = datetime.fromisoformat(utc_max)
                # Pretvaramo ga u string format koji PostgreSQL razumije
                data["utc_max"] = dt_obj.isoformat(sep=' ', timespec='seconds')
                logger.info(f"Successfully parsed utc_max: {data['utc_max']}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid utcMax format: {utc_max}, error: {str(e)}")
        else:
            logger.info("No utc_max provided")
            
        # Log data being sent to Supabase
        logger.info(f"Attempting to save file data with ID {valid_uuid} to files table")
        logger.info(f"Data being sent: {json.dumps(data, default=str)}")
        
        # Posebno logiranje za zeitschrittweite vrijednosti
        logger.info(f"zeitschrittweite_mittelwert value: {data['zeitschrittweite_mittelwert']}")
        logger.info(f"zeitschrittweite_min value: {data['zeitschrittweite_min']}")
        logger.info(f"Original values from frontend - zeitschrittweiteAvgValue: {file_info.get('zeitschrittweiteAvgValue', '')}, zeitschrittweiteMinValue: {file_info.get('zeitschrittweiteMinValue', '')}")
        
        
        # Provjeri da li postoji kolona 'zeithorizont' u podacima
        if 'zeithorizont' in data:
            logger.warning(f"Found 'zeithorizont' key in data which might cause issues")
        
        # Ispiši sve ključeve u podacima
        logger.info(f"All keys in data: {list(data.keys())}")
        
        # Provjeri da li postoji kolona 'zeithorizont_start' i 'zeithorizont_end'
        if 'zeithorizont_start' in data and 'zeithorizont_end' in data:
            logger.info(f"Found both 'zeithorizont_start' and 'zeithorizont_end' keys in data")
        
        # Insert data into files table
        response = supabase.table("files").insert(data).execute()
        
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error saving file info: {response.error}")
            return False
            
        logger.info(f"Successfully saved file info for file {file_info.get('fileName')} in session {database_session_id}")
        return True, valid_uuid
        
    except Exception as e:
        logger.error(f"Error saving file info: {str(e)}")
        return False, None

def save_csv_file_content(file_id: str, session_id: str, file_name: str, file_path: str, file_type: str) -> bool:
    """
    Save CSV file content to Supabase Storage.
    
    Args:
        file_id: ID of the file (from files table)
        session_id: ID of the session
        file_name: Name of the file
        file_path: Path to the file
        file_type: Type of the file ('input' or 'output')
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            logger.error("Supabase client not available")
            return False

        # Determine the bucket name based on the file type
        bucket_name = 'aus-csv-files' if file_type == 'output' else 'csv-files'
        logger.info(f"Uploading {file_name} to bucket: {bucket_name}")
            
        # Read file content
        with open(file_path, 'rb') as f:
            file_content = f.read()
            
        # Get file size
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size} bytes")
        
        # Definisanje putanje u Storage-u
        storage_path = f"{session_id}/{file_name}"
        
        try:
            # Check if file already exists in storage
            try:
                existing_files = supabase.storage.from_(bucket_name).list(session_id)
                file_exists = any(f['name'] == file_name for f in existing_files)
                
                if file_exists:
                    logger.info(f"File {file_name} already exists in storage, updating...")
                    # Update existing file
                    storage_response = supabase.storage.from_(bucket_name).update(
                        path=storage_path,
                        file=file_content,
                        file_options={"content-type": "text/csv"}
                    )
                else:
                    # Upload new file
                    storage_response = supabase.storage.from_(bucket_name).upload(
                        path=storage_path,
                        file=file_content,
                        file_options={"content-type": "text/csv"}
                    )
            except Exception as list_error:
                logger.warning(f"Could not check if file exists, attempting upload: {str(list_error)}")
                # Try to upload anyway
                storage_response = supabase.storage.from_(bucket_name).upload(
                    path=storage_path,
                    file=file_content,
                    file_options={"content-type": "text/csv"}
                )
            
            logger.info(f"Successfully uploaded {file_name} to {bucket_name}/{storage_path}")
            
            # Update the storage_path in files table if file_id is valid
            try:
                uuid_obj = uuid.UUID(file_id)
                valid_file_id = str(uuid_obj)
                
                # Update files table with storage path
                update_response = supabase.table("files").update({
                    "storage_path": storage_path
                }).eq("id", valid_file_id).execute()
                
                if update_response.data:
                    logger.info(f"Updated storage_path in files table for file_id {valid_file_id}")
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Could not update files table - invalid file_id: {file_id}, error: {str(e)}")
                # Continue anyway - file is uploaded to storage
            
            return True
            
        except Exception as storage_error:
            logger.error(f"Error uploading to storage: {str(storage_error)}")
            # If error is "already exists", treat as success
            if "already exists" in str(storage_error).lower():
                logger.info(f"File already exists in storage, considering as success")
                return True
            return False
        
    except Exception as e:
        logger.error(f"Error saving CSV file content: {str(e)}")
        return False

    """
    Transform old time information format to the new JSONB structure format.
    
    Args:
        time_info: Dictionary containing time information in the old format
        
    Returns:
        dict: Time information in the new format with category_data JSONB structure
    """
    logger.info("Starting transformation of time_info to new JSONB format")
    logger.info(f"Input time_info keys: {list(time_info.keys())}")
    
    # Create a new dictionary with the base structure
    new_time_info = {
        "jahr": time_info.get("jahr", False),
        "monat": time_info.get("monat", False),
        "woche": time_info.get("woche", False),
        "feiertag": time_info.get("feiertag", False),
        "tag": time_info.get("tag", False),  # New category, default to False
        "zeitzone": time_info.get("zeitzone", "")
    }
    
    logger.info(f"Base structure created with flags: jahr={new_time_info['jahr']}, monat={new_time_info['monat']}, "
               f"woche={new_time_info['woche']}, feiertag={new_time_info['feiertag']}, tag={new_time_info['tag']}")
    
    # Check if the time_info already has the new structure
    if "category_data" in time_info:
        logger.info("Input data already has category_data structure, using it directly")
        new_time_info["category_data"] = time_info["category_data"]
        logger.info(f"Categories in existing category_data: {list(time_info['category_data'].keys())}")
        return new_time_info
    
    logger.info("Input data uses old format, transforming to new JSONB structure")
    
    # Build category_data for each active category using the old format
    category_data = {}
    
    # Common fields from the old format
    detaillierte_berechnung = time_info.get("detaillierteBerechnung", False)
    datenform = time_info.get("datenform", "")
    zeithorizont_start = time_info.get("zeithorizontStart", "")
    zeithorizont_end = time_info.get("zeithorizontEnd", "")
    skalierung = time_info.get("skalierung", "nein")
    skalierung_min = time_info.get("skalierungMin", "")
    skalierung_max = time_info.get("skalierungMax", "")
    
    logger.info(f"Common fields from old format: detaillierteBerechnung={detaillierte_berechnung}, "
               f"datenform='{datenform}', zeithorizont_start='{zeithorizont_start}', "
               f"zeithorizont_end='{zeithorizont_end}', skalierung='{skalierung}'")
    
    
    # For each active category, create a category-specific entry
    if new_time_info["jahr"]:
        category_data["jahr"] = {
            "detaillierteBerechnung": detaillierte_berechnung,
            "datenform": datenform,
            "zeithorizontStart": zeithorizont_start,
            "zeithorizontEnd": zeithorizont_end,
            "skalierung": skalierung,
            "skalierungMin": skalierung_min,
            "skalierungMax": skalierung_max
        }
    
    if new_time_info["monat"]:
        category_data["monat"] = {
            "detaillierteBerechnung": detaillierte_berechnung,
            "datenform": datenform,
            "zeithorizontStart": zeithorizont_start,
            "zeithorizontEnd": zeithorizont_end,
            "skalierung": skalierung,
            "skalierungMin": skalierung_min,
            "skalierungMax": skalierung_max
        }
    
    if new_time_info["woche"]:
        category_data["woche"] = {
            "detaillierteBerechnung": detaillierte_berechnung,
            "datenform": datenform,
            "zeithorizontStart": zeithorizont_start,
            "zeithorizontEnd": zeithorizont_end,
            "skalierung": skalierung,
            "skalierungMin": skalierung_min,
            "skalierungMax": skalierung_max
        }
    
    # Tag is a new category, so it won't have data in the old format
    
    if new_time_info["feiertag"]:
        category_data["feiertag"] = {
            "detaillierteBerechnung": detaillierte_berechnung,
            "datenform": datenform,
            "zeithorizontStart": zeithorizont_start,
            "zeithorizontEnd": zeithorizont_end,
            "skalierung": skalierung,
            "skalierungMin": skalierung_min,
            "skalierungMax": skalierung_max,
            "land": time_info.get("land", "Deutschland")  # Special field for Feiertag
        }
    
    # Add the category_data to the new time_info
    new_time_info["category_data"] = category_data
    
    return new_time_info

def save_session_to_supabase(session_id: str, n_dat: int = None, file_count: int = None) -> bool:
    """
    Save all session data to Supabase.

    Args:
        session_id: ID of the session (string format)
        n_dat: Total number of data samples (optional)
        file_count: Number of files in the session (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get or create the UUID for this session
        logger.info(f"save_session_to_supabase called with session_id: {session_id}")
        try:
            uuid.UUID(session_id)
            database_session_id = session_id
            logger.info(f"Session_id {session_id} is already a valid UUID")
        except (ValueError, TypeError):
            logger.info(f"Converting string session_id '{session_id}' to UUID format in save_session_to_supabase")
            database_session_id = create_or_get_session_uuid(session_id)
            if not database_session_id:
                logger.error(f"Failed to get UUID for session {session_id}")
                return False
            logger.info(f"Successfully converted '{session_id}' to UUID: '{database_session_id}' in save_session_to_supabase")
        # Base directory for file uploads - use the api/routes directory where files are actually saved
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to my_backend
        upload_base_dir = os.path.join(base_path, 'api', 'routes', 'uploads', 'file_uploads')
        session_dir = os.path.join(upload_base_dir, session_id)
        
        # Check if session directory exists
        if not os.path.exists(session_dir):
            logger.error(f"Session directory not found: {session_dir}")
            return False
            
        # Path to session metadata file
        metadata_path = os.path.join(session_dir, 'session_metadata.json')
        
        # Check if session metadata file exists
        if not os.path.exists(metadata_path):
            logger.error(f"Session metadata file not found: {metadata_path}")
            return False
            
        # Load session metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Save time info
        if 'timeInfo' in metadata:
            # Log the original structure for debugging
            logger.info(f"Original timeInfo structure: {json.dumps(metadata['timeInfo'], indent=2)}")
            
            # Save time info directly using the UUID session - pass the UUID, not the original string
            success = save_time_info(database_session_id, metadata['timeInfo'])
            if not success:
                logger.error(f"Failed to save time_info for session {database_session_id}")
            
        # Save zeitschritte
        if 'zeitschritte' in metadata:
            success = save_zeitschritte(database_session_id, metadata['zeitschritte'])
            if not success:
                logger.error(f"Failed to save zeitschritte for session {database_session_id}")
            
        # Save file info and content
        if 'files' in metadata and isinstance(metadata['files'], list):
            for file_info in metadata['files']:
                # Save file info using the UUID session
                success, valid_uuid = save_file_info(database_session_id, file_info)
                
                # Save file content only if file info was saved successfully
                if success and valid_uuid:
                    file_name = file_info.get('fileName', '')
                    file_type = file_info.get('type', 'input')  # Default to 'input' if not specified
                    file_path = os.path.join(session_dir, file_name)
                    
                    if os.path.exists(file_path):
                        logger.info(f"Uploading file {file_name} to Supabase Storage...")
                        # Koristi valid_uuid umesto originalnog file_id
                        upload_success = save_csv_file_content(valid_uuid, database_session_id, file_name, file_path, file_type)
                        if upload_success:
                            logger.info(f"✅ Successfully uploaded {file_name} to storage")
                        else:
                            logger.error(f"❌ Failed to upload {file_name} to storage")
                    else:
                        logger.warning(f"CSV file not found locally: {file_path}")
                else:
                    logger.warning(f"Failed to save file info for {file_info.get('fileName', 'unknown')}, skipping content upload")
        
        # Update sessions table with finalization data
        if n_dat is not None or file_count is not None:
            session_update_data = {
                "finalized": True,
                "updated_at": datetime.now().isoformat()
            }
            if n_dat is not None:
                session_update_data["n_dat"] = n_dat
            if file_count is not None:
                session_update_data["file_count"] = file_count

            try:
                supabase = get_supabase_client()
                if supabase:
                    session_response = supabase.table("sessions").update(session_update_data).eq("id", database_session_id).execute()
                    if hasattr(session_response, 'error') and session_response.error:
                        logger.error(f"Error updating sessions table: {session_response.error}")
                    else:
                        logger.info(f"Successfully updated sessions table with n_dat={n_dat}, file_count={file_count}")
            except Exception as e:
                logger.error(f"Error updating sessions table: {str(e)}")

        logger.info(f"Successfully saved session {session_id} to Supabase")
        return True
        
    except Exception as e:
        logger.error(f"Error saving session to Supabase: {str(e)}")
        return False
