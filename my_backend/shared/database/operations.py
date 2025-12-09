
"""Database operations module for Supabase.

This module provides database operations using the centralized Supabase client.
"""

import os
import json
import logging
import re
import time
import uuid
from datetime import datetime
from supabase import Client
from shared.database.client import get_supabase_client as get_shared_client, get_supabase_admin_client

class DatabaseConfig:
    """Configuration constants for database operations"""
    DEFAULT_RETRY_ATTEMPTS = 3
    DEFAULT_INITIAL_DELAY = 1.0
    DEFAULT_TIMEOUT_CONNECT = 30.0
    DEFAULT_TIMEOUT_READ = 60.0
    DEFAULT_TIMEOUT_WRITE = 30.0
    DEFAULT_TIMEOUT_POOL = 30.0

    DEFAULT_SKALIERUNG = "nein"
    DEFAULT_MITTELWERTBILDUNG = "nein"
    DEFAULT_ZEITZONE = "UTC"
    DEFAULT_FILE_TYPE = "input"
    DEFAULT_LAND = "Deutschland"

    UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    SESSION_ID_PATTERN = r'^session_\d+_[a-zA-Z0-9]+$'

def validate_session_id(session_id: str) -> bool:
    """Validate session ID format"""
    if not session_id or not isinstance(session_id, str):
        return False
    
    try:
        uuid.UUID(session_id)
        return True
    except ValueError:
        pass
    
    return bool(re.match(DatabaseConfig.SESSION_ID_PATTERN, session_id))

def validate_file_info(file_info: dict) -> bool:
    """Validate file info dictionary structure"""
    if not isinstance(file_info, dict):
        return False
    
    required_fields = ['fileName']
    return all(field in file_info for field in required_fields)

def validate_time_info(time_info: dict) -> bool:
    """Validate time info dictionary structure"""
    if not isinstance(time_info, dict):
        return False
    
    boolean_fields = ['jahr', 'monat', 'woche', 'feiertag', 'tag']
    for field in boolean_fields:
        if field in time_info and not isinstance(time_info[field], bool):
            return False
    
    return True

import httpx
httpx._config.DEFAULT_TIMEOUT_CONFIG = httpx.Timeout(
    connect=DatabaseConfig.DEFAULT_TIMEOUT_CONNECT,
    read=DatabaseConfig.DEFAULT_TIMEOUT_READ,
    write=DatabaseConfig.DEFAULT_TIMEOUT_WRITE,
    pool=DatabaseConfig.DEFAULT_TIMEOUT_POOL
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables are now managed by shared.database.client

class DatabaseError(Exception):
    """Base exception for database operations"""
    pass

class SessionNotFoundError(DatabaseError):
    """Raised when session cannot be found or created"""
    pass

class ValidationError(DatabaseError):
    """Raised when input validation fails"""
    pass

class StorageError(DatabaseError):
    """Raised when file storage operations fail"""
    pass

class ConfigurationError(DatabaseError):
    """Raised when configuration is invalid"""
    pass

def get_supabase_client(use_service_role: bool = False) -> Client:
    """
    Get the Supabase client instance.
    
    This function wraps the centralized client from shared.database.client.

    Args:
        use_service_role: If True, always use service_role key (bypasses RLS).
                         If False, uses anon key for RLS enforcement.

    Returns:
        Client: Supabase client instance
    """
    if use_service_role:
        return get_supabase_admin_client()
    return get_shared_client()

def retry_database_operation(operation_func, max_retries=3, initial_delay=1.0):
    """
    Retry database operations with exponential backoff for DNS timeout issues.
    
    Args:
        operation_func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        
    Returns:
        Result of operation_func or None if all retries failed
    """
    for attempt in range(max_retries + 1):
        try:
            return operation_func()
        except Exception as e:
            error_msg = str(e)
            if "Lookup timed out" in error_msg or "timeout" in error_msg.lower():
                if attempt < max_retries:
                    delay = initial_delay * (2 ** attempt)
                    logger.warning(f"Database operation failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s: {error_msg}")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Database operation failed after {max_retries + 1} attempts: {error_msg}")
                    return None
            else:
                logger.error(f"Database operation failed with non-timeout error: {error_msg}")
                return None
    
    return None

def create_or_get_session_uuid(session_id: str, user_id: str = None) -> str:
    """
    Create or get UUID for a session from the session_mappings table.

    SECURITY: Validates session ownership when user_id is provided.

    Args:
        session_id: The string-based session ID from the frontend.
        user_id: The user ID to associate with the session (REQUIRED for creating new sessions).
                For existing sessions, validates that session belongs to this user.

    Returns:
        str: The UUID of the session.

    Raises:
        ValidationError: If session_id is invalid
        SessionNotFoundError: If session cannot be created
        ValueError: If user_id is not provided when creating a new session
        PermissionError: If session exists but doesn't belong to the user
    """
    if not validate_session_id(session_id):
        raise ValidationError(f"Invalid session_id format: {session_id}")
    
    if DatabaseConfig.UUID_PATTERN.match(session_id):
        logger.info(f"Session ID {session_id} is already a UUID, returning as-is")
        return session_id

    # Use service_role to bypass RLS for session creation/lookup
    supabase = get_supabase_client(use_service_role=True)
    if not supabase:
        raise ConfigurationError("Supabase client not available")

    def check_existing_mapping():
        # Join with sessions table to get user_id for ownership validation
        response = supabase.table('session_mappings')\
            .select('uuid_session_id, sessions!inner(user_id)')\
            .eq('string_session_id', session_id)\
            .execute()

        if response.data and len(response.data) > 0:
            uuid_session_id = response.data[0]['uuid_session_id']
            session_owner = response.data[0]['sessions']['user_id']

            # SECURITY: Validate ownership if user_id is provided
            if user_id and session_owner != user_id:
                logger.warning(
                    f"🚨 SECURITY: User {user_id} attempted to access session {session_id} "
                    f"owned by {session_owner}"
                )
                raise PermissionError(f'Session {session_id} does not belong to user')

            logger.info(f"Found existing mapping for {session_id}: {uuid_session_id} (owner: {session_owner})")
            return uuid_session_id
        return None
    
    existing_uuid = retry_database_operation(
        check_existing_mapping, 
        max_retries=DatabaseConfig.DEFAULT_RETRY_ATTEMPTS, 
        initial_delay=DatabaseConfig.DEFAULT_INITIAL_DELAY
    )
    if existing_uuid:
        return existing_uuid
    
    logger.info(f"No existing mapping found for {session_id}, will create a new one.")

    # Validate user_id is provided for new sessions
    if not user_id:
        raise ValueError("user_id is required when creating a new session")

    def create_new_session_mapping():
        # Include user_id when creating session
        session_data = {
            'user_id': user_id
        }
        session_response = supabase.table('sessions').insert(session_data).execute()
        
        if getattr(session_response, 'error', None):
            raise DatabaseError(f"Error creating new session: {session_response.error}")
        
        if not session_response.data:
            raise DatabaseError("Insert operation into 'sessions' did not return data.")

        new_uuid_session_id = session_response.data[0]['id']

        mapping_response = supabase.table('session_mappings').insert({
            'string_session_id': session_id,
            'uuid_session_id': new_uuid_session_id
        }).execute()

        if getattr(mapping_response, 'error', None):
            raise DatabaseError(f"Error creating session mapping: {mapping_response.error}")

        logger.info(f"Created new session for user {user_id} with mapping {session_id}: {new_uuid_session_id}")
        return new_uuid_session_id
    
    new_uuid = retry_database_operation(
        create_new_session_mapping, 
        max_retries=DatabaseConfig.DEFAULT_RETRY_ATTEMPTS, 
        initial_delay=DatabaseConfig.DEFAULT_INITIAL_DELAY
    )
    if new_uuid:
        return new_uuid
    
    raise SessionNotFoundError(f"Failed to create session and mapping for {session_id} after retries")

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
        response = supabase.table('session_mappings').select('string_session_id').eq('uuid_session_id', uuid_session_id).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]['string_session_id']
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
        
    Raises:
        ValidationError: If input parameters are invalid
        ConfigurationError: If Supabase client is not available
        DatabaseError: If database operations fail
    """
    if not validate_session_id(session_id):
        raise ValidationError(f"Invalid session_id format: {session_id}")
    
    if not validate_time_info(time_info):
        raise ValidationError(f"Invalid time_info structure: {time_info}")
    
    supabase = get_supabase_client()
    if not supabase:
        raise ConfigurationError("Supabase client not available")
    
    logger.info(f"save_time_info called with session_id: {session_id}")
    database_session_id = _get_session_uuid(session_id)
    if not database_session_id:
        raise SessionNotFoundError(f"Failed to convert session_id {session_id} to UUID")
    
    logger.info(f"Processing time_info for session {database_session_id}")
    logger.info(f"time_info keys: {list(time_info.keys())}")
        
    data = {
        "session_id": database_session_id,
        "jahr": time_info.get("jahr", False),
        "woche": time_info.get("woche", False),
        "monat": time_info.get("monat", False),
        "feiertag": time_info.get("feiertag", False),
        "tag": time_info.get("tag", False),
        
        "zeitzone": time_info.get("zeitzone", DatabaseConfig.DEFAULT_ZEITZONE),
        
        "category_data": time_info.get("category_data", {})
    }
    
    if isinstance(data['category_data'], dict):
        category_data_str = json.dumps(data['category_data'], ensure_ascii=False)
        data['category_data'] = json.loads(category_data_str)
    
    logger.info(f"Sending time_info data to database for session {database_session_id}:")
    logger.info(f"Boolean flags: jahr={data['jahr']}, monat={data['monat']}, woche={data['woche']}, tag={data['tag']}, feiertag={data['feiertag']}")
    logger.info(f"Zeitzone: {data['zeitzone']}")
    logger.info(f"Category data keys: {list(data['category_data'].keys())}")
    
    try:
        logger.info(f"Checking for existing time_info record for session {database_session_id}")
        existing = supabase.table("time_info").select("*").eq("session_id", database_session_id).execute()
        logger.info(f"Existing record check successful: found {len(existing.data) if existing.data else 0} records")
        
        if existing.data and len(existing.data) > 0:
            logger.info(f"Found existing time_info record for session {database_session_id}, updating...")
            response = supabase.table("time_info").update(data).eq("session_id", database_session_id).execute()
            logger.info(f"Updated existing time_info for session {database_session_id}")
        else:
            logger.info(f"No existing time_info record found for session {database_session_id}, inserting new record...")
            response = supabase.table("time_info").insert(data).execute()
            logger.info(f"Inserted new time_info for session {database_session_id}")
        
        if hasattr(response, 'error') and response.error:
            raise DatabaseError(f"Error saving time_info: {response.error}")
            
        logger.info(f"Successfully saved time_info for session {database_session_id}")
        return True
        
    except Exception as e:
        if isinstance(e, DatabaseError):
            raise
        raise DatabaseError(f"Error saving time_info: {str(e)}")

def save_zeitschritte(session_id: str, zeitschritte: dict, user_id: str = None) -> bool:
    """
    Save zeitschritte information to the zeitschritte table.

    Args:
        session_id: ID of the session (can be string or UUID)
        zeitschritte: Dictionary containing zeitschritte information
        user_id: User ID (required for creating new sessions)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
        
        try:
            uuid.UUID(session_id)
            database_session_id = session_id
        except (ValueError, TypeError):
            logger.info(f"Converting string session_id {session_id} to UUID format for zeitschritte")
            database_session_id = create_or_get_session_uuid(session_id, user_id=user_id)
            if not database_session_id:
                logger.error(f"Failed to convert session_id {session_id} to UUID for zeitschritte")
                return False
            logger.info(f"Using UUID session_id for zeitschritte: {database_session_id}")
            
        offset_value = zeitschritte.get("offsett", zeitschritte.get("offset", ""))

        def clean_value(value):
            return None if value == "" or value is None else str(value)

        data = {
            "session_id": database_session_id,
            "eingabe": clean_value(zeitschritte.get("eingabe", "")),
            "ausgabe": clean_value(zeitschritte.get("ausgabe", "")),
            "zeitschrittweite": clean_value(zeitschritte.get("zeitschrittweite", "")),
            "offset": clean_value(offset_value)
        }
        logger.info(f"Preparing zeitschritte data with offset value: '{offset_value}'")
        
        logger.info(f"Checking for existing zeitschritte record for session {database_session_id}")
        try:
            existing = supabase.table("zeitschritte").select("*").eq("session_id", database_session_id).execute()
            logger.info(f"Existing zeitschritte check successful: found {len(existing.data) if existing.data else 0} records")
        except Exception as e:
            logger.error(f"Error checking existing zeitschritte: {str(e)}")
            return False
        
        if existing.data and len(existing.data) > 0:
            logger.info(f"Found existing zeitschritte record, updating...")
            try:
                response = supabase.table("zeitschritte").update(data).eq("session_id", database_session_id).execute()
                logger.info(f"Zeitschritte update response: {response}")
                logger.info(f"Updated existing zeitschritte for session {database_session_id}")
            except Exception as e:
                logger.error(f"Error updating zeitschritte: {str(e)}")
                return False
        else:
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
        
        try:
            uuid.UUID(session_id)
            database_session_id = session_id
        except (ValueError, TypeError):
            logger.info(f"Converting string session_id {session_id} to UUID format for file info")
            database_session_id = create_or_get_session_uuid(session_id)
            if not database_session_id:
                logger.error(f"Failed to convert session_id {session_id} to UUID for file info")
                return False, None
            logger.info(f"Using UUID session_id for file info: {database_session_id}")
            
        file_id = file_info.get("id")
        try:
            valid_uuid = str(uuid.UUID(file_id))
        except (ValueError, TypeError, AttributeError):
            valid_uuid = str(uuid.uuid4())
            logger.info(f"Generated new UUID {valid_uuid} for file {file_info.get('fileName')}")
            
        
        storage_path = file_info.get("storagePath", "")
        if not storage_path:
            file_name = file_info.get("fileName", "")
            if file_name:
                storage_path = f"{database_session_id}/{file_name}"
                logger.info(f"Generated storage_path: {storage_path} for file {file_name}")
            else:
                logger.warning("No fileName provided, storage_path will be empty")

        # Get zeitschrittweite values from file_info
        zeitschrittweite_mittelwert = file_info.get("zeitschrittweiteAvgValue", file_info.get("zeitschrittweiteMittelwert")) or None
        zeitschrittweite_min_val = file_info.get("zeitschrittweiteMinValue", file_info.get("zeitschrittweiteMin")) or None

        # For output files without zeitschrittweite values, try to copy from input file in same session
        file_type = file_info.get("type", "")
        if file_type == "output" and (zeitschrittweite_mittelwert is None or zeitschrittweite_min_val is None):
            try:
                input_file = supabase.table("files").select(
                    "zeitschrittweite_mittelwert, zeitschrittweite_min"
                ).eq("session_id", database_session_id).eq("type", "input").limit(1).execute()

                if input_file.data and len(input_file.data) > 0:
                    input_data = input_file.data[0]
                    if zeitschrittweite_mittelwert is None and input_data.get("zeitschrittweite_mittelwert"):
                        zeitschrittweite_mittelwert = input_data["zeitschrittweite_mittelwert"]
                    if zeitschrittweite_min_val is None and input_data.get("zeitschrittweite_min"):
                        zeitschrittweite_min_val = input_data["zeitschrittweite_min"]
            except Exception as copy_err:
                logger.warning(f"Could not copy zeitschrittweite values from input file: {str(copy_err)}")

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
            "zeitschrittweite_mittelwert": zeitschrittweite_mittelwert,
            "zeitschrittweite_min": zeitschrittweite_min_val,
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
        
        utc_min = file_info.get("utcMin")
        utc_max = file_info.get("utcMax")
        if utc_min:
            try:
                dt_obj = datetime.fromisoformat(utc_min)
                data["utc_min"] = dt_obj.isoformat(sep=' ', timespec='seconds')
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid utcMin format: {utc_min}, error: {str(e)}")

        if utc_max:
            try:
                dt_obj = datetime.fromisoformat(utc_max)
                data["utc_max"] = dt_obj.isoformat(sep=' ', timespec='seconds')
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid utcMax format: {utc_max}, error: {str(e)}")

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

        bucket_name = 'aus-csv-files' if file_type == 'output' else 'csv-files'
        logger.info(f"Uploading {file_name} to bucket: {bucket_name}")
            
        with open(file_path, 'rb') as f:
            file_content = f.read()
            
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size} bytes")
        
        storage_path = f"{session_id}/{file_name}"
        
        try:
            try:
                existing_files = supabase.storage.from_(bucket_name).list(session_id)
                file_exists = any(f['name'] == file_name for f in existing_files)
                
                if file_exists:
                    logger.info(f"File {file_name} already exists in storage, updating...")
                    storage_response = supabase.storage.from_(bucket_name).update(
                        path=storage_path,
                        file=file_content,
                        file_options={"content-type": "text/csv"}
                    )
                else:
                    storage_response = supabase.storage.from_(bucket_name).upload(
                        path=storage_path,
                        file=file_content,
                        file_options={"content-type": "text/csv"}
                    )
            except Exception as list_error:
                logger.warning(f"Could not check if file exists, attempting upload: {str(list_error)}")
                storage_response = supabase.storage.from_(bucket_name).upload(
                    path=storage_path,
                    file=file_content,
                    file_options={"content-type": "text/csv"}
                )
            
            logger.info(f"Successfully uploaded {file_name} to {bucket_name}/{storage_path}")
            
            try:
                uuid_obj = uuid.UUID(file_id)
                valid_file_id = str(uuid_obj)
                
                update_response = supabase.table("files").update({
                    "storage_path": storage_path
                }).eq("id", valid_file_id).execute()
                
                if update_response.data:
                    logger.info(f"Updated storage_path in files table for file_id {valid_file_id}")
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Could not update files table - invalid file_id: {file_id}, error: {str(e)}")
            
            return True
            
        except Exception as storage_error:
            logger.error(f"Error uploading to storage: {str(storage_error)}")
            if "already exists" in str(storage_error).lower():
                logger.info(f"File already exists in storage, considering as success")
                return True
            return False
        
    except Exception as e:
        logger.error(f"Error saving CSV file content: {str(e)}")
        return False

def _transform_time_info_to_jsonb(time_info: dict) -> dict:
    """
    Transform old time information format to the new JSONB structure format.

    Args:
        time_info: Dictionary containing time information in the old format

    Returns:
        dict: Time information in the new format with category_data JSONB structure
    """
    logger.info("Starting transformation of time_info to new JSONB format")
    logger.info(f"Input time_info keys: {list(time_info.keys())}")
    
    new_time_info = {
        "jahr": time_info.get("jahr", False),
        "monat": time_info.get("monat", False),
        "woche": time_info.get("woche", False),
        "feiertag": time_info.get("feiertag", False),
        "tag": time_info.get("tag", False),
        "zeitzone": time_info.get("zeitzone", "")
    }
    
    logger.info(f"Base structure created with flags: jahr={new_time_info['jahr']}, monat={new_time_info['monat']}, "
               f"woche={new_time_info['woche']}, feiertag={new_time_info['feiertag']}, tag={new_time_info['tag']}")
    
    if "category_data" in time_info:
        logger.info("Input data already has category_data structure, using it directly")
        new_time_info["category_data"] = time_info["category_data"]
        logger.info(f"Categories in existing category_data: {list(time_info['category_data'].keys())}")
        return new_time_info
    
    logger.info("Input data uses old format, transforming to new JSONB structure")
    
    category_data = {}
    
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
    
    
    if new_time_info["feiertag"]:
        category_data["feiertag"] = {
            "detaillierteBerechnung": detaillierte_berechnung,
            "datenform": datenform,
            "zeithorizontStart": zeithorizont_start,
            "zeithorizontEnd": zeithorizont_end,
            "skalierung": skalierung,
            "skalierungMin": skalierung_min,
            "skalierungMax": skalierung_max,
            "land": time_info.get("land", "Deutschland")
        }
    
    new_time_info["category_data"] = category_data
    
    return new_time_info

def _get_session_uuid(session_id: str) -> str:
    """
    Convert session_id to UUID format if needed.
    
    Args:
        session_id: Original session ID (string or UUID format)
        
    Returns:
        str: UUID format session ID or None if conversion fails
    """
    try:
        uuid.UUID(session_id)
        logger.info(f"Session_id {session_id} is already a valid UUID")
        return session_id
    except (ValueError, TypeError):
        logger.info(f"Converting string session_id '{session_id}' to UUID format")
        database_session_id = create_or_get_session_uuid(session_id)
        if not database_session_id:
            logger.error(f"Failed to get UUID for session {session_id}")
            return None
        logger.info(f"Successfully converted '{session_id}' to UUID: '{database_session_id}'")
        return database_session_id

def _load_session_metadata(session_id: str) -> dict:
    """
    Load session metadata from filesystem.
    
    Args:
        session_id: Session ID for directory structure
        
    Returns:
        dict: Session metadata or None if loading fails
    """
    upload_base_dir = 'uploads/file_uploads'
    session_dir = os.path.join(upload_base_dir, session_id)
    
    if not os.path.exists(session_dir):
        logger.error(f"Session directory not found: {session_dir}")
        return None
        
    metadata_path = os.path.join(session_dir, 'session_metadata.json')
    
    if not os.path.exists(metadata_path):
        logger.error(f"Session metadata file not found: {metadata_path}")
        return None
        
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading session metadata: {str(e)}")
        return None

def _save_metadata_to_database(database_session_id: str, metadata: dict) -> bool:
    """
    Save time info and zeitschritte metadata to database.
    
    Args:
        database_session_id: UUID format session ID
        metadata: Session metadata dictionary
        
    Returns:
        bool: True if successful, False otherwise
    """
    success = True
    
    if 'timeInfo' in metadata:
        logger.info(f"Original timeInfo structure: {json.dumps(metadata['timeInfo'], indent=2)}")
        if not save_time_info(database_session_id, metadata['timeInfo']):
            logger.error(f"Failed to save time_info for session {database_session_id}")
            success = False
        
    if 'zeitschritte' in metadata:
        if not save_zeitschritte(database_session_id, metadata['zeitschritte']):
            logger.error(f"Failed to save zeitschritte for session {database_session_id}")
            success = False
    
    return success

def _prepare_file_batch_data(database_session_id: str, files_list: list) -> list:
    """
    Prepare batch data for file insertion.

    Args:
        database_session_id: UUID format session ID
        files_list: List of file info dictionaries

    Returns:
        list: List of prepared file data for batch insertion
    """
    batch_data = []

    # Pre-extract zeitschrittweite values from input files in the batch
    input_zeitschrittweite_mittelwert = None
    input_zeitschrittweite_min = None
    for file_info in files_list:
        if file_info.get("type") == "input":
            input_zeitschrittweite_mittelwert = file_info.get("zeitschrittweiteAvgValue", file_info.get("zeitschrittweiteMittelwert")) or None
            input_zeitschrittweite_min = file_info.get("zeitschrittweiteMinValue", file_info.get("zeitschrittweiteMin")) or None
            if input_zeitschrittweite_mittelwert or input_zeitschrittweite_min:
                break

    for file_info in files_list:
        if not validate_file_info(file_info):
            logger.warning(f"Skipping invalid file info: {file_info}")
            continue
            
        file_id = file_info.get("id")
        try:
            valid_uuid = str(uuid.UUID(file_id))
        except (ValueError, TypeError, AttributeError):
            valid_uuid = str(uuid.uuid4())
            logger.info(f"Generated new UUID {valid_uuid} for file {file_info.get('fileName')}")
        
        storage_path = file_info.get("storagePath", "")
        if not storage_path:
            file_name = file_info.get("fileName", "")
            if file_name:
                storage_path = f"{database_session_id}/{file_name}"

        # Get zeitschrittweite values, falling back to input file values for output files
        zeitschrittweite_mittelwert = file_info.get("zeitschrittweiteAvgValue", file_info.get("zeitschrittweiteMittelwert")) or None
        zeitschrittweite_min_val = file_info.get("zeitschrittweiteMinValue", file_info.get("zeitschrittweiteMin")) or None

        # For output files without zeitschrittweite values, copy from input file
        file_type = file_info.get("type", "")
        if file_type == "output":
            if zeitschrittweite_mittelwert is None and input_zeitschrittweite_mittelwert:
                zeitschrittweite_mittelwert = input_zeitschrittweite_mittelwert
            if zeitschrittweite_min_val is None and input_zeitschrittweite_min:
                zeitschrittweite_min_val = input_zeitschrittweite_min

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
            "zeitschrittweite_mittelwert": zeitschrittweite_mittelwert,
            "zeitschrittweite_min": zeitschrittweite_min_val,
            "skalierung": file_info.get("skalierung", DatabaseConfig.DEFAULT_SKALIERUNG),
            "skalierung_max": str(file_info.get("skalierungMax", "")),
            "skalierung_min": str(file_info.get("skalierungMin", "")),
            "zeithorizont_start": file_info.get("zeithorizontStart", ""),
            "zeithorizont_end": file_info.get("zeithorizontEnd", ""),
            "zeitschrittweite_transferierten_daten": str(file_info.get("zeitschrittweiteTransferiertenDaten", "")),
            "offset_transferierten_daten": str(file_info.get("offsetTransferiertenDaten", "")),
            "mittelwertbildung_uber_den_zeithorizont": file_info.get("mittelwertbildungÜberDenZeithorizont", DatabaseConfig.DEFAULT_MITTELWERTBILDUNG),
            "storage_path": storage_path,
            "type": file_info.get("type", DatabaseConfig.DEFAULT_FILE_TYPE)
        }
        
        for field_name, value in [("utcMin", file_info.get("utcMin")), ("utcMax", file_info.get("utcMax"))]:
            if value:
                try:
                    dt_obj = datetime.fromisoformat(value)
                    db_field_name = "utc_min" if field_name == "utcMin" else "utc_max"
                    data[db_field_name] = dt_obj.isoformat(sep=' ', timespec='seconds')
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid {field_name} format: {value}, error: {str(e)}")
        
        batch_data.append(data)
    
    return batch_data

def _batch_upsert_files(supabase, database_session_id: str, batch_data: list) -> list:
    """
    Perform smart UPSERT of file data - INSERT new, UPDATE existing, DELETE removed.

    Args:
        supabase: Supabase client
        database_session_id: UUID format session ID
        batch_data: List of file data to upsert

    Returns:
        list: List of successfully upserted file UUIDs
    """
    if not batch_data:
        return []

    try:
        # 1. Fetch existing files for this session
        logger.info(f"Fetching existing files for session {database_session_id}")
        existing_response = supabase.table("files").select("*").eq("session_id", database_session_id).execute()
        existing_files = {f['file_name']: f for f in (existing_response.data or [])}
        logger.info(f"Found {len(existing_files)} existing files in database")

        # 2. Prepare data for INSERT, UPDATE, DELETE operations
        new_file_names = {item['file_name'] for item in batch_data}
        files_to_insert = []
        files_to_update = []
        files_to_delete = []
        upserted_uuids = []

        # 3. Classify each file
        for file_data in batch_data:
            file_name = file_data['file_name']
            if file_name in existing_files:
                # File exists - UPDATE with existing UUID
                existing_file = existing_files[file_name]
                file_data['id'] = existing_file['id']  # Keep existing UUID
                files_to_update.append(file_data)
                upserted_uuids.append(existing_file['id'])
                logger.info(f"File {file_name} exists - will UPDATE")
            else:
                # New file - INSERT
                files_to_insert.append(file_data)
                logger.info(f"File {file_name} is new - will INSERT")

        # 4. Identify files to DELETE (exist in DB but not in new data)
        for file_name, existing_file in existing_files.items():
            if file_name not in new_file_names:
                files_to_delete.append(existing_file['id'])
                logger.info(f"File {file_name} removed - will DELETE")

        # 5. Execute INSERT operations
        if files_to_insert:
            logger.info(f"Inserting {len(files_to_insert)} new files")
            insert_response = supabase.table("files").insert(files_to_insert).execute()
            if hasattr(insert_response, 'error') and insert_response.error:
                raise DatabaseError(f"Batch file insert failed: {insert_response.error}")
            inserted_uuids = [item['id'] for item in insert_response.data if 'id' in item]
            upserted_uuids.extend(inserted_uuids)
            logger.info(f"Successfully inserted {len(inserted_uuids)} new files")

        # 6. Execute UPDATE operations
        if files_to_update:
            logger.info(f"Updating {len(files_to_update)} existing files")
            for file_data in files_to_update:
                file_id = file_data.pop('id')  # Remove ID from data, use it in eq()
                update_response = supabase.table("files").update(file_data).eq("id", file_id).execute()
                if hasattr(update_response, 'error') and update_response.error:
                    logger.error(f"Failed to update file {file_data.get('file_name')}: {update_response.error}")
                else:
                    logger.info(f"Successfully updated file {file_data.get('file_name')}")

        # 7. Execute DELETE operations
        if files_to_delete:
            logger.info(f"Deleting {len(files_to_delete)} removed files")
            for file_id in files_to_delete:
                delete_response = supabase.table("files").delete().eq("id", file_id).execute()
                if hasattr(delete_response, 'error') and delete_response.error:
                    logger.error(f"Failed to delete file {file_id}: {delete_response.error}")
                else:
                    logger.info(f"Successfully deleted file {file_id}")

        logger.info(f"UPSERT completed: {len(files_to_insert)} inserted, {len(files_to_update)} updated, {len(files_to_delete)} deleted")
        return upserted_uuids

    except Exception as e:
        raise DatabaseError(f"Batch file upsert failed: {str(e)}")

def _batch_insert_files(supabase, batch_data: list) -> list:
    """
    DEPRECATED: Use _batch_upsert_files instead.
    Perform batch insertion of file data.

    Args:
        supabase: Supabase client
        batch_data: List of file data to insert

    Returns:
        list: List of successfully inserted file UUIDs
    """
    if not batch_data:
        return []

    try:
        logger.info(f"Performing batch insert of {len(batch_data)} files")
        response = supabase.table("files").insert(batch_data).execute()

        if hasattr(response, 'error') and response.error:
            raise DatabaseError(f"Batch file insert failed: {response.error}")

        inserted_uuids = [item['id'] for item in response.data if 'id' in item]
        logger.info(f"Successfully inserted {len(inserted_uuids)} files in batch")
        return inserted_uuids

    except Exception as e:
        raise DatabaseError(f"Batch file insert failed: {str(e)}")

def _save_files_to_database(database_session_id: str, session_id: str, metadata: dict) -> bool:
    """
    Save file info and content to database and storage using batch operations.
    
    Args:
        database_session_id: UUID format session ID
        session_id: Original session ID for directory structure
        metadata: Session metadata dictionary
        
    Returns:
        bool: True if successful, False otherwise
    """
    if 'files' not in metadata or not isinstance(metadata['files'], list):
        return True
    
    supabase = get_supabase_client()
    if not supabase:
        raise ConfigurationError("Supabase client not available")
    
    upload_base_dir = 'uploads/file_uploads'
    session_dir = os.path.join(upload_base_dir, session_id)
    
    try:
        batch_data = _prepare_file_batch_data(database_session_id, metadata['files'])
        if not batch_data:
            logger.warning("No valid file data to upsert")
            return True

        upserted_uuids = _batch_upsert_files(supabase, database_session_id, batch_data)
        if not upserted_uuids:
            logger.error("Batch file upsert failed - no files were processed")
            return False
        
        uuid_map = {}
        for data, uuid_val in zip(batch_data, upserted_uuids):
            file_name = data.get('file_name', '')
            if file_name:
                uuid_map[file_name] = {
                    'uuid': uuid_val,
                    'type': data.get('type', DatabaseConfig.DEFAULT_FILE_TYPE)
                }
        
        upload_success = True
        for file_info in metadata['files']:
            file_name = file_info.get('fileName', '')
            if file_name not in uuid_map:
                logger.warning(f"File {file_name} not found in UUID mapping, skipping upload")
                continue
                
            file_uuid = uuid_map[file_name]['uuid']
            file_type = uuid_map[file_name]['type']
            file_path = os.path.join(session_dir, file_name)
            
            if os.path.exists(file_path):
                logger.info(f"Uploading file {file_name} to Supabase Storage...")
                try:
                    if not save_csv_file_content(file_uuid, database_session_id, file_name, file_path, file_type):
                        logger.error(f"❌ Failed to upload {file_name} to storage")
                        upload_success = False
                    else:
                        logger.info(f"✅ Successfully uploaded {file_name} to storage")
                except Exception as e:
                    logger.error(f"❌ Error uploading {file_name}: {str(e)}")
                    upload_success = False
            else:
                logger.warning(f"CSV file not found locally: {file_path}")
                upload_success = False
        
        if not upload_success:
            logger.warning("Some file uploads failed, but metadata was saved successfully")

        logger.info(f"Batch file UPSERT completed: {len(upserted_uuids)} files metadata saved")
        return True
        
    except Exception as e:
        if isinstance(e, (DatabaseError, ConfigurationError)):
            raise
        raise DatabaseError(f"Error in batch file save operation: {str(e)}")

def _finalize_session(database_session_id: str, n_dat: int = None, file_count: int = None) -> bool:
    """
    Update sessions table with finalization data.
    
    Args:
        database_session_id: UUID format session ID
        n_dat: Total number of data samples (optional)
        file_count: Number of files in the session (optional)
        
    Returns:
        bool: True if successful, False otherwise
    """
    if n_dat is None and file_count is None:
        return True
    
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
        if not supabase:
            logger.error("Supabase client not available for session finalization")
            return False
            
        session_response = supabase.table("sessions").update(session_update_data).eq("id", database_session_id).execute()
        if hasattr(session_response, 'error') and session_response.error:
            logger.error(f"Error updating sessions table: {session_response.error}")
            return False
        else:
            logger.info(f"Successfully updated sessions table with n_dat={n_dat}, file_count={file_count}")
            return True
    except Exception as e:
        logger.error(f"Error updating sessions table: {str(e)}")
        return False

def update_session_name(session_id: str, session_name: str, user_id: str = None) -> bool:
    """
    Update session name in the sessions table.

    Args:
        session_id: ID of the session (can be string or UUID)
        session_name: New name for the session
        user_id: User ID to validate session ownership (required for security)

    Returns:
        bool: True if successful, False otherwise

    Raises:
        ValidationError: If input parameters are invalid
        ConfigurationError: If Supabase client is not available
        SessionNotFoundError: If session cannot be found
        PermissionError: If session doesn't belong to the user
        DatabaseError: If database operations fail
    """
    if not validate_session_id(session_id):
        raise ValidationError(f"Invalid session_id format: {session_id}")

    if not session_name or not isinstance(session_name, str):
        raise ValidationError(f"Invalid session_name: {session_name}")

    session_name = session_name.strip()
    if len(session_name) == 0:
        raise ValidationError("Session name cannot be empty")

    if len(session_name) > 255:
        raise ValidationError("Session name too long (max 255 characters)")

    supabase = get_supabase_client()
    if not supabase:
        raise ConfigurationError("Supabase client not available")

    database_session_id = _get_session_uuid(session_id)
    if not database_session_id:
        raise SessionNotFoundError(f"Failed to convert session_id {session_id} to UUID")

    logger.info(f"Updating session name for {database_session_id} to: {session_name}")

    try:
        # Validate session ownership if user_id provided
        session_query = supabase.table("sessions").select("id, user_id").eq("id", database_session_id)
        existing = session_query.execute()

        if not existing.data or len(existing.data) == 0:
            raise SessionNotFoundError(f"Session {database_session_id} not found")

        # Verify session belongs to user (if user_id provided)
        if user_id and existing.data[0].get('user_id') != user_id:
            logger.warning(f"User {user_id} attempted to rename session {database_session_id} owned by {existing.data[0].get('user_id')}")
            raise PermissionError(f"Session {session_id} does not belong to user")

        response = supabase.table("sessions").update({
            "session_name": session_name,
            "updated_at": datetime.now().isoformat()
        }).eq("id", database_session_id).execute()

        if hasattr(response, 'error') and response.error:
            raise DatabaseError(f"Error updating session name: {response.error}")

        logger.info(f"Successfully updated session name for {database_session_id}")
        return True

    except Exception as e:
        if isinstance(e, (DatabaseError, SessionNotFoundError)):
            raise
        raise DatabaseError(f"Error updating session name: {str(e)}")

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
        logger.info(f"save_session_to_supabase called with session_id: {session_id}")

        database_session_id = _get_session_uuid(session_id)
        if not database_session_id:
            return False

        metadata = _load_session_metadata(session_id)
        if not metadata:
            return False

        if not _save_metadata_to_database(database_session_id, metadata):
            logger.warning("Some metadata failed to save, continuing with files...")

        if not _save_files_to_database(database_session_id, session_id, metadata):
            logger.warning("Some files failed to save, continuing with finalization...")

        if not _finalize_session(database_session_id, n_dat, file_count):
            logger.warning("Session finalization failed, but core data was saved")

        logger.info(f"Successfully saved session {session_id} to Supabase")
        return True

    except Exception as e:
        logger.error(f"Error saving session to Supabase: {str(e)}")
        return False
