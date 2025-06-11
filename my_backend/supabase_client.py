import os
import json
import logging
import uuid
import base64
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
        logger.error("Supabase URL or key not found in environment variables")
        return None
        
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Error creating Supabase client: {str(e)}")
        return None

def save_time_info(session_id: str, time_info: dict) -> bool:
    """
    Save time information to the time_info table.
    
    Args:
        session_id: ID of the session
        time_info: Dictionary containing time information
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
            
        # Prepare data for insertion
        data = {
            "session_id": session_id,
            "jahr": time_info.get("jahr", False),
            "woche": time_info.get("woche", False),
            "monat": time_info.get("monat", False),
            "feiertag": time_info.get("feiertag", False),
            "lokalzeit": time_info.get("lokalzeit", False),
            "land": time_info.get("land", ""),
            "zeitzone": time_info.get("zeitzone", "")
        }
        
        # Prvo proverimo da li već postoji zapis za ovu sesiju
        existing = supabase.table("time_info").select("*").eq("session_id", session_id).execute()
        
        if existing.data and len(existing.data) > 0:
            # Ako postoji, ažuriramo postojeći zapis
            response = supabase.table("time_info").update(data).eq("session_id", session_id).execute()
            logger.info(f"Updated existing time_info for session {session_id}")
        else:
            # Ako ne postoji, dodajemo novi zapis
            response = supabase.table("time_info").insert(data).execute()
            logger.info(f"Inserted new time_info for session {session_id}")
        
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error saving time_info: {response.error}")
            return False
            
        logger.info(f"Successfully saved time_info for session {session_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving time_info: {str(e)}")
        return False

def save_zeitschritte(session_id: str, zeitschritte: dict) -> bool:
    """
    Save zeitschritte information to the zeitschritte table.
    
    Args:
        session_id: ID of the session
        zeitschritte: Dictionary containing zeitschritte information
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
            
        # Prepare data for insertion
        data = {
            "session_id": session_id,
            "eingabe": zeitschritte.get("eingabe", ""),
            "ausgabe": zeitschritte.get("ausgabe", "")
        }
        
        # Prvo proverimo da li već postoji zapis za ovu sesiju
        existing = supabase.table("zeitschritte").select("*").eq("session_id", session_id).execute()
        
        if existing.data and len(existing.data) > 0:
            # Ako postoji, ažuriramo postojeći zapis
            response = supabase.table("zeitschritte").update(data).eq("session_id", session_id).execute()
            logger.info(f"Updated existing zeitschritte for session {session_id}")
        else:
            # Ako ne postoji, dodajemo novi zapis
            response = supabase.table("zeitschritte").insert(data).execute()
            logger.info(f"Inserted new zeitschritte for session {session_id}")
        
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error saving zeitschritte: {response.error}")
            return False
            
        logger.info(f"Successfully saved zeitschritte for session {session_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving zeitschritte: {str(e)}")
        return False

def save_file_info(session_id: str, file_info: dict) -> tuple:
    """
    Save file information to the files table.
    
    Args:
        session_id: ID of the session
        file_info: Dictionary containing file information
        
    Returns:
        tuple: (success, valid_uuid) where success is True if successful, False otherwise,
               and valid_uuid is the UUID used for the file (either validated or newly generated)
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
            
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
            
        # Prepare data for insertion
        data = {
            "id": valid_uuid,  # Koristimo validiran ili novi UUID
            "session_id": session_id,
            "file_name": file_info.get("fileName", ""),
            "bezeichnung": file_info.get("bezeichnung", ""),
            "utc_min": file_info.get("utcMin", None),
            "utc_max": file_info.get("utcMax", None),
            "zeitschrittweite": file_info.get("zeitschrittweite", ""),
            "min": file_info.get("min", ""),
            "max": file_info.get("max", ""),
            "offsett": file_info.get("offset", ""),  # Ispravljena kolona iz offset u offsett
            "datenpunkte": file_info.get("datenpunkte", ""),
            "numerische_datenpunkte": file_info.get("numerischeDatenpunkte", ""),
            "numerischer_anteil": file_info.get("numerischerAnteil", ""),
            "datenform": file_info.get("datenform", ""),
            "zeithorizont": file_info.get("zeithorizont", ""),
            "datenanpassung": file_info.get("datenanpassung", ""),
            "zeitschrittweite_mittelwert": file_info.get("zeitschrittweiteMittelwert", ""),
            "zeitschrittweite_min": file_info.get("zeitschrittweiteMin", ""),
            "skalierung": file_info.get("skalierung", "nein"),
            "skalierung_max": file_info.get("skalierungMax", ""),
            "skalierung_min": file_info.get("skalierungMin", ""),
            "type": file_info.get("type", "")
        }
        
        # Insert data into files table
        response = supabase.table("files").insert(data).execute()
        
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error saving file info: {response.error}")
            return False
            
        logger.info(f"Successfully saved file info for file {file_info.get('fileName')} in session {session_id}")
        return True, valid_uuid
        
    except Exception as e:
        logger.error(f"Error saving file info: {str(e)}")
        return False, None

def save_csv_file_content(file_id: str, session_id: str, file_name: str, file_path: str) -> bool:
    """
    Save CSV file content to Supabase Storage and save reference in the database.
    
    Args:
        file_id: ID of the file (from files table)
        session_id: ID of the session
        file_name: Name of the file
        file_path: Path to the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
            
        # Read file content
        with open(file_path, 'rb') as f:
            file_content = f.read()
            
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Definisanje putanje u Storage-u
        storage_path = f"{session_id}/{file_name}"
        
        try:
            # Upload fajla u Supabase Storage
            # Napomena: bucket 'csv-files' mora biti kreiran u Supabase konzoli
            storage_response = supabase.storage.from_("csv-files").upload(
                path=storage_path,
                file=file_content,
                file_options={"content-type": "text/csv"}
            )
            
            # Dobijanje javnog URL-a za fajl (opciono)
            # Ovo će raditi samo ako je bucket javni
            # file_url = supabase.storage.from_("csv-files").get_public_url(storage_path)
            
            # Proveri da li je file_id u UUID formatu, ako nije generiši novi UUID
            try:
                # Pokušaj konverziju u UUID format
                uuid_obj = uuid.UUID(file_id)
                valid_file_id = str(uuid_obj)
            except (ValueError, TypeError, AttributeError):
                # Ako konverzija ne uspe, generiši novi UUID
                valid_file_id = str(uuid.uuid4())
                logger.info(f"Generated new UUID {valid_file_id} for file reference {file_name}")
                
            # Čuvanje reference u bazi
            data = {
                "file_id": valid_file_id,
                "session_id": session_id,
                "file_name": file_name,
                "storage_path": storage_path,
                "file_size": file_size
            }
            
            # Proveravamo da li već postoji zapis za ovaj fajl
            existing = supabase.table("csv_file_refs").select("*").eq("file_id", valid_file_id).execute()
            
            if existing.data and len(existing.data) > 0:
                # Ažuriramo postojeći zapis
                db_response = supabase.table("csv_file_refs").update(data).eq("file_id", valid_file_id).execute()
                logger.info(f"Updated existing reference for file {file_name}")
            else:
                # Dodajemo novi zapis
                db_response = supabase.table("csv_file_refs").insert(data).execute()
                logger.info(f"Created new reference for file {file_name}")
            
            logger.info(f"Successfully saved CSV file {file_name} to storage for session {session_id}")
            return True
            
        except Exception as storage_error:
            logger.error(f"Error uploading to storage: {str(storage_error)}")
            return False
        
    except Exception as e:
        logger.error(f"Error saving CSV file content: {str(e)}")
        return False

def save_session_to_supabase(session_id: str) -> bool:
    """
    Save all session data to Supabase.
    
    Args:
        session_id: ID of the session
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Base directory for file uploads
        upload_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads', 'file_uploads')
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
            save_time_info(session_id, metadata['timeInfo'])
            
        # Save zeitschritte
        if 'zeitschritte' in metadata:
            save_zeitschritte(session_id, metadata['zeitschritte'])
            
        # Save file info and content
        if 'files' in metadata and isinstance(metadata['files'], list):
            for file_info in metadata['files']:
                # Save file info
                success, valid_uuid = save_file_info(session_id, file_info)
                
                # Save file content only if file info was saved successfully
                if success and valid_uuid:
                    file_name = file_info.get('fileName', '')
                    file_path = os.path.join(session_dir, file_name)
                    
                    if os.path.exists(file_path):
                        # Koristi valid_uuid umesto originalnog file_id
                        save_csv_file_content(valid_uuid, session_id, file_name, file_path)
                    else:
                        logger.warning(f"CSV file not found: {file_path}")
                else:
                    logger.warning(f"Failed to save file info for {file_info.get('fileName', 'unknown')}, skipping content upload")
        
        logger.info(f"Successfully saved session {session_id} to Supabase")
        return True
        
    except Exception as e:
        logger.error(f"Error saving session to Supabase: {str(e)}")
        return False
