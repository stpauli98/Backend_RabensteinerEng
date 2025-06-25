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
        
        logger.info(f"Processing time_info for session {session_id}")
        logger.info(f"time_info keys: {list(time_info.keys())}")
            
        # Prepare data for insertion with the new JSONB structure
        data = {
            "session_id": session_id,
            # Boolean flags for active categories
            "jahr": time_info.get("jahr", False),
            "woche": time_info.get("woche", False),
            "monat": time_info.get("monat", False),
            "feiertag": time_info.get("feiertag", False),
            "tag": time_info.get("tag", False),  # New tag category
            
            # Global timezone setting
            "zeitzone": time_info.get("zeitzone"),
            
            # JSONB structure for category-specific data
            "category_data": {}
        }
        
        # Check if the time_info already has category_data structure
        if "category_data" in time_info:
            logger.info("Using existing category_data structure from input")
            data["category_data"] = time_info["category_data"]
            # Make sure boolean flags match the category_data
            for category in time_info["category_data"]:
                if category in ["jahr", "woche", "monat", "feiertag", "tag"]:
                    data[category] = True
        else:
            # Build the category_data JSONB structure from old format
            logger.info("Building category_data structure from old format")
            category_data = {}
            
            # Only add data for active categories
            if time_info.get("jahr", False):
                category_data["jahr"] = {
                    "detaillierteBerechnung": time_info.get("jahr_detaillierteBerechnung", False),
                    "datenform": time_info.get("jahr_datenform", ""),
                    "zeithorizontStart": time_info.get("jahr_zeithorizontStart", ""),
                    "zeithorizontEnd": time_info.get("jahr_zeithorizontEnd", ""),
                    "skalierung": time_info.get("jahr_skalierung", "nein"),
                    "skalierungMin": time_info.get("jahr_skalierungMin", ""),
                    "skalierungMax": time_info.get("jahr_skalierungMax", "")
                }
            
            if time_info.get("monat", False):
                category_data["monat"] = {
                    "detaillierteBerechnung": time_info.get("monat_detaillierteBerechnung", False),
                    "datenform": time_info.get("monat_datenform", ""),
                    "zeithorizontStart": time_info.get("monat_zeithorizontStart", ""),
                    "zeithorizontEnd": time_info.get("monat_zeithorizontEnd", ""),
                    "skalierung": time_info.get("monat_skalierung", "nein"),
                    "skalierungMin": time_info.get("monat_skalierungMin", ""),
                    "skalierungMax": time_info.get("monat_skalierungMax", "")
                }
            
            if time_info.get("woche", False):
                category_data["woche"] = {
                    "detaillierteBerechnung": time_info.get("woche_detaillierteBerechnung", False),
                    "datenform": time_info.get("woche_datenform", ""),
                    "zeithorizontStart": time_info.get("woche_zeithorizontStart", ""),
                    "zeithorizontEnd": time_info.get("woche_zeithorizontEnd", ""),
                    "skalierung": time_info.get("woche_skalierung", "nein"),
                    "skalierungMin": time_info.get("woche_skalierungMin", ""),
                    "skalierungMax": time_info.get("woche_skalierungMax", "")
                }
            
            if time_info.get("tag", False):
                category_data["tag"] = {
                    "detaillierteBerechnung": time_info.get("tag_detaillierteBerechnung", False),
                    "datenform": time_info.get("tag_datenform", ""),
                    "zeithorizontStart": time_info.get("tag_zeithorizontStart", ""),
                    "zeithorizontEnd": time_info.get("tag_zeithorizontEnd", ""),
                    "skalierung": time_info.get("tag_skalierung", "nein"),
                    "skalierungMin": time_info.get("tag_skalierungMin", ""),
                    "skalierungMax": time_info.get("tag_skalierungMax", "")
                }
            
            if time_info.get("feiertag", False):
                category_data["feiertag"] = {
                    "detaillierteBerechnung": time_info.get("feiertag_detaillierteBerechnung", False),
                    "datenform": time_info.get("feiertag_datenform", ""),
                    "zeithorizontStart": time_info.get("feiertag_zeithorizontStart", ""),
                    "zeithorizontEnd": time_info.get("feiertag_zeithorizontEnd", ""),
                    "skalierung": time_info.get("feiertag_skalierung", "nein"),
                    "skalierungMin": time_info.get("feiertag_skalierungMin", ""),
                    "skalierungMax": time_info.get("feiertag_skalierungMax", ""),
                    "land": time_info.get("feiertag_land", "Deutschland")  # Special field for Feiertag
                }
            
            # Add the category_data to the main data object
            data["category_data"] = category_data
        
        # Detaljni logovi za praćenje podataka koji se šalju u bazu
        logger.info(f"Sending time_info data to database for session {session_id}:")
        logger.info(f"Boolean flags: jahr={data['jahr']}, monat={data['monat']}, woche={data['woche']}, tag={data['tag']}, feiertag={data['feiertag']}")
        logger.info(f"Zeitzone: {data['zeitzone']}")
        
        # Log category_data structure
        for category, category_values in data['category_data'].items():
            logger.info(f"Category '{category}' data:")
            for key, value in category_values.items():
                logger.info(f"  - {key}: {value}")
        
        # Prvo proverimo da li već postoji zapis za ovu sesiju
        existing = supabase.table("time_info").select("*").eq("session_id", session_id).execute()
        
        if existing.data and len(existing.data) > 0:
            # Ako postoji, ažuriramo postojeći zapis
            logger.info(f"Found existing time_info record for session {session_id}, updating...")
            response = supabase.table("time_info").update(data).eq("session_id", session_id).execute()
            logger.info(f"Updated existing time_info for session {session_id}")
        else:
            # Ako ne postoji, dodajemo novi zapis
            logger.info(f"No existing time_info record found for session {session_id}, inserting new record...")
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
            "ausgabe": zeitschritte.get("ausgabe", ""),
            "zeitschrittweite": zeitschritte.get("zeitschrittweite", ""),
            "offset": zeitschritte.get("offset", "")
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
            
        # Prepare data for insertion/update - strogo prema shemi tabele files
        # Pripremamo podatke točno prema shemi tabele files
        data = {
            "id": valid_uuid,
            "session_id": session_id,
            "file_name": file_info.get("fileName", ""),
            "bezeichnung": file_info.get("bezeichnung", ""),
            "min": file_info.get("min", ""),
            "max": file_info.get("max", ""),
            "offsett": file_info.get("offset", ""),  # Ispravljen naziv
            "datenpunkte": file_info.get("datenpunkte", ""), 
            "numerische_datenpunkte": file_info.get("numerischeDatenpunkte", ""),
            "numerischer_anteil": file_info.get("numerischerAnteil", ""),
            "datenform": file_info.get("datenform", ""),
            "datenanpassung": file_info.get("datenanpassung", ""),
            "zeitschrittweite": file_info.get("zeitschrittweite", ""),
            "zeitschrittweite_mittelwert": file_info.get("zeitschrittweiteAvgValue", file_info.get("zeitschrittweiteMittelwert", "")),
            "zeitschrittweite_min": file_info.get("zeitschrittweiteMinValue", file_info.get("zeitschrittweiteMin", "")),
            "skalierung": file_info.get("skalierung", "nein"),
            "skalierung_max": file_info.get("skalierungMax", ""),
            "skalierung_min": file_info.get("skalierungMin", ""),
            "zeithorizont_start": file_info.get("zeithorizontStart", ""),
            "zeithorizont_end": file_info.get("zeithorizontEnd", ""),
            "zeitschrittweite_transferierten_daten": file_info.get("zeitschrittweiteTransferiertenDaten", ""),
            "offset_transferierten_daten": file_info.get("offsetTransferiertenDaten", ""),
            "mittelwertbildung_uber_den_zeithorizont": file_info.get("mittelwertbildungÜberDenZeithorizont", "nein"),
            "storage_path": file_info.get("storagePath", ""),
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

def transform_time_info_to_new_format(time_info: dict) -> dict:
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
            
        # Save time info - transform to new format first
        if 'timeInfo' in metadata:
            # Log the original structure for debugging
            logger.info(f"Original timeInfo structure: {json.dumps(metadata['timeInfo'], indent=2)}")
            
            # Check if the timeInfo already has the new structure with category_data
            if 'category_data' in metadata['timeInfo']:
                logger.info("Using direct category_data structure from frontend")
                save_time_info(session_id, metadata['timeInfo'])
            else:
                # Transform old format to new format with JSONB structure
                transformed_time_info = transform_time_info_to_new_format(metadata['timeInfo'])
                save_time_info(session_id, transformed_time_info)
            
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
