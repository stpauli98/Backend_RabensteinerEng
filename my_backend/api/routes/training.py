import os
import json
import logging
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO, StringIO
from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
import tempfile
import glob
import shutil
from utils.database import save_session_to_supabase, get_string_id_from_uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create blueprint
bp = Blueprint('training', __name__)

def extract_file_metadata_fields(file_metadata):
    """
    Helper function to extract standardized file metadata fields from a file metadata dictionary.
    
    Args:
        file_metadata: Dictionary containing file metadata
        
    Returns:
        dict: Dictionary containing standardized file metadata fields
    """
    return {
        'id': file_metadata.get('id', ''),
        'fileName': file_metadata.get('fileName', ''),
        'bezeichnung': file_metadata.get('bezeichnung', ''),
        'utcMin': file_metadata.get('utcMin', ''),
        'utcMax': file_metadata.get('utcMax', ''),
        'zeitschrittweite': file_metadata.get('zeitschrittweite', ''),
        'min': file_metadata.get('min', ''),
        'max': file_metadata.get('max', ''),
        'offset': file_metadata.get('offset', ''),
        'datenpunkte': file_metadata.get('datenpunkte', ''),
        'numerischeDatenpunkte': file_metadata.get('numerischeDatenpunkte', ''),
        'numerischerAnteil': file_metadata.get('numerischerAnteil', ''),
        'datenform': file_metadata.get('datenform', ''),
        'zeithorizont': file_metadata.get('zeithorizont', ''),
        'datenanpassung': file_metadata.get('datenanpassung', ''),
        'zeitschrittweiteMittelwert': file_metadata.get('zeitschrittweiteMittelwert', ''),
        'zeitschrittweiteMin': file_metadata.get('zeitschrittweiteMin', ''),
        'skalierung': file_metadata.get('skalierung', ''),
        'skalierungMax': file_metadata.get('skalierungMax', ''),
        'skalierungMin': file_metadata.get('skalierungMin', ''),
        'type': file_metadata.get('type', '') # Dodajemo 'type' polje
    }

def extract_file_metadata(session_id):
    """
    Extracts file metadata from session metadata.
    
    Args:
        session_id: ID of the session to extract metadata from
        
    Returns:
        dict: Dictionary containing file metadata fields or None if not found
    """
    try:
        # Get path to session directory
        upload_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
        metadata_path = os.path.join(upload_dir, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file not found for session {session_id}")
            return None
            
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Find the first chunk which should contain fileMetadata
        for chunk in metadata:
            if 'params' in chunk and 'fileMetadata' in chunk['params']:
                file_metadata = chunk['params']['fileMetadata']
                return extract_file_metadata_fields(file_metadata)
                
        logger.error(f"No file metadata found for session {session_id}")
        return None
    except Exception as e:
        logger.error(f"Error extracting file metadata: {str(e)}")
        return None

# Constants for session management
MAX_SESSIONS_TO_RETURN = 50  # Maximum number of sessions to return in list

# Base directory for file uploads (relative to my_backend)
UPLOAD_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads', 'file_uploads')

def verify_file_hash(file_data: bytes, expected_hash: str) -> bool:
    """Verify the hash of file data matches the expected hash."""
    try:
        file_hash = hashlib.sha256(file_data).hexdigest()
        return file_hash == expected_hash
    except Exception as e:
        logger.error(f"Error verifying file hash: {str(e)}")
        return False

def assemble_file_locally(upload_id: str, filename: str) -> str:
    """
    Assemble a complete file from its chunks stored locally.
    
    Args:
        upload_id: Unique identifier for the upload session
        filename: Name of the file to assemble
        
    Returns:
        str: Path to the assembled file
    """
    try:
        # Definiraj direktorij gdje su spremljeni chunkovi
        upload_dir = os.path.join(UPLOAD_BASE_DIR, upload_id)
        
        # Provjeri postoji li direktorij
        if not os.path.exists(upload_dir):
            raise FileNotFoundError(f"Upload directory not found: {upload_dir}")
        
        # Učitaj metapodatke o chunkovima
        metadata_path = os.path.join(upload_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, 'r') as f:
            chunks_metadata = json.load(f)
        
        # Filtriraj samo chunkove za traženu datoteku
        file_chunks = [c for c in chunks_metadata if c['fileName'] == filename]
        
        if not file_chunks:
            raise FileNotFoundError(f"No chunks found for file {filename}")
            
        # Sortiraj chunkove po indeksu
        file_chunks.sort(key=lambda x: x['chunkIndex'])
        
        # Provjeri jesu li svi chunkovi prisutni
        expected_chunks = file_chunks[0]['totalChunks']
        if len(file_chunks) != expected_chunks:
            raise ValueError(f"Missing chunks for {filename}. Expected {expected_chunks}, found {len(file_chunks)}")
        
        # Kreiraj putanju za sastavljenu datoteku
        assembled_file_path = os.path.join(upload_dir, filename)
        
        # Sastavi datoteku
        with open(assembled_file_path, 'wb') as output_file:
            for chunk_info in file_chunks:
                chunk_path = chunk_info['filePath']
                if not os.path.exists(chunk_path):
                    raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
                    
                with open(chunk_path, 'rb') as chunk_file:
                    output_file.write(chunk_file.read())
        
        # Logiraj uspješno sastavljanje datoteke
        file_size = os.path.getsize(assembled_file_path)
        logger.info(f"Successfully assembled file {filename} from {len(file_chunks)} chunks, total size: {file_size} bytes")
        
        return assembled_file_path
    except Exception as e:
        logger.error(f"Error assembling file locally: {str(e)}")
        raise

def save_session_metadata_locally(session_id: str, metadata: dict) -> bool:
    """Save session metadata to local storage."""
    try:
        # Kreiraj putanju do lokalnog direktorija za upload
        upload_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
        
        # Kreiraj direktorij ako ne postoji
        os.makedirs(upload_dir, exist_ok=True)
        
        # Spremi metapodatke u JSON datoteku
        session_metadata_path = os.path.join(upload_dir, 'session_metadata.json')
        with open(session_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Session metadata saved locally to: {session_metadata_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving session metadata locally: {str(e)}")
        return False

def get_session_metadata_locally(session_id: str) -> dict:
    """Get session metadata from local storage."""
    try:
        # Kreiraj putanju do lokalnog direktorija za upload
        upload_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
        
        # Provjeri postoji li direktorij
        if not os.path.exists(upload_dir):
            logger.warning(f"Upload directory does not exist: {upload_dir}")
            return {}
        
        # Učitaj metapodatke o sesiji
        session_metadata_path = os.path.join(upload_dir, 'session_metadata.json')
        if os.path.exists(session_metadata_path):
            with open(session_metadata_path, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error retrieving session metadata locally: {str(e)}")
        return {}

def print_session_files(session_id, files_data):
    """
    Print information about files in a session for debugging purposes.
    
    Args:
        session_id: ID of the session
        files_data: Dictionary of file names and their data
    """
    try:
        # Only log minimal information unless in debug mode
        logger.info(f"Session {session_id} contains {len(files_data)} files")
        
        # Skip detailed logging if not in debug mode
        if not logger.isEnabledFor(logging.DEBUG):
            return
            
        # Get metadata
        metadata = get_session_metadata_locally(session_id)
                
        # Log timeInfo parameters
        time_info = metadata.get('timeInfo', {})
        if time_info:
            logger.debug(f"Time info: {json.dumps(time_info, indent=2)}")
        
        # Process each file
        for file_name, file_data in files_data.items():
            logger.debug(f"File: {file_name}, Size: {len(file_data)} bytes")
            
            # Try to parse as CSV but only log basic info
            try:
                df = pd.read_csv(BytesIO(file_data), encoding='utf-8')
                logger.debug(f"CSV rows: {len(df)}, columns: {len(df.columns)}")
            except Exception as e:
                logger.debug(f"Not parseable as CSV: {str(e)}")
                
            try:
                file_type = magic.from_buffer(file_data[:1024], mime=True)
                logger.debug(f"File type: {file_type}")
            except ImportError:
                # Ako magic modul nije dostupan, pokušaj odrediti tip na osnovu ekstenzije
                _, ext = os.path.splitext(file_name)
                logger.debug(f"File extension: {ext}")
                    
            # Ako je tekstualna datoteka, prikaži preview
            try:
                preview = file_data.decode('utf-8')[:1000]
                logger.debug("\nFile preview:")
                logger.debug(preview)
            except UnicodeDecodeError:
                logger.debug("Error decoding file as UTF-8 for preview")
                logger.debug(f"Binary file, size: {len(file_data)} bytes")
                
    except Exception as e:
        logger.error(f"Error processing session files: {str(e)}")

@bp.route('/upload-chunk', methods=['POST'])
def upload_chunk():
    """Handle chunk upload from frontend - saving locally."""
    try:
        # Provjeri ima li chunk u zahtjevu
        if 'chunk' not in request.files:
            return jsonify({'success': False, 'error': 'No chunk in request'}), 400
            
        chunk_file = request.files['chunk']
        if not chunk_file.filename:
            return jsonify({'success': False, 'error': 'No chunk file selected'}), 400
            
        # Dohvati metapodatke
        if 'metadata' not in request.form:
            return jsonify({'success': False, 'error': 'No metadata provided'}), 400
            
        metadata = json.loads(request.form['metadata'])
        logger.info(f"Received chunk {metadata['chunkIndex']} of {metadata['totalChunks']} for {metadata['fileName']}")
        
        # Pročitaj podatke chunka
        chunk_data = chunk_file.read()
        
        # Koristi session ID kao ID uploada
        upload_id = metadata['sessionId']
        
        # Kreiraj direktorij za spremanje chunkova ako ne postoji
        upload_dir = os.path.join(UPLOAD_BASE_DIR, upload_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Spremi chunk lokalno
        chunk_filename = f"{metadata['fileName']}_{metadata['chunkIndex']}"
        chunk_path = os.path.join(upload_dir, chunk_filename)
        
        with open(chunk_path, 'wb') as f:
            f.write(chunk_data)
        
        # Spremi metapodatke o chunku
        metadata_path = os.path.join(upload_dir, 'metadata.json')
        
        # Dohvati postojeće metapodatke ako postoje
        chunk_metadata = []
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    chunk_metadata = json.load(f)
            except json.JSONDecodeError:
                chunk_metadata = []
        
        # Ekstrahiraj sve parametre iz frontenda
        frontend_params = {}
        
        # Dodaj sve metapodatke
        for key, value in metadata.items():
            if key not in ['sessionId', 'chunkIndex', 'totalChunks', 'fileName', 'fileType']:
                frontend_params[key] = value
        
        # Obradi dodatne podatke iz zahtjeva
        if 'additionalData' in request.form:
            additional_data = json.loads(request.form['additionalData'])
            
            # Dodaj metapodatke o datoteci
            if 'fileMetadata' in additional_data:
                frontend_params['fileMetadata'] = additional_data['fileMetadata']
            
            # Dodaj informacije o sesiji
            if 'sessionInfo' in additional_data:
                frontend_params['sessionInfo'] = additional_data['sessionInfo']
            
            # Dodaj informacije o vremenu i zeitschritte
            if 'timeInfo' in additional_data:
                frontend_params['timeInfo'] = additional_data['timeInfo']
            if 'zeitschritte' in additional_data:
                frontend_params['zeitschritte'] = additional_data['zeitschritte']
        
        # Dodaj sve ostale parametre iz forme
        for key, value in request.form.items():
            if key not in ['metadata', 'additionalData']:
                try:
                    # Pokušaj parsirati kao JSON
                    frontend_params[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # Ako nije JSON, spremi kao tekst
                    frontend_params[key] = value
        
        # Dodaj query parametre ako postoje
        for key, value in request.args.items():
            frontend_params[key] = value
        
        # Logiraj parametre koji se spremaju
        if frontend_params:
            logger.info(f"Saving chunk with parameters: {', '.join(frontend_params.keys())}")
            
            # Detaljni ispis samo u debug modu
            if logger.isEnabledFor(logging.DEBUG):
                # Osnovni podaci iz metadata
                logger.debug(f"Session ID: {upload_id}")
                logger.debug(f"File name: {metadata.get('fileName', 'N/A')}")
                
                # Podaci iz fileMetadata ako postoje
                file_metadata = frontend_params.get('fileMetadata', {})
                if file_metadata:
                    logger.debug(f"File metadata: {json.dumps(file_metadata, indent=2)}")
        
        # Kreiraj metapodatke o chunku
        chunk_info = {
            'chunkIndex': metadata['chunkIndex'],
            'totalChunks': metadata['totalChunks'],
            'fileName': metadata['fileName'],
            'filePath': chunk_path,
            'fileType': metadata.get('fileType', 'unknown'),
            'createdAt': datetime.now().isoformat(),
            'params': frontend_params
        }
        
        # Dodaj ili ažuriraj metapodatke o chunku
        existing_chunk = next((c for c in chunk_metadata if c['chunkIndex'] == metadata['chunkIndex']), None)
        if existing_chunk:
            # Ažuriraj postojeći chunk
            existing_chunk.update(chunk_info)
        else:
            # Dodaj novi chunk
            chunk_metadata.append(chunk_info)
        
        # Spremi ažurirane metapodatke
        with open(metadata_path, 'w') as f:
            json.dump(chunk_metadata, f, indent=2)
        
        # Spremi session metadata lokalno ako je ovo prvi chunk
        if metadata['chunkIndex'] == 0 and 'additionalData' in request.form:
            additional_data = json.loads(request.form['additionalData'])
            
            # Dohvati metapodatke o datoteci
            file_metadata = additional_data.get('fileMetadata', {})
            file_type = file_metadata.get('type', 'unknown')
            file_name = metadata['fileName']
            
            # Dohvati informacije o sesiji
            session_info = additional_data.get('sessionInfo', {})
            
            # Dohvati informacije o vremenu i zeitschritte
            time_info = additional_data.get('timeInfo', {})
            zeitschritte = additional_data.get('zeitschritte', {})
            
            # Učitaj postojeće metapodatke o sesiji ako postoje
            session_metadata_path = os.path.join(upload_dir, 'session_metadata.json')
            session_metadata = {}
            if os.path.exists(session_metadata_path):
                try:
                    with open(session_metadata_path, 'r') as f:
                        session_metadata = json.load(f)
                except json.JSONDecodeError:
                    session_metadata = {}
            
            # Dodaj ili ažuriraj osnovne informacije o sesiji
            if 'timeInfo' not in session_metadata:
                session_metadata['timeInfo'] = time_info
            if 'zeitschritte' not in session_metadata:
                session_metadata['zeitschritte'] = zeitschritte
            if 'sessionInfo' not in session_metadata:
                session_metadata['sessionInfo'] = session_info
            
            # Dodaj ili ažuriraj metapodatke o datoteci
            if 'files' not in session_metadata:
                session_metadata['files'] = []
                
            # Provjeri postoji li već ova datoteka u metapodacima
            file_exists = False
            for i, existing_file in enumerate(session_metadata.get('files', [])):
                if existing_file.get('fileName') == file_name:
                    logger.debug(f"DEBUG: Updating existing file metadata for {file_name}: {file_metadata}")
                    # Ažuriraj postojeće metapodatke
                    session_metadata['files'][i] = file_metadata
                    file_exists = True
                    break
                    
            # Ako datoteka ne postoji u metapodacima, dodaj je
            if not file_exists and file_metadata:
                logger.debug(f"DEBUG: Adding new file metadata for {file_name}: {file_metadata}")
                session_metadata['files'].append(file_metadata)
            
            # Ažuriraj vrijeme zadnje promjene
            session_metadata['lastUpdated'] = datetime.now().isoformat()
            
            # Spremi metapodatke o sesiji lokalno
            with open(session_metadata_path, 'w') as f:
                json.dump(session_metadata, f, indent=2)
            
            logger.info(f"Saved session metadata for session {upload_id} with file {file_name}")
        
        # Ako je ovo zadnji chunk, sastavi datoteku
        if metadata['chunkIndex'] == metadata['totalChunks'] - 1:
            try:
                # Sastavi datoteku iz chunkova
                assembled_file_path = assemble_file_locally(upload_id, metadata['fileName'])
                
                # Prikaži samo osnovne informacije o datoteci
                file_size = os.path.getsize(assembled_file_path)
                logger.info(f"Successfully assembled file: {metadata['fileName']} at {assembled_file_path}, size: {file_size/1024:.2f} KB")
                
                # Učitaj datoteku kao DataFrame ako je CSV i prikaži osnovne informacije
                try:
                    df = pd.read_csv(assembled_file_path)
                    logger.info(f"CSV file processed: {len(df)} rows, {len(df.columns)} columns")
                except Exception:
                    # Nije CSV ili nije moguće učitati - preskoci
                    pass
                    
            except Exception as e:
                logger.error(f"Error assembling file: {str(e)}")
                return jsonify({
                    'success': False, 
                    'error': f"Error assembling file: {str(e)}"
                }), 500
        
        return jsonify({
            'success': True,
            'message': f"Successfully received chunk {metadata['chunkIndex']} of {metadata['totalChunks']}"
        })
        
    except Exception as e:
        logger.error(f"Error processing chunk upload: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def update_session_metadata(session_id, data):
    """
    Update session metadata with finalization info from request data.
    
    Args:
        session_id: ID of the session
        data: Request data containing timeInfo and zeitschritte
        
    Returns:
        dict: Updated metadata
    """
    existing_metadata = get_session_metadata_locally(session_id)
    
    finalization_metadata = {
        **existing_metadata,  # Keep existing metadata
        'finalized': True,
        'finalizationTime': datetime.now().isoformat(),
        'timeInfo': data.get('timeInfo', existing_metadata.get('timeInfo', {})),
        'zeitschritte': data.get('zeitschritte', existing_metadata.get('zeitschritte', {}))
    }
    
    save_session_metadata_locally(session_id, finalization_metadata)
    return finalization_metadata

def verify_session_files(session_id, metadata):
    """
    Verify files in the session directory and update metadata accordingly.
    
    Args:
        session_id: ID of the session
        metadata: Session metadata
        
    Returns:
        tuple: (updated_metadata, file_count)
    """
    upload_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
    file_count = 0
    files_metadata = metadata.get('files', [])
    
    # Ako nema metapodataka o datotekama, pokušaj ih pronaći u direktoriju
    if not files_metadata:
        for file_name in os.listdir(upload_dir):
            if os.path.isfile(os.path.join(upload_dir, file_name)) and not file_name.endswith(('_metadata.json', '.json')) and '_' not in file_name:
                file_count += 1
                # Dodaj osnovne metapodatke o datoteci
                files_metadata.append({
                    'fileName': file_name,
                    'createdAt': datetime.now().isoformat()
                })
    else:
        # Provjeri postoje li datoteke navedene u metapodacima
        for file_info in files_metadata:
            filename = file_info.get('fileName')
            if filename:
                file_path = os.path.join(upload_dir, filename)
                if os.path.exists(file_path):
                    file_count += 1
    
    # Ažuriraj metapodatke o datotekama
    metadata['files'] = files_metadata
    
    # Spremi ažurirane metapodatke
    save_session_metadata_locally(session_id, metadata)
    
    return metadata, file_count

def save_session_to_database(session_id):
    """
    Save session data to Supabase database.
    
    Args:
        session_id: ID of the session
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase_result = save_session_to_supabase(session_id)
        if supabase_result:
            logger.info(f"Session {session_id} data saved to Supabase successfully")
            return True
        else:
            logger.warning(f"Failed to save session {session_id} data to Supabase")
            return False
    except Exception as e:
        logger.error(f"Error saving session data to Supabase: {str(e)}")
        return False

@bp.route('/finalize-session', methods=['POST'])
def finalize_session():
    """Finalize a session after all files have been uploaded."""
    try:
        data = request.json
        if not data or 'sessionId' not in data:
            return jsonify({'success': False, 'error': 'No session ID provided'}), 400
            
        session_id = data['sessionId']
        
        # 1. Update session metadata with finalization info
        updated_metadata = update_session_metadata(session_id, data)
        
        # 2. Verify files and update metadata
        final_metadata, file_count = verify_session_files(session_id, updated_metadata)
        
        logger.info(f"Session {session_id} finalized with {file_count} files")

        # 3. Save session data to Supabase
        try:
            success = save_session_to_database(session_id)
            if not success:
                logger.warning(f"Failed to save session {session_id} to database, but continuing")
        except Exception as e:
            logger.error(f"Error saving session {session_id} to database: {str(e)}")
            # Continue even if database save fails - don't block the response
        
        return jsonify({
            'success': True,
            'message': f"Session {session_id} finalized successfully",
            'sessionId': session_id
        })
        
    except Exception as e:
        logger.error(f"Error finalizing session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/list-sessions', methods=['GET'])
def list_sessions():
    """List all available training sessions from local storage."""
    try:
        # Define base directory for file uploads
        base_dir = UPLOAD_BASE_DIR
        
        # Check if directory exists
        if not os.path.exists(base_dir):
            return jsonify({'success': True, 'sessions': []})
        
        # Get all session directories
        session_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        sessions = []
        for session_id in session_dirs:
            session_dir = os.path.join(base_dir, session_id)
            metadata_path = os.path.join(session_dir, 'session_metadata.json')
            
            # Get session metadata if available
            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except json.JSONDecodeError:
                    metadata = {}
            
            # Get file count
            files = [f for f in os.listdir(session_dir) if os.path.isfile(os.path.join(session_dir, f)) and not f.endswith('.json')]
            
            # Get creation time from directory stats
            created_at = datetime.fromtimestamp(os.path.getctime(session_dir)).isoformat()
            updated_at = datetime.fromtimestamp(os.path.getmtime(session_dir)).isoformat()
            
            session_info = {
                'sessionId': session_id,
                'createdAt': created_at,
                'lastUpdated': updated_at,
                'fileCount': len(files),
                'finalized': metadata.get('finalized', False),
                'timeInfo': metadata.get('timeInfo', {}),
                'zeitschritte': metadata.get('zeitschritte', {})
            }
            
            sessions.append(session_info)
        
        # Sort sessions by creation time (newest first)
        sessions.sort(key=lambda x: x['createdAt'], reverse=True)
        
        # Limit the number of sessions to return
        sessions = sessions[:MAX_SESSIONS_TO_RETURN]
        
        return jsonify({
            'success': True,
            'sessions': sessions
        })
        
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get detailed information about a specific session from local storage."""
    try:
        if not session_id:
            return jsonify({'success': False, 'error': 'No session ID provided'}), 400

        # Check if the provided ID is a UUID, and if so, get the string ID
        try:
            import uuid
            uuid.UUID(session_id)
            string_session_id = get_string_id_from_uuid(session_id)
            if not string_session_id:
                return jsonify({'success': False, 'error': 'Session mapping not found for UUID'}), 404
        except (ValueError, TypeError):
            string_session_id = session_id

        # Dohvati metapodatke o sesiji
        session_metadata = get_session_metadata_locally(string_session_id)
        
        # Koristimo lokalne podatke
        if not session_metadata:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
            
        # Dohvati informacije o datotekama
        upload_dir = os.path.join(UPLOAD_BASE_DIR, string_session_id)
        files = []
        
        if os.path.exists(upload_dir):
            for file_name in os.listdir(upload_dir):
                if os.path.isfile(os.path.join(upload_dir, file_name)) and not file_name.endswith('.json'):
                    file_path = os.path.join(upload_dir, file_name)
                    file_size = os.path.getsize(file_path)
                    files.append({
                        'fileName': file_name,
                        'size': file_size,
                        'createdAt': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                    })
        
        session_info = {
            'sessionId': string_session_id,
            'files': files,
            'timeInfo': session_metadata.get('timeInfo', {}),
            'zeitschritte': session_metadata.get('zeitschritte', {}),
            'finalized': session_metadata.get('finalized', False),
            'createdAt': datetime.fromtimestamp(os.path.getctime(upload_dir)).isoformat(),
            'lastUpdated': datetime.fromtimestamp(os.path.getmtime(upload_dir)).isoformat()
        }
        
        return jsonify({
            'success': True,
            'session': session_info
        })
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/session-status/<session_id>', methods=['GET'])
def session_status(session_id):
    """Get the status of a session."""
    try:
        if not session_id:
            return jsonify({
                'status': 'error',
                'progress': 0,
                'message': 'Missing session ID'
            }), 400
            
        # Check if the provided ID is a UUID, and if so, get the string ID
        try:
            import uuid
            uuid.UUID(session_id)
            # It's a UUID, get the original string ID for local file access
            string_session_id = get_string_id_from_uuid(session_id)
            if not string_session_id:
                 return jsonify({'status': 'error', 'message': 'Session mapping not found for UUID'}), 404
        except (ValueError, TypeError):
            # It's a string ID, use it directly
            string_session_id = session_id

        # Provjeri postoji li direktorij sesije
        upload_dir = os.path.join(UPLOAD_BASE_DIR, string_session_id)
        if not os.path.exists(upload_dir):
            return jsonify({
                'status': 'error',
                'progress': 0,
                'message': 'Session not found'
            }), 404
            
        # Učitaj metapodatke o sesiji
        session_metadata_path = os.path.join(upload_dir, 'session_metadata.json')
        if not os.path.exists(session_metadata_path):
            return jsonify({
                'status': 'pending',
                'progress': 10,
                'message': 'Session initialized but no metadata found'
            })
            
        with open(session_metadata_path, 'r') as f:
            session_metadata = json.load(f)
            
        # Provjeri je li sesija finalizirana
        if session_metadata.get('finalized', False):
            return jsonify({
                'status': 'completed',
                'progress': 100,
                'message': 'Session completed successfully'
            })
            
        # Ako nije finalizirana, provjeri koliko je datoteka uploadano
        metadata_path = os.path.join(upload_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                chunks_metadata = json.load(f)
                
            # Izračunaj progress na temelju broja uploadanih chunkova
            total_files = session_metadata.get('sessionInfo', {}).get('totalFiles', 0)
            if total_files > 0:
                # Pronađi jedinstvene datoteke
                unique_files = set()
                for chunk in chunks_metadata:
                    unique_files.add(chunk.get('fileName'))
                    
                progress = (len(unique_files) / total_files) * 90  # 90% za upload, 10% za finalizaciju
                
                return jsonify({
                    'status': 'processing',
                    'progress': int(progress),
                    'message': f'Uploading files: {len(unique_files)}/{total_files}'
                })
        
        # Ako nema metadata.json, sesija je tek inicijalizirana
        return jsonify({
            'status': 'pending',
            'progress': 5,
            'message': 'Session initialized, waiting for files'
        })
        
    except Exception as e:
        logger.error(f"Error getting session status for {session_id}: {str(e)}")
        return jsonify({
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }), 500

@bp.route('/get-file-metadata/<session_id>', methods=['GET'])
def get_file_metadata(session_id):
    """Get file metadata for a specific session."""
    try:
        # Extract file metadata
        metadata = extract_file_metadata(session_id)
        
        if metadata is None:
            return jsonify({'success': False, 'error': 'File metadata not found'}), 404
        
        return jsonify({
            'success': True,
            'metadata': metadata
        })
    except Exception as e:
        logger.error(f"Error getting file metadata for session {session_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/init-session', methods=['POST'])
def init_session():
    """Initialize a new upload session."""
    try:
        data = request.json
        session_id = data.get('sessionId')
        time_info = data.get('timeInfo', {})
        zeitschritte = data.get('zeitschritte', {})
        
        if not session_id:
            return jsonify({'success': False, 'error': 'Missing session ID'}), 400
            
        # Kreiraj direktorij za sesiju
        upload_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Spremi inicijalne metapodatke o sesiji
        session_metadata = {
            'timeInfo': time_info,
            'zeitschritte': zeitschritte,
            'sessionInfo': {
                'sessionId': session_id,
                'totalFiles': 0
            },
            'lastUpdated': datetime.now().isoformat()
        }
        
        # Save session metadata
        session_metadata_path = os.path.join(upload_dir, 'session_metadata.json')
        with open(session_metadata_path, 'w') as f:
            json.dump(session_metadata, f, indent=2)
            
        # Create session in Supabase and save session data
        try:
            from utils.database import create_or_get_session_uuid
            session_uuid = create_or_get_session_uuid(session_id)
            if session_uuid:
                logger.info(f"Created session UUID {session_uuid} for session {session_id}")
                
                # Save session data to Supabase
                success = save_session_to_supabase(session_id)
                if success:
                    logger.info(f"Session {session_id} data saved to Supabase successfully")
                else:
                    logger.warning(f"Failed to save session {session_id} data to Supabase")
            else:
                logger.warning(f"Failed to create session UUID for {session_id}")
        except Exception as e:
            logger.error(f"Error saving session data to Supabase: {str(e)}")
            # Continue even if Supabase save fails - don't block the response
            
        logger.info(f"Session {session_id} initialized successfully")
        return jsonify({
            'success': True,
            'sessionId': session_id,
            'message': f"Session {session_id} initialized successfully"
        })
    except Exception as e:
        logger.error(f"Error initializing session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/get-all-files-metadata/<session_id>', methods=['GET'])
def get_all_files_metadata(session_id):
    """Get metadata for all files in a session."""
    try:
        # Check if the provided ID is a UUID, and if so, get the string ID
        try:
            import uuid
            uuid.UUID(session_id)
            string_session_id = get_string_id_from_uuid(session_id)
            if not string_session_id:
                return jsonify({'success': False, 'error': 'Session mapping not found for UUID'}), 404
        except (ValueError, TypeError):
            string_session_id = session_id

        # Provjeri postoji li direktorij sesije
        upload_dir = os.path.join(UPLOAD_BASE_DIR, string_session_id)
        if not os.path.exists(upload_dir):
            return jsonify({'success': False, 'error': 'Session not found'}), 404
            
        # Prvo pokušaj učitati session_metadata.json jer sadrži metapodatke za sve datoteke
        session_metadata_path = os.path.join(upload_dir, 'session_metadata.json')
        if os.path.exists(session_metadata_path):
            with open(session_metadata_path, 'r') as f:
                session_metadata = json.load(f)
                
            # Dohvati metapodatke o datotekama iz session_metadata
            files_list = session_metadata.get('files', [])
            
            # Ako postoje metapodaci o datotekama u session_metadata, koristi njih
            if files_list:
                result = []
                for file_metadata in files_list:
                    if file_metadata:
                        # Provjeri postoji li datoteka
                        file_name = file_metadata.get('fileName', '')
                        file_path = os.path.join(upload_dir, file_name) if file_name else None
                        
                        if file_path and os.path.exists(file_path):
                            # Formatiraj metapodatke za frontend koristeći pomoćnu funkciju
                            result.append(extract_file_metadata_fields(file_metadata))
                
                return jsonify({
                    'success': True,
                    'files': result
                })
        
        # Ako nema session_metadata.json ili nema metapodataka o datotekama, pokušaj s metadata.json
        metadata_path = os.path.join(upload_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            # Ako nema ni metadata.json, pokušaj pronaći datoteke u direktoriju
            files_metadata = {}
            for file_name in os.listdir(upload_dir):
                if os.path.isfile(os.path.join(upload_dir, file_name)) and not file_name.endswith(('.json', '_0', '_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9')):
                    files_metadata[file_name] = {
                        'id': str(uuid.uuid4()),
                        'fileName': file_name,
                        'bezeichnung': os.path.splitext(file_name)[0]
                    }
            
            # Pretvori u listu za lakši prikaz u tablici
            result = list(files_metadata.values())
            
            return jsonify({
                'success': True,
                'files': result
            })
            
        # Učitaj metadata.json datoteku
        with open(metadata_path, 'r') as f:
            chunks_metadata = json.load(f)
            
        # Pronađi sve jedinstvene datoteke i njihove metapodatke
        files_metadata = {}
        for chunk in chunks_metadata:
            file_name = chunk.get('fileName')
            
            # Uzmi samo prvi chunk svake datoteke jer on sadrži metapodatke
            if file_name and chunk.get('chunkIndex') == 0 and file_name not in files_metadata:
                params = chunk.get('params', {})
                file_metadata = params.get('fileMetadata', {})
                
                if file_metadata:
                    files_metadata[file_name] = extract_file_metadata_fields(file_metadata)
        
        # Pretvori u listu za lakši prikaz u tablici
        result = list(files_metadata.values())
        
        return jsonify({
            'success': True,
            'files': result
        })
    except Exception as e:
        logger.error(f"Error getting all files metadata for session {session_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/save-time-info', methods=['POST'])
def save_time_info_endpoint():
    """Save time information via API endpoint."""
    try:
        # Log the raw request data for debugging
        logger.info(f"Received save-time-info request from {request.remote_addr}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request content length: {request.content_length}")
        
        # Check content type
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        # Get request data
        try:
            data = request.get_json(force=True)
        except Exception as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            return jsonify({'success': False, 'error': f'Invalid JSON: {str(e)}'}), 400
        
        # Log the parsed data
        logger.info(f"Parsed request data keys: {list(data.keys()) if data else 'None'}")
        
        if not data or 'sessionId' not in data or 'timeInfo' not in data:
            logger.error("Missing required fields in request")
            return jsonify({'success': False, 'error': 'Missing sessionId or timeInfo'}), 400
            
        session_id = data['sessionId']
        time_info = data['timeInfo']
        
        # Validate session_id format
        if not session_id or not isinstance(session_id, str):
            logger.error(f"Invalid session_id format: {session_id}")
            return jsonify({'success': False, 'error': 'Invalid session_id format'}), 400
        
        logger.info(f"Processing time_info save for session: {session_id}")
        logger.info(f"Time info keys: {list(time_info.keys()) if time_info else 'None'}")
        
        from utils.database import save_time_info
        success = save_time_info(session_id, time_info)
        
        if success:
            logger.info(f"Successfully saved time_info for session {session_id}")
            return jsonify({'success': True, 'message': 'Time info saved successfully'})
        else:
            logger.error(f"Failed to save time_info for session {session_id}")
            return jsonify({'success': False, 'error': 'Failed to save time info'}), 500
            
    except Exception as e:
        logger.error(f"Error saving time info: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/create-database-session', methods=['POST'])
def create_database_session():
    """Create a new session in Supabase database and return UUID."""
    try:
        data = request.json
        session_id = data.get('sessionId') if data else None
        
        from utils.database import create_or_get_session_uuid
        session_uuid = create_or_get_session_uuid(session_id)
        
        if session_uuid:
            return jsonify({
                'success': True, 
                'sessionUuid': session_uuid,
                'message': f'Database session created with UUID: {session_uuid}'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to create database session'}), 500
            
    except Exception as e:
        logger.error(f"Error creating database session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/test-data-loading/<session_id>', methods=['GET'])
def test_data_loading(session_id):
    """Test endpoint to check if data exists for a session ID."""
    try:
        # Get UUID for session
        from utils.database import create_or_get_session_uuid, get_supabase_client
        session_uuid = create_or_get_session_uuid(session_id)
        
        if not session_uuid:
            return jsonify({'success': False, 'error': 'Could not get session UUID'}), 400
            
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
            
        # Check for time_info
        time_info = supabase.table('time_info').select('*').eq('session_id', session_uuid).execute()
        
        # Check for zeitschritte
        zeitschritte = supabase.table('zeitschritte').select('*').eq('session_id', session_uuid).execute()
        
        # Check for files
        files = supabase.table('files').select('*').eq('session_id', session_uuid).execute()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'session_uuid': session_uuid,
            'data_exists': {
                'time_info': len(time_info.data) > 0,
                'zeitschritte': len(zeitschritte.data) > 0,
                'files': len(files.data) > 0
            },
            'data': {
                'time_info': time_info.data[0] if time_info.data else None,
                'zeitschritte': zeitschritte.data[0] if zeitschritte.data else None,
                'files_count': len(files.data)
            }
        })
        
    except Exception as e:
        logger.error(f"Error testing data loading: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/get-session-uuid/<session_id>', methods=['GET'])
def get_session_uuid(session_id):
    """Get the UUID session ID for a string session ID."""
    try:
        if not session_id:
            return jsonify({'success': False, 'error': 'Missing session ID'}), 400
        
        # Check if it's already a UUID
        try:
            import uuid
            uuid.UUID(session_id)
            # It's already a UUID
            return jsonify({
                'success': True,
                'sessionUuid': session_id,
                'message': 'Session ID is already in UUID format'
            })
        except (ValueError, TypeError):
            # It's a string session ID, get or create the UUID
            from utils.database import create_or_get_session_uuid
            session_uuid = create_or_get_session_uuid(session_id)
            
            if session_uuid:
                return jsonify({
                    'success': True,
                    'sessionUuid': session_uuid,
                    'message': f'Found/created UUID for session: {session_uuid}'
                })
            else:
                return jsonify({
                    'success': False, 
                    'error': f'Failed to get UUID for session {session_id}'
                }), 404
                
    except Exception as e:
        logger.error(f"Error getting session UUID: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/save-zeitschritte', methods=['POST'])
def save_zeitschritte_endpoint():
    """Save zeitschritte information via API endpoint."""
    try:
        # Log the raw request data for debugging
        logger.info(f"Received save-zeitschritte request from {request.remote_addr}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request content length: {request.content_length}")
        
        # Check content type
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        # Get request data
        try:
            data = request.get_json(force=True)
        except Exception as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            return jsonify({'success': False, 'error': f'Invalid JSON: {str(e)}'}), 400
        
        # Log the parsed data
        logger.info(f"Parsed request data keys: {list(data.keys()) if data else 'None'}")
        
        if not data or 'sessionId' not in data or 'zeitschritte' not in data:
            logger.error("Missing required fields in request")
            return jsonify({'success': False, 'error': 'Missing sessionId or zeitschritte'}), 400
            
        session_id = data['sessionId']
        zeitschritte = data['zeitschritte']
        
        # Validate session_id format
        if not session_id or not isinstance(session_id, str):
            logger.error(f"Invalid session_id format: {session_id}")
            return jsonify({'success': False, 'error': 'Invalid session_id format'}), 400
        
        logger.info(f"Processing zeitschritte save for session: {session_id}")
        logger.info(f"Zeitschritte keys: {list(zeitschritte.keys()) if zeitschritte else 'None'}")
        logger.info(f"Zeitschritte data received: {zeitschritte}")
        
        from utils.database import save_zeitschritte
        success = save_zeitschritte(session_id, zeitschritte)
        
        if success:
            logger.info(f"Successfully saved zeitschritte for session {session_id}")
            return jsonify({'success': True, 'message': 'Zeitschritte saved successfully'})
        else:
            logger.error(f"Failed to save zeitschritte for session {session_id}")
            return jsonify({'success': False, 'error': 'Failed to save zeitschritte'}), 500
            
    except Exception as e:
        logger.error(f"Error saving zeitschritte: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/session/<session_id>/delete', methods=['POST'])
def delete_session(session_id):
    """Delete a specific session and all its files from local storage."""
    try:
        # Check if the provided ID is a UUID, and if so, get the string ID
        try:
            import uuid
            uuid.UUID(session_id)
            string_session_id = get_string_id_from_uuid(session_id)
            if not string_session_id:
                return jsonify({'success': False, 'error': 'Session mapping not found for UUID'}), 404
        except (ValueError, TypeError):
            string_session_id = session_id

        # Definiraj putanju do lokalnog direktorija za upload
        upload_dir = os.path.join(UPLOAD_BASE_DIR, string_session_id)
        
        # Provjeri postoji li direktorij
        if not os.path.exists(upload_dir):
            logger.warning(f"Upload directory does not exist: {upload_dir}")
            return jsonify({
                'success': False,
                'error': 'Session not found'
            }), 404
        
        # Obriši sve datoteke u direktoriju
        for root, dirs, files in os.walk(upload_dir, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")
            
            # Obriši poddirektorije
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                try:
                    os.rmdir(dir_path)
                    logger.info(f"Deleted directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Error deleting directory {dir_path}: {str(e)}")
        
        # Na kraju obriši glavni direktorij
        try:
            os.rmdir(upload_dir)
            logger.info(f"Deleted session directory: {upload_dir}")
        except Exception as e:
            logger.error(f"Error deleting session directory {upload_dir}: {str(e)}")
        
        return jsonify({
            'success': True,
            'message': f"Session {string_session_id} deleted successfully"
        })
        
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/file/download/<session_id>/<file_type>/<file_name>', methods=['GET'])
def download_file(session_id, file_type, file_name):
    """
    Downloads a file from Supabase Storage.
    """
    try:
        from utils.database import get_supabase_client, get_string_id_from_uuid
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase client not available'}), 500

        # Ensure session_id is UUID for Supabase calls
        try:
            import uuid
            uuid.UUID(session_id)
            uuid_session_id = session_id
        except (ValueError, TypeError):
            uuid_session_id = get_string_id_from_uuid(session_id)
            if not uuid_session_id:
                return jsonify({'success': False, 'error': 'Session mapping not found'}), 404

        bucket_name = 'aus-csv-files' if file_type == 'output' else 'csv-files'
        storage_path = f"{uuid_session_id}/{file_name}"

        logger.info(f"Attempting to download file {file_name} from bucket {bucket_name} at path {storage_path}")

        try:
            response = supabase.storage.from_(bucket_name).download(storage_path)
            file_content = response
        except Exception as e:
            logger.error(f"Error downloading file from Supabase Storage: {e}")
            return jsonify({'success': False, 'error': f'File download failed: {str(e)}'}), 500

        # Return the file content
        return current_app.response_class(
            file_content,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{file_name}"'}
        )

    except Exception as e:
        logger.error(f"Error in download_file endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/run-analysis/<session_id>', methods=['POST'])
def run_analysis(session_id):
    """
    Triggers the execution of training pipeline using modern approach.
    Now uses TrainingPipeline directly instead of subprocess.
    """
    import threading
    from middleman_runner import ModernMiddlemanRunner
    from flask import current_app

    if not session_id:
        return jsonify({'success': False, 'error': 'Missing session ID'}), 400

    try:
        # Get SocketIO instance from app extensions
        socketio_instance = current_app.extensions.get('socketio')
        
        # Create modern middleman runner with SocketIO support
        runner = ModernMiddlemanRunner()
        if socketio_instance:
            runner.set_socketio(socketio_instance)  # Pass SocketIO for real-time updates
        
        # Run training in background thread to avoid blocking the request
        def run_training_async():
            try:
                logger.info(f"Starting async training for session {session_id}")
                result = runner.run_training_script(session_id)
                
                if result['success']:
                    logger.info(f"Training completed successfully for session {session_id}")
                    # Emit completion event via SocketIO
                    if socketio_instance:
                        socketio_instance.emit('training_completed', {
                            'session_id': session_id,
                            'status': 'completed',
                            'message': 'Training completed successfully'
                        }, room=session_id)
                else:
                    logger.error(f"Training failed for session {session_id}: {result.get('error', 'Unknown error')}")
                    # Emit error event via SocketIO
                    if socketio_instance:
                        socketio_instance.emit('training_error', {
                            'session_id': session_id,
                            'status': 'failed',
                            'error': result.get('error', 'Unknown error')
                        }, room=session_id)
                    
            except Exception as e:
                logger.error(f"Async training failed for session {session_id}: {str(e)}")
                if socketio_instance:
                    socketio_instance.emit('training_error', {
                        'session_id': session_id,
                        'status': 'failed',
                        'error': str(e)
                    }, room=session_id)
        
        # Start training in background thread
        training_thread = threading.Thread(target=run_training_async)
        training_thread.daemon = True
        training_thread.start()
        
        logger.info(f"Modern training pipeline for session {session_id} triggered successfully.")
        return jsonify({
            'success': True, 
            'message': f'Modern training pipeline for session {session_id} started.',
            'note': 'Using real extracted functions instead of subprocess'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to trigger modern training for session {session_id}: {e}")
        return jsonify({'success': False, 'error': f'Failed to start modern training: {str(e)}'}), 500

@bp.route('/get-zeitschritte/<session_id>', methods=['GET'])
def get_zeitschritte(session_id):
    """Get zeitschritte data for a session."""
    try:
        from utils.database import get_supabase_client
        supabase = get_supabase_client()
        
        # Get zeitschritte from database
        response = supabase.table('zeitschritte').select('*').eq('session_id', session_id).single().execute()
        
        if response.data:
            # Transform database data back to frontend format (offsett -> offset)
            data = dict(response.data)
            if 'offsett' in data:
                data['offset'] = data['offsett']
                del data['offsett']
            
            return jsonify({
                'success': True,
                'data': data
            })
        else:
            return jsonify({
                'success': False,
                'data': None,
                'message': 'No zeitschritte found for this session'
            }), 404
            
    except Exception as e:
        logger.error(f"Error getting zeitschritte for {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/get-time-info/<session_id>', methods=['GET'])
def get_time_info(session_id):
    """Get time info data for a session."""
    try:
        from utils.database import get_supabase_client
        supabase = get_supabase_client()
        
        # Get time_info from database
        response = supabase.table('time_info').select('*').eq('session_id', session_id).single().execute()
        
        if response.data:
            return jsonify({
                'success': True,
                'data': response.data
            })
        else:
            return jsonify({
                'success': False,
                'data': None,
                'message': 'No time info found for this session'
            }), 404
            
    except Exception as e:
        logger.error(f"Error getting time info for {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============== SUPABASE CRUD ENDPOINTS ==============

@bp.route('/supabase/test-connection', methods=['GET'])
def test_supabase_connection():
    """Test Supabase connection and return status."""
    try:
        from utils.database import get_supabase_client
        
        supabase_client = get_supabase_client()
        
        # Test by fetching sessions count
        response = supabase_client.table('sessions').select('id', count='exact').execute()
        
        return jsonify({
            'status': 'connected',
            'message': 'Successfully connected to Supabase',
            'sessions_count': response.count if hasattr(response, 'count') else len(response.data),
            'database_url': os.environ.get('SUPABASE_URL', 'Not configured')
        }), 200
        
    except Exception as e:
        logger.error(f"Supabase connection test failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to connect to Supabase: {str(e)}'
        }), 500

@bp.route('/supabase/files/create', methods=['POST'])
def create_file_record():
    """Create a new file record in Supabase."""
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        import uuid
        
        data = request.json
        session_id = data.get('session_id')
        file_data = data.get('file_data')
        
        if not session_id or not file_data:
            return jsonify({'success': False, 'error': 'Missing required data'}), 400
            
        # Get UUID session ID
        uuid_session_id = create_or_get_session_uuid(session_id)
        if not uuid_session_id:
            return jsonify({'success': False, 'error': 'Could not get session UUID'}), 400
            
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
            
        # Generate new UUID for file
        file_id = str(uuid.uuid4())
        
        # Prepare file record
        file_record = {
            'id': file_id,
            'session_id': uuid_session_id,
            'file_name': file_data.get('fileName'),
            'bezeichnung': file_data.get('bezeichnung'),
            'utc_min': file_data.get('utcMin'),
            'utc_max': file_data.get('utcMax'),
            'zeitschrittweite': str(file_data.get('zeitschrittweite', '')),
            'min': str(file_data.get('min', '')),
            'max': str(file_data.get('max', '')),
            'offsett': str(file_data.get('offset', '')),
            'datenpunkte': str(file_data.get('datenpunkte', '')),
            'numerische_datenpunkte': str(file_data.get('numerischeDatenpunkte', '')),
            'numerischer_anteil': str(file_data.get('numerischerAnteil', '')),
            'datenform': file_data.get('datenform', ''),
            'zeithorizont_start': file_data.get('zeithorizontStart', ''),
            'zeithorizont_end': file_data.get('zeithorizontEnd', ''),
            'zeitschrittweite_transferierten_daten': str(file_data.get('zeitschrittweiteTransferiertenDaten', '')),
            'offset_transferierten_daten': str(file_data.get('offsetTransferiertenDaten', '')),
            'mittelwertbildung_uber_den_zeithorizont': file_data.get('mittelwertbildungÜberDenZeithorizont', 'nein'),
            'datenanpassung': file_data.get('datenanpassung', ''),
            'zeitschrittweite_min': str(file_data.get('zeitschrittweiteMinValue', '')),
            'zeitschrittweite_mittelwert': str(file_data.get('zeitschrittweiteAvgValue', '')),
            'skalierung': file_data.get('skalierung', 'nein'),
            'skalierung_max': str(file_data.get('skalierungMax', '')),
            'skalierung_min': str(file_data.get('skalierungMin', '')),
            'storage_path': file_data.get('storagePath', f"{uuid_session_id}/{file_data.get('fileName', '')}"),
            'type': file_data.get('type', 'input')
        }
        
        # Insert into files table
        response = supabase.table('files').insert(file_record).execute()
        
        if response.data:
            # Also create csv_file_refs entry
            ref_record = {
                'file_id': file_id,
                'session_id': uuid_session_id,
                'file_name': file_data.get('fileName'),
                'storage_path': file_record['storage_path'],
                'file_size': file_data.get('fileSize')
            }
            supabase.table('csv_file_refs').insert(ref_record).execute()
            
            return jsonify({'success': True, 'file_id': file_id, 'data': response.data[0]})
        else:
            return jsonify({'success': False, 'error': 'Failed to create file record'}), 500
            
    except Exception as e:
        logger.error(f"Error creating file record: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/supabase/files/update/<file_id>', methods=['PUT'])
def update_file_record(file_id):
    """Update a file record in Supabase."""
    try:
        from utils.database import get_supabase_client
        
        file_data = request.json
        
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
            
        # Prepare update data
        update_data = {
            'file_name': file_data.get('fileName'),
            'bezeichnung': file_data.get('bezeichnung'),
            'utc_min': file_data.get('utcMin'),
            'utc_max': file_data.get('utcMax'),
            'zeitschrittweite': str(file_data.get('zeitschrittweite', '')),
            'min': str(file_data.get('min', '')),
            'max': str(file_data.get('max', '')),
            'offsett': str(file_data.get('offset', '')),
            'datenpunkte': str(file_data.get('datenpunkte', '')),
            'numerische_datenpunkte': str(file_data.get('numerischeDatenpunkte', '')),
            'numerischer_anteil': str(file_data.get('numerischerAnteil', '')),
            'datenform': file_data.get('datenform', ''),
            'zeithorizont_start': file_data.get('zeithorizontStart', ''),
            'zeithorizont_end': file_data.get('zeithorizontEnd', ''),
            'zeitschrittweite_transferierten_daten': str(file_data.get('zeitschrittweiteTransferiertenDaten', '')),
            'offset_transferierten_daten': str(file_data.get('offsetTransferiertenDaten', '')),
            'mittelwertbildung_uber_den_zeithorizont': file_data.get('mittelwertbildungÜberDenZeithorizont', 'nein'),
            'datenanpassung': file_data.get('datenanpassung', ''),
            'zeitschrittweite_min': str(file_data.get('zeitschrittweiteMinValue', '')),
            'zeitschrittweite_mittelwert': str(file_data.get('zeitschrittweiteAvgValue', '')),
            'skalierung': file_data.get('skalierung', 'nein'),
            'skalierung_max': str(file_data.get('skalierungMax', '')),
            'skalierung_min': str(file_data.get('skalierungMin', '')),
            'storage_path': file_data.get('storagePath', ''),
            'type': file_data.get('type', 'input')
        }
        
        # Update files table
        response = supabase.table('files').update(update_data).eq('id', file_id).execute()
        
        if response.data:
            # Update csv_file_refs
            ref_update = {
                'file_name': file_data.get('fileName'),
                'storage_path': update_data['storage_path'],
                'file_size': file_data.get('fileSize')
            }
            supabase.table('csv_file_refs').update(ref_update).eq('file_id', file_id).execute()
            
            return jsonify({'success': True, 'data': response.data[0]})
        else:
            return jsonify({'success': False, 'error': 'Failed to update file record'}), 500
            
    except Exception as e:
        logger.error(f"Error updating file record: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/supabase/files/delete/<file_id>', methods=['DELETE'])
def delete_file_record(file_id):
    """Delete a file record from Supabase."""
    try:
        from utils.database import get_supabase_client
        
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
            
        # Get file info first
        file_response = supabase.table('files').select('storage_path, type').eq('id', file_id).single().execute()
        
        if file_response.data:
            storage_path = file_response.data.get('storage_path')
            file_type = file_response.data.get('type', 'input')
            
            # Delete from storage
            if storage_path:
                bucket_name = 'aus-csv-files' if file_type == 'output' else 'csv-files'
                supabase.storage.from_(bucket_name).remove([storage_path])
            
            # Delete references
            supabase.table('csv_file_refs').delete().eq('file_id', file_id).execute()
            
            # Delete file record
            response = supabase.table('files').delete().eq('id', file_id).execute()
            
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'File not found'}), 404
            
    except Exception as e:
        logger.error(f"Error deleting file record: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/supabase/sessions/create', methods=['POST'])
def create_session_record():
    """Create a new session in Supabase."""
    try:
        from utils.database import get_supabase_client
        import uuid
        
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
            
        # Create new session
        response = supabase.table('sessions').insert({}).execute()
        
        if response.data:
            return jsonify({'success': True, 'session_id': response.data[0]['id']})
        else:
            return jsonify({'success': False, 'error': 'Failed to create session'}), 500
            
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/supabase/sessions/delete/<session_id>', methods=['DELETE'])
def delete_session_record(session_id):
    """Delete a session and all related data from Supabase."""
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        
        # Get UUID session ID
        uuid_session_id = create_or_get_session_uuid(session_id)
        if not uuid_session_id:
            return jsonify({'success': False, 'error': 'Could not get session UUID'}), 400
            
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
            
        # Get all files for deletion
        files_response = supabase.table('files').select('id, storage_path, type').eq('session_id', uuid_session_id).execute()
        
        # Delete storage files
        if files_response.data:
            for file in files_response.data:
                if file.get('storage_path'):
                    bucket_name = 'aus-csv-files' if file.get('type') == 'output' else 'csv-files'
                    try:
                        supabase.storage.from_(bucket_name).remove([file['storage_path']])
                    except:
                        pass  # Continue even if storage deletion fails
        
        # Delete in order (foreign key constraints)
        supabase.table('csv_file_refs').delete().eq('session_id', uuid_session_id).execute()
        supabase.table('files').delete().eq('session_id', uuid_session_id).execute()
        supabase.table('time_info').delete().eq('session_id', uuid_session_id).execute()
        supabase.table('zeitschritte').delete().eq('session_id', uuid_session_id).execute()
        supabase.table('training_logs').delete().eq('session_id', uuid_session_id).execute()
        supabase.table('training_progress').delete().eq('session_id', uuid_session_id).execute()
        supabase.table('training_results').delete().eq('session_id', uuid_session_id).execute()
        supabase.table('training_visualizations').delete().eq('session_id', uuid_session_id).execute()
        supabase.table('session_mappings').delete().eq('uuid_session_id', uuid_session_id).execute()
        supabase.table('sessions').delete().eq('id', uuid_session_id).execute()
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/debug-files-table/<session_id>', methods=['GET'])
def debug_files_table(session_id):
    """Debug endpoint to inspect files table data for a session."""
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        
        # Convert session_id to UUID if needed
        try:
            import uuid
            uuid.UUID(session_id)
            uuid_session_id = session_id
        except (ValueError, TypeError):
            uuid_session_id = create_or_get_session_uuid(session_id)
            if not uuid_session_id:
                return jsonify({'success': False, 'error': 'Could not get session UUID'}), 400
        
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
            
        # Get all files for this session
        response = supabase.table('files').select('*').eq('session_id', uuid_session_id).execute()
        
        logger.info(f"Debug: Files table query for session {session_id} (UUID: {uuid_session_id})")
        logger.info(f"Debug: Found {len(response.data)} files")
        
        # Also check csv_file_refs table
        refs_response = supabase.table('csv_file_refs').select('*').eq('session_id', uuid_session_id).execute()
        logger.info(f"Debug: Found {len(refs_response.data)} CSV file references")
        
        result = {
            'success': True,
            'session_id': session_id,
            'uuid_session_id': uuid_session_id,
            'files_count': len(response.data),
            'files': response.data,
            'csv_refs_count': len(refs_response.data),
            'csv_refs': refs_response.data
        }
        
        # Check for empty storage_path values
        empty_storage_paths = []
        for file_data in response.data:
            if not file_data.get('storage_path'):
                empty_storage_paths.append({
                    'file_id': file_data.get('id'),
                    'file_name': file_data.get('file_name'),
                    'storage_path': file_data.get('storage_path')
                })
        
        if empty_storage_paths:
            result['empty_storage_paths'] = empty_storage_paths
            result['empty_storage_paths_count'] = len(empty_storage_paths)
            logger.warning(f"Found {len(empty_storage_paths)} files with empty storage_path")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error debugging files table for {session_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500