import os
import json
import logging
import hashlib
import uuid
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO, StringIO
from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
import tempfile
import glob
import shutil
import re
import time
from pathlib import Path
from typing import Optional
from utils.database import save_session_to_supabase, get_string_id_from_uuid, create_or_get_session_uuid, get_supabase_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create blueprint
bp = Blueprint('training', __name__)

# Validation helper functions
def validate_session_id(session_id):
    """Validate session ID format - should be either UUID or session_XXXXXX_XXXXXX format"""
    if not session_id or not isinstance(session_id, str):
        return False

    # Check if it's a valid UUID
    try:
        uuid.UUID(session_id)
        return True
    except ValueError:
        pass

    # Check if it's in session_XXXXX_XXXXX format
    pattern = r'^session_\d+_[a-zA-Z0-9]+$'
    return bool(re.match(pattern, session_id))

def create_error_response(message, status_code=400):
    """Create standardized error response"""
    return jsonify({
        'success': False,
        'error': message,
        'data': None
    }), status_code

def create_success_response(data=None, message=None):
    """Create standardized success response"""
    response = {
        'success': True,
        'data': data
    }
    if message:
        response['message'] = message
    return jsonify(response)

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

def calculate_n_dat_from_session(session_id):
    """
    Calculate n_dat (total number of data samples) from uploaded CSV files in a session.
    This mimics the n_dat = i_array_3D.shape[0] calculation from training_original.py
    
    Args:
        session_id: ID of the session
        
    Returns:
        int: Total number of data samples (n_dat)
    """
    try:
        upload_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
        if not os.path.exists(upload_dir):
            logger.warning(f"Upload directory not found for session {session_id}")
            return 0
        
        total_samples = 0
        csv_files = []
        
        # Find all CSV files in the upload directory
        for file_name in os.listdir(upload_dir):
            if file_name.lower().endswith('.csv') and os.path.isfile(os.path.join(upload_dir, file_name)):
                csv_files.append(file_name)
        
        if not csv_files:
            logger.warning(f"No CSV files found in session {session_id}")
            return 0
        
        # Process each CSV file to count data samples
        for csv_file in csv_files:
            file_path = os.path.join(upload_dir, csv_file)
            try:
                # Read CSV file to count rows (excluding header)
                df = pd.read_csv(file_path)
                file_samples = len(df)
                total_samples += file_samples
                logger.info(f"File {csv_file}: {file_samples} samples")
            except Exception as e:
                logger.error(f"Error reading CSV file {csv_file}: {str(e)}")
                continue
        
        logger.info(f"Session {session_id} total n_dat: {total_samples}")
        return total_samples
        
    except Exception as e:
        logger.error(f"Error calculating n_dat for session {session_id}: {str(e)}")
        return 0

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
    Securely assemble a complete file from its chunks.

    Args:
        upload_id: Unique identifier for the upload session
        filename: Name of the file to assemble

    Returns:
        str: Path to the assembled file
    """
    # 🛡️ Security: Sanitize filename
    safe_filename = secure_filename(filename)
    if not safe_filename or safe_filename != filename:
        raise ValueError(f"Invalid filename: {filename}")

    # 🛡️ Security: Validate upload_id format
    if not upload_id.replace('_', '').replace('-', '').isalnum():
        raise ValueError(f"Invalid upload_id format: {upload_id}")

    upload_dir = os.path.join(UPLOAD_BASE_DIR, upload_id)

    # 🛡️ Security: Ensure upload_dir is within base directory
    upload_dir = os.path.abspath(upload_dir)
    base_dir = os.path.abspath(UPLOAD_BASE_DIR)
    if not upload_dir.startswith(base_dir):
        raise ValueError("Path traversal attempt detected")

    if not os.path.exists(upload_dir):
        raise FileNotFoundError(f"Upload directory not found: {upload_dir}")

    # Load and validate metadata
    metadata_path = os.path.join(upload_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            chunks_metadata = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Invalid metadata file: {e}")

    # Filter and validate chunks
    file_chunks = [c for c in chunks_metadata if c.get('fileName') == filename]
    if not file_chunks:
        raise FileNotFoundError(f"No chunks found for file {filename}")

    # Sort and validate chunk sequence
    file_chunks.sort(key=lambda x: x.get('chunkIndex', 0))
    expected_chunks = file_chunks[0].get('totalChunks', 0)

    if len(file_chunks) != expected_chunks:
        raise ValueError(
            f"Missing chunks for {filename}. "
            f"Expected {expected_chunks}, found {len(file_chunks)}"
        )

    # Validate chunk indices are sequential
    for i, chunk in enumerate(file_chunks):
        if chunk.get('chunkIndex') != i:
            raise ValueError(f"Non-sequential chunk index: expected {i}, got {chunk.get('chunkIndex')}")

    # 🛡️ Security: Ensure assembled file is in upload directory
    assembled_file_path = os.path.join(upload_dir, safe_filename)
    assembled_file_path = os.path.abspath(assembled_file_path)

    if not assembled_file_path.startswith(upload_dir):
        raise ValueError("Invalid file path detected")

    # ⚡ Performance: Stream assembly with buffer
    BUFFER_SIZE = 64 * 1024  # 64KB buffer

    try:
        with open(assembled_file_path, 'wb') as output_file:
            for chunk_info in file_chunks:
                chunk_path = chunk_info.get('filePath')
                if not chunk_path or not os.path.exists(chunk_path):
                    raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

                # 🛡️ Security: Validate chunk path
                chunk_path = os.path.abspath(chunk_path)
                if not chunk_path.startswith(base_dir):
                    raise ValueError("Invalid chunk path detected")

                # ⚡ Performance: Stream copy with buffer
                with open(chunk_path, 'rb') as chunk_file:
                    while True:
                        chunk_data = chunk_file.read(BUFFER_SIZE)
                        if not chunk_data:
                            break
                        output_file.write(chunk_data)

        file_size = os.path.getsize(assembled_file_path)
        logger.info(
            f"Successfully assembled {safe_filename} from {len(file_chunks)} chunks, "
            f"total size: {file_size:,} bytes"
        )

        return assembled_file_path

    except Exception as e:
        # Cleanup partial file on error
        if os.path.exists(assembled_file_path):
            try:
                os.remove(assembled_file_path)
            except OSError:
                pass
        logger.error(f"Error assembling file locally: {str(e)}")
        raise

def save_session_metadata_locally(session_id: str, metadata: dict) -> bool:
    """
    Securely save session metadata to local storage.

    Args:
        session_id: Session identifier (will be validated)
        metadata: Metadata dictionary to save

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # 🛡️ Security: Validate session_id
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Invalid session_id")

        # Remove any path traversal attempts
        clean_session_id = session_id.replace('..', '').replace('/', '').replace('\\', '')
        if clean_session_id != session_id:
            raise ValueError("Invalid characters in session_id")

        upload_dir = os.path.join(UPLOAD_BASE_DIR, clean_session_id)

        # 🛡️ Security: Ensure directory is within base
        upload_dir = os.path.abspath(upload_dir)
        base_dir = os.path.abspath(UPLOAD_BASE_DIR)
        if not upload_dir.startswith(base_dir):
            raise ValueError("Path traversal attempt detected")

        # Create directory with proper permissions
        os.makedirs(upload_dir, mode=0o755, exist_ok=True)

        # Save metadata with atomic write
        session_metadata_path = os.path.join(upload_dir, 'session_metadata.json')
        temp_path = session_metadata_path + '.tmp'

        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Atomic move
            os.rename(temp_path, session_metadata_path)

            logger.info(f"Session metadata saved: {session_metadata_path}")
            return True

        except Exception as e:
            # Cleanup temp file on error
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise e

    except Exception as e:
        logger.error(f"Error saving session metadata: {e}")
        return False

def get_session_metadata_locally(session_id: str) -> Optional[dict]:
    """
    Securely retrieve session metadata from local storage.

    Args:
        session_id: Session identifier

    Returns:
        dict: Metadata if found, None if not found
    """
    try:
        # 🛡️ Security: Validate session_id
        if not session_id or not isinstance(session_id, str):
            logger.warning("Invalid session_id provided")
            return {}

        clean_session_id = session_id.replace('..', '').replace('/', '').replace('\\', '')
        if clean_session_id != session_id:
            logger.warning("Invalid characters in session_id")
            return {}

        upload_dir = os.path.join(UPLOAD_BASE_DIR, clean_session_id)

        # 🛡️ Security: Ensure directory is within base
        upload_dir = os.path.abspath(upload_dir)
        base_dir = os.path.abspath(UPLOAD_BASE_DIR)
        if not upload_dir.startswith(base_dir):
            logger.warning("Path traversal attempt detected")
            return {}

        if not os.path.exists(upload_dir):
            logger.debug(f"Upload directory does not exist: {upload_dir}")
            return {}

        session_metadata_path = os.path.join(upload_dir, 'session_metadata.json')
        if not os.path.exists(session_metadata_path):
            logger.debug(f"Session metadata file not found: {session_metadata_path}")
            return {}

        with open(session_metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        logger.debug(f"Session metadata loaded for: {session_id}")
        return metadata

    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Error parsing session metadata for {session_id}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error retrieving session metadata for {session_id}: {e}")
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
        # logger.info(f"Session {session_id} contains {len(files_data)} files")
        
        # Skip detailed logging if not in debug mode
        if not logger.isEnabledFor(logging.DEBUG):
            return
            
        # Get metadata
        metadata = get_session_metadata_locally(session_id)
                
        # Log timeInfo parameters
        time_info = metadata.get('timeInfo', {})
        if time_info:
            # logger.debug(f"Time info: {json.dumps(time_info, indent=2)}")
            pass
        
        # Process each file
        for file_name, file_data in files_data.items():
            # logger.debug(f"File: {file_name}, Size: {len(file_data)} bytes")
            
            # Try to parse as CSV but only log basic info
            try:
                df = pd.read_csv(BytesIO(file_data), encoding='utf-8')
                # logger.debug(f"CSV rows: {len(df)}, columns: {len(df.columns)}")
            except Exception as e:
                # logger.debug(f"Not parseable as CSV: {str(e)}")
                pass
                
            try:
                file_type = magic.from_buffer(file_data[:1024], mime=True)
                # logger.debug(f"File type: {file_type}")
            except ImportError:
                # Ako magic modul nije dostupan, pokušaj odrediti tip na osnovu ekstenzije
                _, ext = os.path.splitext(file_name)
                # logger.debug(f"File extension: {ext}")
                    
            # Ako je tekstualna datoteka, prikaži preview
            try:
                preview = file_data.decode('utf-8')[:1000]
                # logger.debug("\nFile preview:")
                # logger.debug(preview)
            except UnicodeDecodeError:
                # logger.debug("Error decoding file as UTF-8 for preview")
                # logger.debug(f"Binary file, size: {len(file_data)} bytes")
                pass
                
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
        # logger.info(f"Received chunk {metadata['chunkIndex']} of {metadata['totalChunks']} for {metadata['fileName']}")
        
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
            # logger.info(f"Saving chunk with parameters: {', '.join(frontend_params.keys())}")
            
            # Detaljni ispis samo u debug modu
            if logger.isEnabledFor(logging.DEBUG):
                # Osnovni podaci iz metadata
                # logger.debug(f"Session ID: {upload_id}")
                # logger.debug(f"File name: {metadata.get('fileName', 'N/A')}")
                
                # Podaci iz fileMetadata ako postoje
                file_metadata = frontend_params.get('fileMetadata', {})
                if file_metadata:
                    # logger.debug(f"File metadata: {json.dumps(file_metadata, indent=2)}")
                    pass
        
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
                    # logger.debug(f"DEBUG: Updating existing file metadata for {file_name}: {file_metadata}")
                    # Ažuriraj postojeće metapodatke
                    session_metadata['files'][i] = file_metadata
                    file_exists = True
                    break
                    
            # Ako datoteka ne postoji u metapodacima, dodaj je
            if not file_exists and file_metadata:
                # logger.debug(f"DEBUG: Adding new file metadata for {file_name}: {file_metadata}")
                session_metadata['files'].append(file_metadata)
            
            # Ažuriraj vrijeme zadnje promjene
            session_metadata['lastUpdated'] = datetime.now().isoformat()
            
            # Spremi metapodatke o sesiji lokalno
            with open(session_metadata_path, 'w') as f:
                json.dump(session_metadata, f, indent=2)
            
            # logger.info(f"Saved session metadata for session {upload_id} with file {file_name}")
        
        # Ako je ovo zadnji chunk, sastavi datoteku
        if metadata['chunkIndex'] == metadata['totalChunks'] - 1:
            try:
                # Sastavi datoteku iz chunkova
                assembled_file_path = assemble_file_locally(upload_id, metadata['fileName'])
                
                # Prikaži samo osnovne informacije o datoteci
                file_size = os.path.getsize(assembled_file_path)
                # logger.info(f"Successfully assembled file: {metadata['fileName']} at {assembled_file_path}, size: {file_size/1024:.2f} KB")
                
                # Učitaj datoteku kao DataFrame ako je CSV i prikaži osnovne informacije
                try:
                    df = pd.read_csv(assembled_file_path)
                    # logger.info(f"CSV file processed: {len(df)} rows, {len(df.columns)} columns")
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

def save_session_to_database(session_id, n_dat=None, file_count=None):
    """
    Save session data to Supabase database.

    Args:
        session_id: ID of the session
        n_dat: Total number of data samples (optional)
        file_count: Number of files in the session (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase_result = save_session_to_supabase(session_id, n_dat, file_count)
        if supabase_result:
            # logger.info(f"Session {session_id} data saved to Supabase successfully")
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
        
        # 3. Calculate n_dat (total number of data samples)
        n_dat = calculate_n_dat_from_session(session_id)
        final_metadata['n_dat'] = n_dat
        
        # Save updated metadata with n_dat
        save_session_metadata_locally(session_id, final_metadata)
        
        # logger.info(f"Session {session_id} finalized with {file_count} files")

        # 4. Save session data to Supabase
        try:
            success = save_session_to_database(session_id, n_dat, file_count)
            if not success:
                logger.warning(f"Failed to save session {session_id} to database, but continuing")
        except Exception as e:
            logger.error(f"Error saving session {session_id} to database: {str(e)}")
            # Continue even if database save fails - don't block the response
        
        return jsonify({
            'success': True,
            'message': f"Session {session_id} finalized successfully",
            'sessionId': session_id,
            'n_dat': n_dat,
            'file_count': file_count
        })
        
    except Exception as e:
        logger.error(f"Error finalizing session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/list-sessions', methods=['GET'])
def list_sessions():
    """List all available training sessions from Supabase database."""
    try:
        from utils.database import get_supabase_client

        # Get Supabase client
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({
                'success': False,
                'error': 'Database connection not available'
            }), 500

        # Query sessions with related data
        # Get sessions with file counts and metadata
        sessions_query = """
        SELECT
            sm.string_session_id as session_id,
            s.session_name,
            s.created_at,
            s.updated_at,
            s.finalized,
            s.file_count,
            s.n_dat,
            (SELECT COUNT(*) FROM files f WHERE f.session_id = s.id) as actual_file_count,
            (SELECT COUNT(*) FROM zeitschritte z WHERE z.session_id = s.id) as zeitschritte_count,
            (SELECT COUNT(*) FROM time_info ti WHERE ti.session_id = s.id) as time_info_count,
            (SELECT json_agg(
                json_build_object(
                    'type', f.type,
                    'file_name', f.file_name,
                    'bezeichnung', f.bezeichnung
                )
            ) FROM files f WHERE f.session_id = s.id) as files_info
        FROM session_mappings sm
        JOIN sessions s ON sm.uuid_session_id = s.id
        ORDER BY s.created_at DESC
        LIMIT %s
        """

        # Execute query with limit
        limit = MAX_SESSIONS_TO_RETURN if 'MAX_SESSIONS_TO_RETURN' in globals() else 50

        try:
            # Use raw SQL query for complex JOIN
            response = supabase.rpc('execute_sql', {
                'sql_query': sessions_query,
                'params': [limit]
            }).execute()

            if response.data:
                sessions_data = response.data
            else:
                # Fallback to simpler query if RPC doesn't work
                sessions_response = supabase.table('session_mappings').select(
                    'string_session_id, sessions(id, session_name, created_at, updated_at, finalized, file_count, n_dat)'
                ).execute()

                sessions_data = []
                for session_mapping in sessions_response.data:
                    if session_mapping.get('sessions'):
                        session = session_mapping['sessions']
                        session_id = session_mapping['string_session_id']

                        # Get file count from files table
                        files_response = supabase.table('files').select('id, type, file_name, bezeichnung').eq('session_id', session['id']).execute()
                        file_count = len(files_response.data) if files_response.data else 0

                        # Get zeitschritte count
                        zeit_response = supabase.table('zeitschritte').select('id', count='exact').eq('session_id', session['id']).execute()
                        zeitschritte_count = zeit_response.count or 0

                        # Get time_info count
                        time_response = supabase.table('time_info').select('id', count='exact').eq('session_id', session['id']).execute()
                        time_info_count = time_response.count or 0

                        sessions_data.append({
                            'session_id': session_id,
                            'session_name': session.get('session_name'),
                            'created_at': session['created_at'],
                            'updated_at': session['updated_at'],
                            'finalized': session.get('finalized', False),
                            'file_count': session.get('file_count', 0),
                            'n_dat': session.get('n_dat', 0),
                            'actual_file_count': file_count,
                            'zeitschritte_count': zeitschritte_count,
                            'time_info_count': time_info_count,
                            'files_info': files_response.data if files_response.data else []
                        })

        except Exception as query_error:
            logger.warning(f"Complex query failed, using simple approach: {str(query_error)}")

            # Simple fallback query
            sessions_response = supabase.table('session_mappings').select(
                'string_session_id, uuid_session_id, created_at'
            ).order('created_at', desc=True).limit(limit).execute()

            sessions_data = []
            for session_mapping in sessions_response.data:
                session_uuid = session_mapping['uuid_session_id']
                session_id = session_mapping['string_session_id']

                # Get session details
                session_response = supabase.table('sessions').select('*').eq('id', session_uuid).execute()
                session_details = session_response.data[0] if session_response.data else {}

                # Get actual counts from database
                files_response = supabase.table('files').select('id, type, file_name, bezeichnung').eq('session_id', session_uuid).execute()
                file_count = len(files_response.data) if files_response.data else 0

                zeit_response = supabase.table('zeitschritte').select('id', count='exact').eq('session_id', session_uuid).execute()
                zeitschritte_count = zeit_response.count or 0

                time_response = supabase.table('time_info').select('id', count='exact').eq('session_id', session_uuid).execute()
                time_info_count = time_response.count or 0

                sessions_data.append({
                    'session_id': session_id,
                    'session_name': session_details.get('session_name'),
                    'created_at': session_mapping.get('created_at') or session_details.get('created_at'),
                    'updated_at': session_details.get('updated_at'),
                    'finalized': session_details.get('finalized', False),
                    'file_count': session_details.get('file_count', 0),
                    'n_dat': session_details.get('n_dat', 0),
                    'actual_file_count': file_count,
                    'zeitschritte_count': zeitschritte_count,
                    'time_info_count': time_info_count,
                    'files_info': files_response.data if files_response.data else []
                })

        # Format sessions for frontend
        sessions = []
        for session_data in sessions_data:
            session_info = {
                'sessionId': session_data['session_id'],
                'sessionName': session_data.get('session_name'),
                'createdAt': session_data['created_at'],
                'lastUpdated': session_data['updated_at'] or session_data['created_at'],
                'fileCount': session_data.get('actual_file_count', session_data.get('file_count', 0)),
                'finalized': session_data.get('finalized', False),
                'nDat': session_data.get('n_dat', 0),
                'zeitschritteCount': session_data.get('zeitschritte_count', 0),
                'timeInfoCount': session_data.get('time_info_count', 0),
                'filesInfo': session_data.get('files_info', [])
            }
            sessions.append(session_info)

        logger.info(f"Retrieved {len(sessions)} sessions from database")

        return jsonify({
            'success': True,
            'sessions': sessions,
            'total_count': len(sessions)
        })

    except Exception as e:
        logger.error(f"Error listing sessions from database: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to retrieve sessions from database'
        }), 500

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
            'n_dat': session_metadata.get('n_dat', 0),
            'file_count': len(files),
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

@bp.route('/session/<session_id>/database', methods=['GET'])
def get_session_from_database(session_id):
    """Get detailed information about a specific session from Supabase database."""
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
            database_session_id = session_id
        except (ValueError, TypeError):
            string_session_id = session_id
            database_session_id = create_or_get_session_uuid(session_id)
            if not database_session_id:
                return jsonify({'success': False, 'error': 'Could not find or create database session'}), 404

        supabase = get_supabase_client()
        if not supabase:
            return jsonify({'success': False, 'error': 'Database connection not available'}), 500

        # Get session data from database
        session_response = supabase.table('sessions').select('*').eq('id', database_session_id).execute()

        if not session_response.data or len(session_response.data) == 0:
            return jsonify({'success': False, 'error': 'Session not found in database'}), 404

        session_data = session_response.data[0]
        
        # Get files data
        files_response = supabase.table('files').select('*').eq('session_id', database_session_id).execute()
        files_data = files_response.data if files_response.data else []
        
        # Get time info
        time_info_response = supabase.table('time_info').select('*').eq('session_id', database_session_id).execute()
        time_info_data = time_info_response.data[0] if time_info_response.data else {}
        
        # Get zeitschritte
        zeitschritte_response = supabase.table('zeitschritte').select('*').eq('session_id', database_session_id).execute()
        zeitschritte_data = zeitschritte_response.data[0] if zeitschritte_response.data else {}

        # Format the response
        session_info = {
            'sessionId': string_session_id,
            'databaseSessionId': database_session_id,
            'n_dat': session_data.get('n_dat', 0),
            'finalized': session_data.get('finalized', False),
            'file_count': session_data.get('file_count', 0),
            'files': [
                {
                    'id': f['id'],
                    'fileName': f['file_name'],
                    'bezeichnung': f['bezeichnung'],
                    'min': f['min'],
                    'max': f['max'],
                    'datenpunkte': f['datenpunkte'],
                    'type': f['type']
                } for f in files_data
            ],
            'timeInfo': {
                'jahr': time_info_data.get('jahr', False),
                'monat': time_info_data.get('monat', False),
                'woche': time_info_data.get('woche', False),
                'tag': time_info_data.get('tag', False),
                'feiertag': time_info_data.get('feiertag', False),
                'zeitzone': time_info_data.get('zeitzone', 'UTC'),
                'category_data': time_info_data.get('category_data', {})
            },
            'zeitschritte': {
                'eingabe': zeitschritte_data.get('eingabe', ''),
                'ausgabe': zeitschritte_data.get('ausgabe', ''),
                'zeitschrittweite': zeitschritte_data.get('zeitschrittweite', ''),
                'offset': zeitschritte_data.get('offset', '')
            },
            'createdAt': session_data.get('created_at'),
            'updatedAt': session_data.get('updated_at')
        }
        
        return jsonify({
            'success': True,
            'session': session_info
        })
        
    except Exception as e:
        logger.error(f"Error getting session from database {session_id}: {str(e)}")
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

        # Provjeri postoji li sesija u bazi podataka
        from utils.database import get_supabase_client, create_or_get_session_uuid
        try:
            supabase = get_supabase_client()
            logger.info(f"Checking session status for: {string_session_id}")
            if supabase:
                # Provjeri postoji li sesija u bazi
                session_uuid = create_or_get_session_uuid(string_session_id)
                logger.info(f"Session UUID: {session_uuid}")
                if session_uuid:
                    # Sesija postoji u bazi
                    session_response = supabase.table('sessions').select('*').eq('id', session_uuid).execute()
                    logger.info(f"Session response data: {session_response.data}")
                    if session_response.data and len(session_response.data) > 0:
                        session_data = session_response.data[0]
                        # Sesija postoji u bazi, provjeri lokalne fajlove
                        upload_dir = os.path.join(UPLOAD_BASE_DIR, string_session_id)
                        if not os.path.exists(upload_dir):
                            # Sesija u bazi postoji ali lokalni fajlovi ne postoje
                            # Dozvoli korisniku da ponovo uploada fajlove
                            return jsonify({
                                'status': 'pending',
                                'progress': 0,
                                'message': 'Session exists but no files uploaded yet',
                                'finalized': session_data.get('finalized', False)
                            })
                    else:
                        # Sesija ne postoji u bazi
                        return jsonify({
                            'status': 'error',
                            'progress': 0,
                            'message': 'Session not found in database'
                        }), 404
                else:
                    # Ne može se pronaći UUID mapping
                    return jsonify({
                        'status': 'error',
                        'progress': 0,
                        'message': 'Session mapping not found'
                    }), 404
            else:
                # Nema Supabase klijenta, provjeri samo lokalno
                upload_dir = os.path.join(UPLOAD_BASE_DIR, string_session_id)
                if not os.path.exists(upload_dir):
                    return jsonify({
                        'status': 'error',
                        'progress': 0,
                        'message': 'Session not found'
                    }), 404
        except Exception as db_error:
            logger.warning(f"Database check failed: {str(db_error)}, falling back to local check")
            # Fallback na lokalnu provjeru
            upload_dir = os.path.join(UPLOAD_BASE_DIR, string_session_id)
            if not os.path.exists(upload_dir):
                return jsonify({
                    'status': 'error',
                    'progress': 0,
                    'message': 'Session not found'
                }), 404

        # Direktorij postoji, nastavi sa provjerom
        upload_dir = os.path.join(UPLOAD_BASE_DIR, string_session_id)
            
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
                # logger.info(f"Created session UUID {session_uuid} for session {session_id}")
                
                # Save session data to Supabase
                success = save_session_to_supabase(session_id)
                if success:
                    # logger.info(f"Session {session_id} data saved to Supabase successfully")
                    pass
                else:
                    logger.warning(f"Failed to save session {session_id} data to Supabase")
            else:
                logger.warning(f"Failed to create session UUID for {session_id}")
        except Exception as e:
            logger.error(f"Error saving session data to Supabase: {str(e)}")
            # Continue even if Supabase save fails - don't block the response
            
        # logger.info(f"Session {session_id} initialized successfully")
        return jsonify({
            'success': True,
            'sessionId': session_id,
            'message': f"Session {session_id} initialized successfully"
        })
    except Exception as e:
        logger.error(f"Error initializing session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/save-time-info', methods=['POST'])
def save_time_info_endpoint():
    """Save time information via API endpoint."""
    try:
        # Log the raw request data for debugging
        # logger.info(f"Received save-time-info request from {request.remote_addr}")
        # logger.info(f"Request headers: {dict(request.headers)}")
        # logger.info(f"Request content type: {request.content_type}")
        # logger.info(f"Request content length: {request.content_length}")
        
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
        # logger.info(f"Parsed request data keys: {list(data.keys()) if data else 'None'}")
        
        if not data or 'sessionId' not in data or 'timeInfo' not in data:
            logger.error("Missing required fields in request")
            return jsonify({'success': False, 'error': 'Missing sessionId or timeInfo'}), 400
            
        session_id = data['sessionId']
        time_info = data['timeInfo']
        
        # Validate session_id format
        if not session_id or not isinstance(session_id, str):
            logger.error(f"Invalid session_id format: {session_id}")
            return jsonify({'success': False, 'error': 'Invalid session_id format'}), 400
        
        # logger.info(f"Processing time_info save for session: {session_id}")
        # logger.info(f"Time info keys: {list(time_info.keys()) if time_info else 'None'}")
        
        from utils.database import save_time_info
        success = save_time_info(session_id, time_info)
        
        if success:
            # logger.info(f"Successfully saved time_info for session {session_id}")
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
        # logger.info(f"Received save-zeitschritte request from {request.remote_addr}")
        # logger.info(f"Request headers: {dict(request.headers)}")
        # logger.info(f"Request content type: {request.content_type}")
        # logger.info(f"Request content length: {request.content_length}")
        
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
        # logger.info(f"Parsed request data keys: {list(data.keys()) if data else 'None'}")
        
        if not data or 'sessionId' not in data or 'zeitschritte' not in data:
            logger.error("Missing required fields in request")
            return jsonify({'success': False, 'error': 'Missing sessionId or zeitschritte'}), 400
            
        session_id = data['sessionId']
        zeitschritte = data['zeitschritte']
        
        # Validate session_id format
        if not session_id or not isinstance(session_id, str):
            logger.error(f"Invalid session_id format: {session_id}")
            return jsonify({'success': False, 'error': 'Invalid session_id format'}), 400
        
        # logger.info(f"Processing zeitschritte save for session: {session_id}")
        # logger.info(f"Zeitschritte keys: {list(zeitschritte.keys()) if zeitschritte else 'None'}")
        
        from utils.database import save_zeitschritte
        success = save_zeitschritte(session_id, zeitschritte)
        
        if success:
            # logger.info(f"Successfully saved zeitschritte for session {session_id}")
            return jsonify({'success': True, 'message': 'Zeitschritte saved successfully'})
        else:
            logger.error(f"Failed to save zeitschritte for session {session_id}")
            return jsonify({'success': False, 'error': 'Failed to save zeitschritte'}), 500
            
    except Exception as e:
        logger.error(f"Error saving zeitschritte: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/session/<session_id>/delete', methods=['POST'])
def delete_session(session_id):
    """Delete a specific session and all its files from local storage and Supabase database."""
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        
        # Check if the provided ID is a UUID, and if so, get the string ID
        try:
            import uuid
            uuid.UUID(session_id)
            string_session_id = get_string_id_from_uuid(session_id)
            uuid_session_id = session_id
            if not string_session_id:
                return jsonify({'success': False, 'error': 'Session mapping not found for UUID'}), 404
        except (ValueError, TypeError):
            string_session_id = session_id
            uuid_session_id = create_or_get_session_uuid(session_id)
            if not uuid_session_id:
                return jsonify({'success': False, 'error': 'Could not get session UUID'}), 404

        logger.info(f"Deleting session {string_session_id} (UUID: {uuid_session_id})")
        
        # 1. Delete from Supabase database first
        supabase = get_supabase_client()
        database_errors = []
        
        if supabase:
            try:
                # Delete from files table (this will also trigger CSV file storage deletion)
                files_response = supabase.table('files').select('storage_path, type').eq('session_id', uuid_session_id).execute()
                
                # Delete CSV files from Supabase Storage
                for file_data in files_response.data:
                    storage_path = file_data.get('storage_path')
                    file_type = file_data.get('type', 'input')
                    
                    if storage_path:
                        try:
                            bucket_name = 'aus-csv-files' if file_type == 'output' else 'csv-files'
                            supabase.storage.from_(bucket_name).remove([storage_path])
                            logger.info(f"Deleted file from storage: {bucket_name}/{storage_path}")
                        except Exception as storage_error:
                            logger.warning(f"Could not delete file from storage {storage_path}: {str(storage_error)}")
                            database_errors.append(f"Storage deletion failed: {storage_path}")
                
                # Delete from database tables (order matters due to foreign keys)
                tables_to_delete = [
                    'training_results',      # Results and visualizations
                    'training_visualizations',
                    'evaluation_tables',
                    'files',                 # File metadata
                    'zeitschritte',          # Time step data
                    'time_info',             # Time configuration
                    'session_mappings',      # Session mapping (last)
                    'sessions'               # Session record (last)
                ]
                
                for table in tables_to_delete:
                    try:
                        if table in ['session_mappings']:
                            # For session_mappings, use string_session_id
                            response = supabase.table(table).delete().eq('string_session_id', string_session_id).execute()
                        elif table in ['sessions']:
                            # For sessions, use uuid directly
                            response = supabase.table(table).delete().eq('id', uuid_session_id).execute()
                        else:
                            # For all other tables, use session_id
                            response = supabase.table(table).delete().eq('session_id', uuid_session_id).execute()
                        
                        deleted_count = len(response.data) if response.data else 0
                        if deleted_count > 0:
                            logger.info(f"Deleted {deleted_count} records from {table}")
                            
                    except Exception as table_error:
                        logger.error(f"Error deleting from {table}: {str(table_error)}")
                        database_errors.append(f"Table {table}: {str(table_error)}")
                        
            except Exception as db_error:
                logger.error(f"Database deletion error: {str(db_error)}")
                database_errors.append(f"Database error: {str(db_error)}")
        else:
            database_errors.append("Supabase client not available")

        # 2. Delete local files
        upload_dir = os.path.join(UPLOAD_BASE_DIR, string_session_id)
        local_errors = []
        
        if os.path.exists(upload_dir):
            # Delete all files in directory
            for root, dirs, files in os.walk(upload_dir, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted local file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {str(e)}")
                        local_errors.append(f"File {file_path}: {str(e)}")
                
                # Delete subdirectories
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    try:
                        os.rmdir(dir_path)
                        logger.info(f"Deleted local directory: {dir_path}")
                    except Exception as e:
                        logger.error(f"Error deleting directory {dir_path}: {str(e)}")
                        local_errors.append(f"Directory {dir_path}: {str(e)}")
            
            # Delete main directory
            try:
                os.rmdir(upload_dir)
                logger.info(f"Deleted session directory: {upload_dir}")
            except Exception as e:
                logger.error(f"Error deleting session directory {upload_dir}: {str(e)}")
                local_errors.append(f"Session directory: {str(e)}")
        else:
            logger.info(f"Local upload directory does not exist: {upload_dir}")
        
        # Prepare response
        all_errors = database_errors + local_errors
        
        if not all_errors:
            return jsonify({
                'success': True,
                'message': f"Session {string_session_id} completely deleted from database and local storage"
            })
        else:
            return jsonify({
                'success': True,  # Partial success
                'message': f"Session {string_session_id} deleted with some warnings",
                'warnings': all_errors
            })
        
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/get-zeitschritte/<session_id>', methods=['GET'])
def get_zeitschritte(session_id):
    """Get zeitschritte data for a session."""
    try:
        # Validate session_id format
        if not validate_session_id(session_id):
            return create_error_response('Invalid session ID format', 400)

        from utils.database import get_supabase_client, create_or_get_session_uuid
        supabase = get_supabase_client()

        if not supabase:
            return create_error_response('Database connection not available', 500)

        # Convert session_id to UUID format if needed
        try:
            uuid.UUID(session_id)
            database_session_id = session_id
        except (ValueError, TypeError):
            database_session_id = create_or_get_session_uuid(session_id)
            if not database_session_id:
                return create_error_response('Session not found', 404)
        
        # Get zeitschritte from database
        response = supabase.table('zeitschritte').select('*').eq('session_id', database_session_id).execute()
        
        if response.data and len(response.data) > 0:
            # Transform database data back to frontend format (offsett -> offset)
            data = dict(response.data[0])  # Take the first record
            if 'offsett' in data:
                data['offset'] = data['offsett']
                del data['offsett']

            return create_success_response(data)
        else:
            return create_error_response('No zeitschritte found for this session', 404)
            
    except Exception as e:
        logger.error(f"Error getting zeitschritte for {session_id}: {str(e)}")
        return create_error_response(f'Internal server error: {str(e)}', 500)

@bp.route('/get-time-info/<session_id>', methods=['GET'])
def get_time_info(session_id):
    """Get time info data for a session."""
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        supabase = get_supabase_client()
        
        # Convert session_id to UUID format if needed
        try:
            uuid.UUID(session_id)
            database_session_id = session_id
        except (ValueError, TypeError):
            database_session_id = create_or_get_session_uuid(session_id)
            if not database_session_id:
                return jsonify({'success': False, 'error': 'Invalid session ID'}), 400
        
        # Get time_info from database
        response = supabase.table('time_info').select('*').eq('session_id', database_session_id).execute()

        if response.data and len(response.data) > 0:
            return jsonify({
                'success': True,
                'data': response.data[0]
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

# CSV Files Management Endpoints

@bp.route('/csv-files/<session_id>', methods=['GET'])
def get_csv_files(session_id):
    """Get all CSV files for a session."""
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        supabase = get_supabase_client()
        
        # Convert session_id to UUID format if needed
        try:
            uuid.UUID(session_id)
            database_session_id = session_id
        except (ValueError, TypeError):
            database_session_id = create_or_get_session_uuid(session_id)
            if not database_session_id:
                return jsonify({'success': False, 'error': 'Invalid session ID'}), 400
        
        # Get file type filter from query params
        file_type = request.args.get('type', None)  # 'input' or 'output'
        
        # Build query
        query = supabase.table('files').select('*').eq('session_id', database_session_id)
        if file_type:
            query = query.eq('type', file_type)
        
        response = query.execute()
        
        if response.data:
            return jsonify({
                'success': True,
                'data': response.data
            })
        else:
            return jsonify({
                'success': True,
                'data': [],
                'message': f'No CSV files found for session {session_id}'
            })
            
    except Exception as e:
        logger.error(f"Error getting CSV files for session {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/csv-files', methods=['POST'])
def create_csv_file():
    """Create a new CSV file entry."""
    try:
        from utils.database import save_file_info, save_csv_file_content
        
        # Get JSON data and file from request
        if 'file' in request.files:
            # Handle file upload with form data
            file = request.files['file']
            session_id = request.form.get('sessionId')
            file_data = request.form.to_dict()
        else:
            # Handle JSON data only
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            session_id = data.get('sessionId')
            file_data = data.get('fileData', {})
            file = None
        
        if not session_id:
            return jsonify({'success': False, 'error': 'Session ID is required'}), 400
            
        # Save file metadata to database
        success, file_uuid = save_file_info(session_id, file_data)
        if not success:
            return jsonify({'success': False, 'error': 'Failed to save file metadata'}), 500
            
        # If file was uploaded, save to storage
        if file and file_uuid:
            # Save file temporarily
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                file.save(temp_file.name)
                temp_path = temp_file.name
            
            try:
                # Upload to Supabase storage
                file_type = file_data.get('type', 'input')
                storage_success = save_csv_file_content(
                    file_uuid, session_id, file.filename, temp_path, file_type
                )
                
                if not storage_success:
                    logger.warning(f"Failed to upload file to storage for file {file_uuid}")
                    
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        return jsonify({
            'success': True,
            'data': {
                'id': file_uuid,
                'message': 'CSV file created successfully'
            }
        })
        
    except Exception as e:
        logger.error(f"Error creating CSV file: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/csv-files/<file_id>', methods=['PUT'])
def update_csv_file(file_id):
    """Update CSV file metadata."""
    try:
        from utils.database import get_supabase_client
        supabase = get_supabase_client()
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Validate file_id is UUID
        try:
            uuid.UUID(file_id)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'Invalid file ID format'}), 400
        
        # Prepare update data (only allow specific fields to be updated)
        allowed_fields = [
            'file_name', 'bezeichnung', 'min', 'max', 'offset', 'datenpunkte',
            'numerische_datenpunkte', 'numerischer_anteil', 'datenform', 
            'datenanpassung', 'zeitschrittweite', 'zeitschrittweite_mittelwert',
            'zeitschrittweite_min', 'skalierung', 'skalierung_max', 'skalierung_min',
            'zeithorizont_start', 'zeithorizont_end', 'type'
        ]
        
        update_data = {}
        for field in allowed_fields:
            if field in data:
                # Convert numbers to strings for database storage
                if field in ['min', 'max', 'offset', 'datenpunkte', 'numerische_datenpunkte', 
                           'numerischer_anteil', 'zeitschrittweite', 'zeitschrittweite_mittelwert',
                           'zeitschrittweite_min', 'skalierung_max', 'skalierung_min']:
                    update_data[field] = str(data[field])
                else:
                    update_data[field] = data[field]
        
        if not update_data:
            return jsonify({'success': False, 'error': 'No valid fields to update'}), 400
        
        # Update file in database
        response = supabase.table('files').update(update_data).eq('id', file_id).execute()
        
        if response.data:
            return jsonify({
                'success': True,
                'data': response.data[0]
            })
        else:
            return jsonify({
                'success': False,
                'error': 'File not found or update failed'
            }), 404
            
    except Exception as e:
        logger.error(f"Error updating CSV file {file_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/csv-files/<file_id>', methods=['DELETE'])
def delete_csv_file(file_id):
    """Delete CSV file from database and storage."""
    try:
        from utils.database import get_supabase_client
        supabase = get_supabase_client()
        
        # Validate file_id is UUID
        try:
            uuid.UUID(file_id)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'Invalid file ID format'}), 400
        
        # Get file info first to know storage path and type
        file_response = supabase.table('files').select('*').eq('id', file_id).execute()
        if not file_response.data or len(file_response.data) == 0:
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404

        file_info = file_response.data[0]
        storage_path = file_info.get('storage_path', '')
        file_type = file_info.get('type', 'input')
        
        # Delete from storage if storage_path exists
        if storage_path:
            try:
                bucket_name = 'aus-csv-files' if file_type == 'output' else 'csv-files'
                storage_response = supabase.storage.from_(bucket_name).remove([storage_path])
                logger.info(f"Deleted file from storage: {bucket_name}/{storage_path}")
            except Exception as storage_error:
                logger.warning(f"Could not delete file from storage: {str(storage_error)}")
                # Continue with database deletion even if storage deletion fails
        
        # Delete from database
        db_response = supabase.table('files').delete().eq('id', file_id).execute()
        
        if db_response.data:
            return jsonify({
                'success': True,
                'message': 'CSV file deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to delete file from database'
            }), 500
            
    except Exception as e:
        logger.error(f"Error deleting CSV file {file_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/results/<session_id>', methods=['GET'])
def get_training_results(session_id):
    """
    Get training results for a session.
    Checks both UUID and string session IDs.
    """
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        supabase = get_supabase_client()
        
        # Get or create UUID for session ID
        uuid_session_id = create_or_get_session_uuid(session_id)
        
        # logger.info(f"Getting training results for session {session_id} (UUID: {uuid_session_id})")
        
        # Get training results from database (limit to 1 most recent result)
        response = supabase.table('training_results').select('*').eq('session_id', uuid_session_id).order('created_at.desc').limit(1).execute()

        if response.data and len(response.data) > 0:
            return jsonify({
                'success': True,
                'results': response.data,
                'count': len(response.data)
            })
        else:
            # No results found - this is normal if training hasn't been run yet
            # logger.info(f"No training results found for session {session_id}")
            return jsonify({
                'success': True,
                'message': 'No training results yet - training may not have been started',
                'results': [],
                'count': 0
            }), 200
            
    except Exception as e:
        logger.error(f"Error getting training results for {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/get-training-results/<session_id>', methods=['GET'])
def get_training_results_details(session_id):
    """
    Get detailed training results for a session (used by frontend)
    This is an alias for /results/<session_id> to maintain compatibility
    """
    return get_training_results(session_id)


@bp.route('/plot-variables/<session_id>', methods=['GET'])
def get_plot_variables(session_id):
    """
    Get available input and output variables for plotting
    """
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)

        # Get training results
        response = supabase.table('training_results').select('results').eq('session_id', uuid_session_id).order('created_at.desc').limit(1).execute()

        input_variables = []
        output_variables = []

        if response.data and len(response.data) > 0:
            results = response.data[0].get('results', {})

            # Try different places where variable names might be stored
            input_variables = (
                results.get('input_features') or
                results.get('input_columns') or
                results.get('data_info', {}).get('input_columns', [])
            )
            output_variables = (
                results.get('output_features') or
                results.get('output_columns') or
                results.get('data_info', {}).get('output_columns', [])
            )

        # If no variables found, try to get from files
        if not input_variables and not output_variables:
            file_response = supabase.table('files').select('*').eq('session_id', uuid_session_id).execute()

            if file_response.data:
                for file_data in file_response.data:
                    file_type = file_data.get('file_type', '')
                    columns = file_data.get('columns', [])

                    if file_type == 'input' and not input_variables:
                        input_variables = [col for col in columns if col.lower() not in ['timestamp', 'utc', 'zeit', 'datetime']]
                    elif file_type == 'output' and not output_variables:
                        output_variables = [col for col in columns if col.lower() not in ['timestamp', 'utc', 'zeit', 'datetime']]

        # Default fallback
        if not input_variables:
            input_variables = ['Temperature', 'Load']
        if not output_variables:
            output_variables = ['Predicted_Load']

        return jsonify({
            'success': True,
            'session_id': session_id,
            'input_variables': input_variables if isinstance(input_variables, list) else [],
            'output_variables': output_variables if isinstance(output_variables, list) else []
        })

    except Exception as e:
        logger.error(f"Error getting plot variables for {session_id}: {str(e)}")
        return jsonify({
            'success': True,
            'session_id': session_id,
            'input_variables': ['Temperature', 'Load'],
            'output_variables': ['Predicted_Load']
        })


@bp.route('/visualizations/<session_id>', methods=['GET'])
def get_training_visualizations(session_id):
    """
    Get training visualizations (violin plots) for a session
    """
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)

        # Get visualizations from database
        response = supabase.table('training_visualizations').select('*').eq('session_id', uuid_session_id).execute()

        if not response.data or len(response.data) == 0:
            return jsonify({
                'session_id': session_id,
                'plots': {},
                'message': 'No visualizations found for this session'
            }), 404

        # Organize plots by plot_name
        plots = {}
        metadata = {}
        created_at = None

        for viz in response.data:
            plot_name = viz.get('plot_name', 'unknown')
            plots[plot_name] = viz.get('image_data', '')

            if not metadata:
                metadata = viz.get('plot_metadata', {})
                created_at = viz.get('created_at')

        return jsonify({
            'session_id': session_id,
            'plots': plots,
            'metadata': metadata,
            'created_at': created_at,
            'message': 'Visualizations retrieved successfully'
        })

    except Exception as e:
        logger.error(f"Error retrieving visualizations for {session_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve training visualizations',
            'message': str(e)
        }), 500


@bp.route('/generate-plot', methods=['POST'])
def generate_plot():
    """
    Generate plot based on user selections matching original training_original.py

    Expected request body:
    {
        'session_id': str,
        'plot_settings': {
            'num_sbpl': int,
            'x_sbpl': str ('UTC' or 'ts'),
            'y_sbpl_fmt': str ('original' or 'skaliert'),
            'y_sbpl_set': str ('gemeinsame Achse' or 'separate Achsen')
        },
        'df_plot_in': dict,  # {feature_name: bool, ...}
        'df_plot_out': dict,  # {feature_name: bool, ...}
        'df_plot_fcst': dict  # {feature_name: bool, ...}
    }
    """
    try:
        data = request.json
        session_id = data.get('session_id')

        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Session ID is required'
            }), 400

        # Get plot settings
        plot_settings = data.get('plot_settings', {})
        num_sbpl = plot_settings.get('num_sbpl', 17)
        x_sbpl = plot_settings.get('x_sbpl', 'UTC')
        y_sbpl_fmt = plot_settings.get('y_sbpl_fmt', 'original')
        y_sbpl_set = plot_settings.get('y_sbpl_set', 'separate Achsen')

        # Get plot selections
        df_plot_in = data.get('df_plot_in', {})
        df_plot_out = data.get('df_plot_out', {})
        df_plot_fcst = data.get('df_plot_fcst', {})

        # Optional: specific model_id to use for plot generation
        model_id = data.get('model_id')  # If not provided, uses most recent model

        logger.info(f"Generate plot for session {session_id}" + (f" with model_id {model_id}" if model_id else " (using most recent model)"))
        logger.info(f"Plot settings: num_sbpl={num_sbpl}, x_sbpl={x_sbpl}, y_sbpl_fmt={y_sbpl_fmt}, y_sbpl_set={y_sbpl_set}")
        logger.info(f"Input variables selected: {[k for k, v in df_plot_in.items() if v]}")
        logger.info(f"Output variables selected: {[k for k, v in df_plot_out.items() if v]}")
        logger.info(f"Forecast variables selected: {[k for k, v in df_plot_fcst.items() if v]}")

        # Load model data from database
        try:
            from utils.database import create_or_get_session_uuid, get_supabase_client
            import numpy as np
            import pandas as pd

            # Get UUID for session
            uuid_session_id = create_or_get_session_uuid(session_id)
            if not uuid_session_id:
                return jsonify({
                    'success': False,
                    'error': 'Session not found'
                }), 404

            # Get Supabase client
            supabase = get_supabase_client()

            # Fetch training results from database
            # If model_id is provided, fetch that specific model, otherwise get most recent
            query = supabase.table('training_results').select('results').eq('session_id', uuid_session_id)

            if model_id:
                # Fetch specific model by ID
                query = query.eq('id', model_id)
            else:
                # Fetch most recent model
                query = query.order('created_at', desc=True).limit(1)

            response = query.execute()

            if not response.data or len(response.data) == 0:
                return jsonify({
                    'success': False,
                    'error': 'Model not trained yet. Please train the model first.'
                }), 400

            # Extract model data from results
            results = response.data[0]['results']

            # Get model data from results and deserialize if needed
            model_data = results.get('trained_model')

            # Deserialize model if it's in serialized format
            trained_model = None
            if model_data:
                # Handle list format (model_data is a list with one dict element)
                if isinstance(model_data, list) and len(model_data) > 0:
                    model_data = model_data[0]

                if isinstance(model_data, dict) and model_data.get('_model_type') == 'serialized_model':
                    try:
                        import pickle
                        import base64
                        # Decode base64 and deserialize model
                        model_bytes = base64.b64decode(model_data['_model_data'])
                        trained_model = pickle.loads(model_bytes)
                        logger.info(f"Successfully deserialized model of type {model_data.get('_model_class')}")
                    except Exception as e:
                        logger.error(f"Failed to deserialize model: {e}")
                        trained_model = None
                else:
                    # Legacy format - model stored as string or other format
                    trained_model = model_data

            test_data = results.get('test_data', {})
            metadata = results.get('metadata', {})
            scalers = results.get('scalers', {})

            # Check if we have test data - support both naming conventions
            has_test_data = False
            if test_data:
                # Check for both possible naming conventions
                if 'X' in test_data and 'y' in test_data:
                    # New format: X, y
                    tst_x = np.array(test_data.get('X'))
                    tst_y = np.array(test_data.get('y'))
                    has_test_data = True
                elif 'X_test' in test_data and 'y_test' in test_data:
                    # Old format: X_test, y_test
                    tst_x = np.array(test_data.get('X_test'))
                    tst_y = np.array(test_data.get('y_test'))
                    has_test_data = True

            if not has_test_data:
                logger.error(f"No test data found in database for session {session_id}")
                return jsonify({
                    'success': False,
                    'error': 'No training data available for this session',
                    'message': 'Please train a model first before generating plots'
                }), 400
            else:

                # Generate predictions using trained model if available
                if trained_model and hasattr(trained_model, 'predict'):
                    # Get model type to handle SVR models differently
                    model_type = results.get('model_type', 'Unknown')

                    # SVR models need special handling (2D input instead of 3D)
                    if model_type in ['SVR_dir', 'SVR_MIMO', 'LIN']:
                        logger.info(f"Using SVR/LIN prediction logic for model type: {model_type}")
                        # For SVR models, we need to predict on 2D data
                        n_samples = tst_x.shape[0]
                        n_outputs = tst_y.shape[-1] if len(tst_y.shape) > 2 else 1

                        # Initialize predictions array
                        tst_fcst = []

                        # Predict for each sample
                        for i in range(n_samples):
                            # Squeeze to 2D (timesteps, features)
                            inp = np.squeeze(tst_x[i:i+1], axis=0)

                            # trained_model is a list of models for SVR_dir/SVR_MIMO
                            if isinstance(trained_model, list):
                                pred = []
                                for model_i in trained_model:
                                    pred.append(model_i.predict(inp))
                                out = np.array(pred).T  # Transpose to (timesteps, outputs)
                            else:
                                # For LIN or single model
                                out = trained_model.predict(inp)

                            # Expand dims back to 3D
                            out = np.expand_dims(out, axis=0)
                            tst_fcst.append(out[0])

                        tst_fcst = np.array(tst_fcst)
                    else:
                        # For Dense, CNN, LSTM, AR LSTM - direct prediction
                        logger.info(f"Using standard prediction logic for model type: {model_type}")
                        tst_fcst = trained_model.predict(tst_x)

                        # CNN models need to squeeze the last dimension
                        if model_type == 'CNN':
                            logger.info(f"Squeezing last dimension for CNN model, shape before: {tst_fcst.shape}")
                            tst_fcst = np.squeeze(tst_fcst, axis=-1)
                            logger.info(f"Shape after squeeze: {tst_fcst.shape}")
                else:
                    # Model is not available as object, cannot generate predictions
                    logger.error(f"Model is not available as an object for session {session_id} (stored as: {type(trained_model).__name__})")
                    return jsonify({
                        'success': False,
                        'error': 'Model not available for predictions',
                        'message': 'The trained model cannot be used for predictions. Please retrain the model.'
                    }), 400

            # Import matplotlib for plotting
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            from datetime import datetime
            import io
            import base64

            # Create figure based on settings
            num_sbpl = min(num_sbpl, len(tst_x))  # Limit to available test samples
            num_sbpl_x = int(np.ceil(np.sqrt(num_sbpl)))
            num_sbpl_y = int(np.ceil(num_sbpl / num_sbpl_x))

            fig, axs = plt.subplots(num_sbpl_y, num_sbpl_x,
                                   figsize=(20, 13),
                                   layout='constrained')

            # Flatten axs array for easier indexing
            if num_sbpl == 1:
                axs = [axs]
            else:
                axs = axs.flatten()

            # Color palette - make sure we have enough colors
            # Count total number of variables to plot
            total_vars = len([k for k, v in df_plot_in.items() if v]) + \
                        len([k for k, v in df_plot_out.items() if v]) + \
                        len([k for k, v in df_plot_fcst.items() if v])
            palette = sns.color_palette("tab20", max(20, total_vars))

            # Plot each subplot
            for i_sbpl in range(num_sbpl):
                ax = axs[i_sbpl] if num_sbpl > 1 else axs[0]

                # Create x-axis values
                if x_sbpl == 'UTC':
                    # Create UTC timestamps
                    x_values = pd.date_range(start='2024-01-01',
                                            periods=tst_x.shape[1],
                                            freq='1h')
                else:
                    # Use timestep indices
                    x_values = np.arange(tst_x.shape[1])

                # Plot selected input variables
                color_idx = 0
                for var_name, selected in df_plot_in.items():
                    if selected and color_idx < tst_x.shape[-1]:
                        if y_sbpl_fmt == 'original':
                            y_values = tst_x[i_sbpl, :, color_idx]
                        else:
                            y_values = tst_x[i_sbpl, :, color_idx]  # Already scaled

                        ax.plot(x_values, y_values,
                               label=f'IN: {var_name}',
                               color=palette[color_idx % len(palette)],
                               marker='o', markersize=2,
                               linewidth=1)
                        color_idx += 1

                # Plot selected output variables (ground truth)
                for i_out, (var_name, selected) in enumerate(df_plot_out.items()):
                    if selected and i_out < tst_y.shape[-1]:
                        if y_sbpl_fmt == 'original':
                            y_values = tst_y[i_sbpl, :, i_out]
                        else:
                            y_values = tst_y[i_sbpl, :, i_out]  # Already scaled

                        ax.plot(x_values[:len(y_values)], y_values,
                               label=f'OUT: {var_name}',
                               color=palette[color_idx % len(palette)],
                               marker='s', markersize=2,
                               linewidth=1)
                        color_idx += 1

                # Plot selected forecast variables (predictions)
                for i_fcst, (var_name, selected) in enumerate(df_plot_fcst.items()):
                    if selected and i_fcst < tst_fcst.shape[-1] if len(tst_fcst.shape) > 2 else 1:
                        if len(tst_fcst.shape) == 3:
                            y_values = tst_fcst[i_sbpl, :, i_fcst]
                        elif len(tst_fcst.shape) == 2:
                            y_values = tst_fcst[i_sbpl, :]
                        else:
                            y_values = tst_fcst

                        ax.plot(x_values[:len(y_values)], y_values,
                               label=f'FCST: {var_name}',
                               color=palette[color_idx % len(palette)],
                               marker='^', markersize=2,
                               linewidth=1, linestyle='--')
                        color_idx += 1

                # Configure subplot
                ax.set_title(f'Sample {i_sbpl + 1}', fontsize=10)
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3)

                if x_sbpl == 'UTC':
                    ax.set_xlabel('Time (UTC)', fontsize=9)
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                else:
                    ax.set_xlabel('Timestep', fontsize=9)

                ax.set_ylabel('Value', fontsize=9)

                # Handle y-axis configuration
                if y_sbpl_set == 'separate Achsen':
                    # Each line gets its own y-axis scale
                    pass  # Already handled by matplotlib auto-scaling

            # Remove empty subplots
            for i in range(num_sbpl, len(axs)):
                fig.delaxes(axs[i])

            # Save plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            logger.info(f"Plot generated successfully for session {session_id}")

            return jsonify({
                'success': True,
                'session_id': session_id,
                'plot_data': f'data:image/png;base64,{plot_data}',
                'message': 'Plot generated successfully'
            })

        except Exception as e:
            logger.error(f"Error generating plot: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Failed to generate plot: {str(e)}'
            }), 500

    except Exception as e:
        logger.error(f"Error in generate_plot endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to process plot request',
            'message': str(e)
        }), 500


@bp.route('/status/<session_id>', methods=['GET'])
@bp.route('/session-status/<session_id>', methods=['GET'])  # Alias for frontend compatibility
def get_training_status(session_id: str):
    """
    Get training status for a session

    Args:
        session_id: Session identifier

    Returns:
        JSON response with training status
    """
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)

        # Check training_results table first
        results_response = supabase.table('training_results').select('*').eq('session_id', uuid_session_id).order('created_at.desc').limit(1).execute()

        # Check training_logs table for detailed progress
        logs_response = supabase.table('training_logs').select('*').eq('session_id', uuid_session_id).order('created_at.desc').limit(1).execute()

        if results_response.data and len(results_response.data) > 0:
            # Training is completed
            result_data = results_response.data[0]
            status = {
                'session_id': session_id,
                'status': result_data.get('status', 'completed'),
                'progress': 100,
                'current_step': 'Training completed',
                'total_steps': 7,
                'completed_steps': 7,
                'started_at': result_data.get('created_at'),
                'completed_at': result_data.get('completed_at'),
                'message': 'Training completed successfully'
            }
        elif logs_response.data and len(logs_response.data) > 0:
            # Training is in progress
            log_data = logs_response.data[0]
            progress_data = log_data.get('progress', {})
            status = {
                'session_id': session_id,
                'status': 'in_progress',
                'progress': progress_data.get('overall', 0) if isinstance(progress_data, dict) else 0,
                'current_step': progress_data.get('current_step', 'Processing') if isinstance(progress_data, dict) else 'Processing',
                'total_steps': progress_data.get('total_steps', 7) if isinstance(progress_data, dict) else 7,
                'completed_steps': progress_data.get('completed_steps', 0) if isinstance(progress_data, dict) else 0,
                'started_at': log_data.get('created_at'),
                'completed_at': None,
                'message': 'Training in progress'
            }
        else:
            # No training found
            status = {
                'session_id': session_id,
                'status': 'not_found',
                'progress': 0,
                'current_step': 'Not started',
                'total_steps': 7,
                'completed_steps': 0,
                'started_at': None,
                'completed_at': None,
                'message': 'No training found for this session'
            }

        return jsonify(status)

    except Exception as e:
        logger.error(f"Error getting training status for {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get training status',
            'message': str(e),
            'session_id': session_id,
            'status': 'error'
        }), 500


@bp.route('/generate-datasets/<session_id>', methods=['POST'])
def generate_datasets(session_id):
    """
    Generate datasets and violin plots WITHOUT training models.
    This is phase 1 of the training workflow - data visualization only.
    Following the original implementation approach.
    """
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Extract model parameters and training split for data preparation
        model_parameters = data.get('model_parameters', {})
        training_split = data.get('training_split', {})
        
        logger.info(f"Generating violin plots for session {session_id} WITHOUT training")
        
        # Import the violin plot generator
        import numpy as np
        from services.training.violin_plot_generator import generate_violin_plots_from_data
        
        # Load data directly from uploaded CSV files (following original training_original.py approach)
        try:
            logger.info(f"Loading raw CSV data for session {session_id} (original approach)")

            from services.training.data_loader import DataLoader
            data_loader = DataLoader()

            # Check if files exist in database
            session_data = data_loader.load_session_data(session_id)
            files_info = session_data.get('files', [])

            if not files_info:
                return jsonify({
                    'success': False,
                    'error': 'No data available for visualization',
                    'message': 'Please upload CSV files first'
                }), 400

            # Download and read CSV files
            downloaded_files = data_loader.download_session_files(session_id)

            # Load CSV data for visualization with separator detection
            import os
            csv_data = {}
            for file_type, file_path in downloaded_files.items():
                if os.path.exists(file_path):
                    # Try different separators (following original training_original.py approach)
                    try:
                        df = pd.read_csv(file_path, sep=';')  # Try semicolon first (original script uses this)
                        if df.shape[1] == 1:  # If still one column, try comma
                            df = pd.read_csv(file_path, sep=',')
                    except:
                        df = pd.read_csv(file_path)  # Fallback to default

                    csv_data[file_type] = df
                    logger.info(f"Loaded {file_type} file with {len(df)} rows and {len(df.columns)} columns: {list(df.columns)}")

            if not csv_data:
                return jsonify({
                    'success': False,
                    'error': 'Could not load CSV data',
                    'message': 'CSV files could not be read'
                }), 400

            # Create data_info structure from CSV files (following original script approach)
            input_data = None
            output_data = None
            input_features = []
            output_features = []

            # Process input files (assume they have numeric data)
            if 'input' in csv_data:
                input_df = csv_data['input']
                # Get numeric columns only (skip UTC/timestamp columns)
                numeric_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    input_data = input_df[numeric_cols].values
                    input_features = numeric_cols

            # Process output files
            if 'output' in csv_data:
                output_df = csv_data['output']
                numeric_cols = output_df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    output_data = output_df[numeric_cols].values
                    output_features = numeric_cols

            if input_data is None and output_data is None:
                return jsonify({
                    'success': False,
                    'error': 'No numeric data found in CSV files',
                    'message': 'CSV files must contain numeric columns for visualization'
                }), 400

            data_info = {
                'success': True,
                'input_data': input_data,
                'output_data': output_data,
                'input_features': input_features,
                'output_features': output_features
            }

            # Generate plots from CSV data
            plot_result = generate_violin_plots_from_data(
                session_id,
                input_data=data_info.get('input_data'),
                output_data=data_info.get('output_data'),
                input_features=data_info.get('input_features'),
                output_features=data_info.get('output_features')
            )

        except Exception as e:
            logger.error(f"Error loading CSV data for violin plots: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Failed to load CSV data: {str(e)}',
                'message': 'Could not read uploaded files'
            }), 500
        
        # Return plot results without training
        result = {
            'success': plot_result['success'],
            'violin_plots': plot_result.get('plots', {}),
            'message': 'Violin plots generated successfully (no training performed)'
        }
        
        if result['success']:
            # logger.info(f"Dataset generation completed for session {session_id}")
            
            # Save visualizations to database
            violin_plots = result.get('violin_plots', {})
            if violin_plots:
                from services.training.training_api import save_visualization_to_database
                
                for plot_name, plot_data in violin_plots.items():
                    try:
                        if plot_data:  # Only save if data exists
                            save_visualization_to_database(session_id, plot_name, plot_data)
                            # logger.info(f"Saved visualization {plot_name} for session {session_id}")
                    except Exception as viz_error:
                        logger.error(f"Failed to save visualization {plot_name}: {str(viz_error)}")
                        # Continue even if one visualization fails to save
            
            return jsonify({
                'success': True,
                'message': 'Datasets generated successfully',
                'dataset_count': result.get('dataset_count', 10),
                'violin_plots': violin_plots
            })
        else:
            logger.error(f"Dataset generation failed: {result.get('error')}")
            return jsonify({
                'success': False,
                'error': result.get('error', 'Dataset generation failed')
            }), 500
            
    except Exception as e:
        logger.error(f"Error in generate_datasets: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/train-models/<session_id>', methods=['POST'])
def train_models(session_id):
    """
    Train models with user-specified parameters.
    This is phase 2 of the training workflow.
    """
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Extract model parameters
        model_parameters = data.get('model_parameters', {})
        training_split = data.get('training_split', {})

        # logger.info(f"Training models for session {session_id}")
        # logger.info(f"Model parameters: {model_parameters}")

        import threading
        from services.training.middleman_runner import ModernMiddlemanRunner
        from flask import current_app

        # Get SocketIO instance
        socketio_instance = current_app.extensions.get('socketio')

        # Create runner
        runner = ModernMiddlemanRunner()
        if socketio_instance:
            runner.set_socketio(socketio_instance)

        # Normalize activation function to lowercase for backend compatibility
        # Frontend sends: ReLU, Tanh, Sigmoid, etc.
        # Backend expects: relu, tanh, sigmoid, etc.
        actf = model_parameters.get('ACTF')
        if actf:
            actf = actf.lower()

        # Prepare full model configuration
        model_config = {
            'MODE': model_parameters.get('MODE', 'Linear'),
            'LAY': model_parameters.get('LAY'),
            'N': model_parameters.get('N'),
            'EP': model_parameters.get('EP'),
            'ACTF': actf,
            'K': model_parameters.get('K'),
            'KERNEL': model_parameters.get('KERNEL'),
            'C': model_parameters.get('C'),
            'EPSILON': model_parameters.get('EPSILON'),
            'random_dat': not training_split.get('shuffle', True)
        }
        
        # Run training in background thread
        def run_training_async():
            try:
                # logger.info(f"Starting async training for session {session_id}")
                result = runner.run_training_script(session_id, model_config)
                
                if result['success']:
                    # logger.info(f"Training completed successfully for session {session_id}")
                    
                    # Save training results to database
                    try:
                        from utils.database import get_supabase_client, create_or_get_session_uuid
                        import json
                        import numpy as np
                        import pandas as pd
                        from datetime import datetime
                        
                        supabase = get_supabase_client()
                        uuid_session_id = create_or_get_session_uuid(session_id)
                        
                        # Check if we got a valid UUID
                        if not uuid_session_id:
                            logger.error(f"Failed to get UUID for session {session_id}")
                            # Try once more after a short delay
                            import time
                            time.sleep(1)
                            uuid_session_id = create_or_get_session_uuid(session_id)
                            
                            if not uuid_session_id:
                                logger.error(f"Failed to get UUID for session {session_id} after retry")
                                return jsonify({
                                    'success': False,
                                    'error': 'Failed to save training results - session mapping error',
                                    'message': 'Please try training again'
                                }), 500
                        
                        # Extract and clean results from training
                        training_results = result.get('results', {})
                        
                        # Helper function to clean non-serializable objects
                        def clean_for_json(obj):
                            # Check for custom MDL class
                            if hasattr(obj, '__class__') and obj.__class__.__name__ == 'MDL':
                                # Convert MDL object to dict with its attributes
                                return {
                                    'MODE': getattr(obj, 'MODE', 'Dense'),
                                    'LAY': getattr(obj, 'LAY', None),
                                    'N': getattr(obj, 'N', None),
                                    'EP': getattr(obj, 'EP', None),
                                    'ACTF': getattr(obj, 'ACTF', None),
                                    'K': getattr(obj, 'K', None),
                                    'KERNEL': getattr(obj, 'KERNEL', None),
                                    'C': getattr(obj, 'C', None),
                                    'EPSILON': getattr(obj, 'EPSILON', None)
                                }
                            # Check numpy array first before other checks
                            elif isinstance(obj, np.ndarray):
                                return obj.tolist()
                            elif isinstance(obj, (pd.Timestamp, datetime)):
                                return obj.isoformat()
                            elif isinstance(obj, (np.int64, np.int32)):
                                return int(obj)
                            elif isinstance(obj, (np.float64, np.float32)):
                                return float(obj)
                            elif isinstance(obj, (np.integer)):
                                return int(obj)
                            elif isinstance(obj, (np.floating)):
                                return float(obj)
                            elif isinstance(obj, dict):
                                return {k: clean_for_json(v) for k, v in obj.items()}
                            elif isinstance(obj, (list, tuple)):
                                return [clean_for_json(item) for item in obj]
                            # Check for NaN after numpy types since pd.isna might fail on arrays
                            elif obj is None:
                                return None
                            else:
                                try:
                                    if pd.isna(obj):
                                        return None
                                except (ValueError, TypeError):
                                    pass
                                
                                # Special handling for ML models and scalers - serialize to base64
                                try:
                                    # Check if it's a sklearn scaler or transformer
                                    if (hasattr(obj, 'fit') and hasattr(obj, 'transform')) or \
                                       (hasattr(obj, 'predict') and hasattr(obj, 'fit')) or \
                                       (hasattr(obj, '__class__') and 'sklearn' in str(obj.__class__.__module__)):
                                        import pickle
                                        import base64
                                        # Serialize model/scaler to bytes
                                        model_bytes = pickle.dumps(obj)
                                        # Encode to base64 for JSON storage
                                        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
                                        return {
                                            '_model_type': 'serialized_model',
                                            '_model_class': obj.__class__.__name__,
                                            '_model_data': model_b64
                                        }
                                except Exception as e:
                                    logger.warning(f"Could not serialize model/scaler: {e}")
                                    pass
                                
                                # Try to convert unknown objects to string as last resort
                                try:
                                    import json
                                    json.dumps(obj)
                                    return obj
                                except (TypeError, ValueError):
                                    return str(obj)
                        
                        # Clean the results for JSON serialization
                        # Get metrics from multiple possible locations
                        evaluation_metrics = result.get('evaluation_metrics', {})
                        if not evaluation_metrics:
                            evaluation_metrics = training_results.get('evaluation_metrics', {})
                        if not evaluation_metrics:
                            evaluation_metrics = training_results.get('metrics', {})
                        
                        # IMPORTANT: Save ALL training data for plotting
                        # This includes the trained model, test data, scalers, etc.
                        cleaned_results = clean_for_json({
                            'model_type': model_config.get('MODE', 'Dense'),
                            'parameters': model_config,
                            'metrics': evaluation_metrics,
                            'training_split': training_split,
                            'dataset_count': result.get('dataset_count', 0),
                            'evaluation_metrics': evaluation_metrics,
                            'metadata': training_results.get('metadata', {}),
                            # Add full training results for plotting
                            'trained_model': training_results.get('trained_model'),
                            'train_data': training_results.get('train_data', {}),
                            'val_data': training_results.get('val_data', {}),
                            'test_data': training_results.get('test_data', {}),
                            'scalers': training_results.get('scalers', {}),
                            'input_features': training_results.get('metadata', {}).get('input_features', []),
                            'output_features': training_results.get('metadata', {}).get('output_features', [])
                        })
                        
                        # Prepare results data for database
                        # The training_results table expects: session_id, results, created_at, status
                        training_data = {
                            'session_id': uuid_session_id,
                            'results': cleaned_results,
                            'status': 'completed'
                        }
                        
                        supabase.table('training_results').insert(training_data).execute()
                        logger.info(f"Training results saved for session {uuid_session_id}")
                        
                        # Save violin plots if available
                        if result.get('violin_plots'):
                            try:
                                # Check if violin_plots is a dictionary
                                violin_plots = result.get('violin_plots')
                                if isinstance(violin_plots, dict):
                                    # Save each violin plot separately
                                    for plot_name, plot_data in violin_plots.items():
                                        # Extract just the base64 part if it includes data URI prefix
                                        if isinstance(plot_data, str) and plot_data.startswith('data:image'):
                                            # Keep the full data URI for display
                                            base64_data = plot_data
                                        else:
                                            base64_data = plot_data
                                        
                                        viz_data = {
                                            'session_id': uuid_session_id,
                                            'plot_type': 'violin',
                                            'plot_name': plot_name,
                                            'dataset_name': plot_name.replace('_distribution', '').replace('_plot', ''),
                                            'model_name': model_config.get('MODE', 'Linear'),
                                            'plot_data_base64': base64_data,  # The base64 encoded image with data URI
                                            'metadata': {
                                                'dataset_count': result.get('dataset_count', 0),
                                                'generated_during': 'model_training',
                                                'created_at': datetime.now().isoformat()
                                            }
                                        }
                                        supabase.table('training_visualizations').insert(viz_data).execute()
                                    logger.info(f"Violin plots saved for session {uuid_session_id}")
                                else:
                                    logger.warning(f"Violin plots not in expected format: {type(violin_plots)}")
                            except Exception as viz_error:
                                logger.error(f"Failed to save violin plots: {str(viz_error)}")
                                import traceback
                                logger.error(traceback.format_exc())
                        
                    except Exception as e:
                        logger.error(f"Failed to save training results: {str(e)}")
                    
                    if socketio_instance:
                        # Don't send full results as they may contain non-serializable objects
                        # Only send essential information
                        socketio_instance.emit('training_completed', {
                            'session_id': session_id,
                            'status': 'completed',
                            'message': 'Training completed successfully'
                        }, room=session_id)
                else:
                    logger.error(f"Training failed: {result.get('error')}")
                    if socketio_instance:
                        socketio_instance.emit('training_error', {
                            'session_id': session_id,
                            'status': 'failed',
                            'error': result.get('error', 'Unknown error')
                        }, room=session_id)
                        
            except Exception as e:
                logger.error(f"Async training error: {str(e)}")
                if socketio_instance:
                    socketio_instance.emit('training_error', {
                        'session_id': session_id,
                        'status': 'failed',
                        'error': str(e)
                    }, room=session_id)
        
        # Start training in background
        training_thread = threading.Thread(target=run_training_async)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Model training started for session {session_id}',
            'note': 'Training is running in background, listen for SocketIO events for progress'
        })
        
    except Exception as e:
        logger.error(f"Error in train_models: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def get_evaluation_tables(session_id):
    """
    Get evaluation metrics formatted as tables for display.
    Returns df_eval and df_eval_ts structures matching the original training output.
    """
    try:
        # Get UUID for session
        from utils.database import get_supabase_client, create_or_get_session_uuid
        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)
        
        # Get latest training results
        response = supabase.table('training_results') \
            .select('results') \
            .eq('session_id', uuid_session_id) \
            .order('created_at.desc') \
            .limit(1) \
            .execute()
        
        if not response.data:
            return jsonify({
                'success': False,
                'error': 'No training results found for this session',
                'session_id': session_id
            }), 404
        
        results = response.data[0]['results']
        
        # Check for evaluation metrics
        eval_metrics = results.get('evaluation_metrics', {})
        if not eval_metrics or eval_metrics.get('error'):
            # Try alternative location
            eval_metrics = results.get('metrics', {})
        
        # Debug: Log what we found
        logger.info(f"Found eval_metrics keys: {eval_metrics.keys() if eval_metrics else 'None'}")
        
        # If we have metrics but they contain an error, still try to use them if test/val metrics exist
        if eval_metrics and eval_metrics.get('test_metrics_scaled'):
            # We have some metrics, proceed
            pass
        elif not eval_metrics or (eval_metrics.get('error') and not eval_metrics.get('test_metrics_scaled')):
            return jsonify({
                'success': False,
                'error': f"No valid evaluation metrics found. Metrics: {eval_metrics}",
                'session_id': session_id
            }), 404
        
        # Format metrics to match original training_original.py structure exactly
        # The original uses nested dictionaries with output feature names as keys
        
        # Get output feature names (default to "Netzlast [kW]" if not available)
        output_features = results.get('output_features', ['Netzlast [kW]'])
        if not output_features:
            output_features = ['Netzlast [kW]']  # Default from original
        
        # df_eval - Dictionary with output features as keys
        # Each contains a DataFrame with metrics for different time deltas
        df_eval = {}
        
        # df_eval_ts - Dictionary with output features as keys
        # Each contains another dict with time deltas as keys
        df_eval_ts = {}
        
        # Time deltas in minutes (matching original n_max=12)
        time_deltas = [15, 30, 45, 60, 90, 120, 180, 240, 300, 360, 420, 480]
        
        # Process each output feature
        for feature_name in output_features:
            # Initialize lists for df_eval metrics
            delt_list = []
            mae_list = []
            mape_list = []
            mse_list = []
            rmse_list = []
            nrmse_list = []
            wape_list = []
            smape_list = []
            mase_list = []
            
            # Initialize df_eval_ts for this feature
            df_eval_ts[feature_name] = {}
            
            # Get test metrics if available
            test_metrics = eval_metrics.get('test_metrics_scaled', {})
            val_metrics = eval_metrics.get('val_metrics_scaled', {})
            
            # For each time delta, calculate or use available metrics
            for delta in time_deltas:
                # Add delta to list
                delt_list.append(float(delta))
                
                # For now, use the same test metrics for all deltas
                # In production, these would be calculated for each time averaging
                mae_list.append(float(test_metrics.get('MAE', 0.0)))
                mape_list.append(float(test_metrics.get('MAPE', 0.0)))
                mse_list.append(float(test_metrics.get('MSE', 0.0)))
                rmse_list.append(float(test_metrics.get('RMSE', 0.0)))
                nrmse_list.append(float(test_metrics.get('NRMSE', 0.0)))
                wape_list.append(float(test_metrics.get('WAPE', 0.0)))
                smape_list.append(float(test_metrics.get('sMAPE', 0.0)))
                mase_list.append(float(test_metrics.get('MASE', 0.0)))
                
                # Create df_eval_ts entry for this delta
                # This should contain per-timestep metrics
                timestep_metrics = []
                n_timesteps = results.get('n_timesteps', 96)  # Default 96 timesteps
                
                for ts in range(n_timesteps):
                    # In production, these would be actual per-timestep metrics
                    timestep_metrics.append({
                        'MAE': float(test_metrics.get('MAE', 0.0)),
                        'MAPE': float(test_metrics.get('MAPE', 0.0)),
                        'MSE': float(test_metrics.get('MSE', 0.0)),
                        'RMSE': float(test_metrics.get('RMSE', 0.0)),
                        'NRMSE': float(test_metrics.get('NRMSE', 0.0)),
                        'WAPE': float(test_metrics.get('WAPE', 0.0)),
                        'sMAPE': float(test_metrics.get('sMAPE', 0.0)),
                        'MASE': float(test_metrics.get('MASE', 0.0))
                    })
                
                df_eval_ts[feature_name][float(delta)] = timestep_metrics
            
            # Create DataFrame-like structure for df_eval
            df_eval[feature_name] = {
                "delta [min]": delt_list,
                "MAE": mae_list,
                "MAPE": mape_list,
                "MSE": mse_list,
                "RMSE": rmse_list,
                "NRMSE": nrmse_list,
                "WAPE": wape_list,
                "sMAPE": smape_list,
                "MASE": mase_list
            }
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'df_eval': df_eval,
            'df_eval_ts': df_eval_ts,
            'model_type': eval_metrics.get('model_type', 'Unknown')
        })
        
    except Exception as e:
        logger.error(f"Error getting evaluation tables: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'session_id': session_id
        }), 500


@bp.route('/delete-all-sessions', methods=['POST'])
def delete_all_sessions():
    """
    Delete ALL sessions and associated data from Supabase database and local storage.
    ⚠️ WARNING: This will permanently delete all data! Use with extreme caution.
    """
    try:
        from utils.database import get_supabase_client
        import os
        import shutil
        
        logger.warning("🚨 DELETE ALL SESSIONS operation initiated!")
        
        # Get request data for confirmation (handle both JSON and form data)
        data = {}
        try:
            # Try JSON first
            if request.is_json:
                data = request.get_json() or {}
            elif request.content_type and 'application/json' in request.content_type:
                data = request.get_json(force=True) or {}
            elif hasattr(request, 'json') and request.json:
                data = request.json
            else:
                # Try form data as fallback
                data = request.form.to_dict()
                # Convert string values to proper types
                if 'confirm_delete_all' in data:
                    data['confirm_delete_all'] = data['confirm_delete_all'].lower() in ['true', '1', 'yes']
        except Exception as parse_error:
            logger.warning(f"Could not parse request data: {str(parse_error)}")
            # Last resort - check raw data
            try:
                import json
                raw_data = request.get_data(as_text=True)
                if raw_data:
                    data = json.loads(raw_data)
            except:
                data = {}
        
        confirmation = data.get('confirm_delete_all', False)
        logger.info(f"Parsed confirmation: {confirmation} from data: {data}")
        
        # Require explicit confirmation
        if not confirmation:
            return jsonify({
                'success': False,
                'error': 'Confirmation required',
                'message': 'To delete all sessions, send {"confirm_delete_all": true} in request body'
            }), 400
        
        # Get Supabase client
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({
                'success': False,
                'error': 'Database connection not available'
            }), 500
            
        # Count current data before deletion
        initial_counts = {}
        database_errors = []
        
        try:
            # Get initial counts
            tables_to_check = [
                'training_results',
                'training_visualizations', 
                'files',
                'zeitschritte',
                'time_info',
                'session_mappings',
                'sessions'
            ]
            
            for table in tables_to_check:
                try:
                    count_response = supabase.table(table).select('id', count='exact').execute()
                    initial_counts[table] = count_response.count
                    logger.info(f"Found {count_response.count} records in {table}")
                except Exception as e:
                    logger.warning(f"Could not count records in {table}: {str(e)}")
                    initial_counts[table] = 'unknown'
            
        except Exception as e:
            logger.error(f"Error getting initial counts: {str(e)}")
        
        # 1. Delete all CSV files from Supabase Storage
        storage_deleted = {'csv-files': 0, 'aus-csv-files': 0}
        
        try:
            # Get all files to delete from storage
            files_response = supabase.table('files').select('storage_path, type').execute()
            
            for file_data in files_response.data:
                storage_path = file_data.get('storage_path')
                file_type = file_data.get('type', 'input')
                
                if storage_path:
                    try:
                        bucket_name = 'aus-csv-files' if file_type == 'output' else 'csv-files'
                        supabase.storage.from_(bucket_name).remove([storage_path])
                        storage_deleted[bucket_name] += 1
                        logger.info(f"Deleted from storage: {bucket_name}/{storage_path}")
                    except Exception as storage_error:
                        logger.warning(f"Could not delete from storage {storage_path}: {str(storage_error)}")
                        database_errors.append(f"Storage: {storage_path} - {str(storage_error)}")
                        
        except Exception as e:
            logger.error(f"Error deleting from storage: {str(e)}")
            database_errors.append(f"Storage deletion error: {str(e)}")
        
        # 2. Delete from database tables (order matters due to foreign keys)
        tables_to_delete = [
            'training_results',      # Results and visualizations first
            'training_visualizations',
            'evaluation_tables',     # May not exist
            'files',                 # File metadata
            'zeitschritte',          # Time step data
            'time_info',             # Time configuration
            'session_mappings',      # Session mapping
            'sessions'               # Session records last
        ]
        
        deleted_counts = {}
        
        for table in tables_to_delete:
            try:
                # Delete all records from table
                response = supabase.table(table).delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
                deleted_count = len(response.data) if response.data else 0
                deleted_counts[table] = deleted_count
                
                if deleted_count > 0:
                    logger.info(f"✅ Deleted {deleted_count} records from {table}")
                else:
                    logger.info(f"ℹ️  No records to delete from {table}")
                    
            except Exception as table_error:
                logger.error(f"Error deleting from {table}: {str(table_error)}")
                database_errors.append(f"Table {table}: {str(table_error)}")
                deleted_counts[table] = 'error'
        
        # 3. Delete all local session directories
        local_deleted = {'directories': 0, 'files': 0}
        local_errors = []
        
        try:
            upload_base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'api', 'routes', 'uploads', 'file_uploads')
            
            if os.path.exists(upload_base):
                # Get all session directories
                for item in os.listdir(upload_base):
                    item_path = os.path.join(upload_base, item)
                    
                    # Only delete directories that look like session directories
                    if os.path.isdir(item_path) and item.startswith('session_'):
                        try:
                            # Count files before deletion
                            file_count = sum([len(files) for r, d, files in os.walk(item_path)])
                            local_deleted['files'] += file_count
                            
                            # Delete directory and all contents
                            shutil.rmtree(item_path)
                            local_deleted['directories'] += 1
                            logger.info(f"🗂️  Deleted local directory: {item_path} ({file_count} files)")
                            
                        except Exception as dir_error:
                            logger.error(f"Error deleting directory {item_path}: {str(dir_error)}")
                            local_errors.append(f"Directory {item}: {str(dir_error)}")
            else:
                logger.info("📁 Local upload directory does not exist")
                
        except Exception as e:
            logger.error(f"Error during local cleanup: {str(e)}")
            local_errors.append(f"Local cleanup error: {str(e)}")
        
        # Prepare response
        all_errors = database_errors + local_errors
        
        # Calculate totals
        total_database_records = sum([count for count in deleted_counts.values() if isinstance(count, int)])
        total_storage_files = sum(storage_deleted.values())
        
        response_data = {
            'success': True,
            'message': 'All sessions deletion completed',
            'summary': {
                'database_records_deleted': total_database_records,
                'storage_files_deleted': total_storage_files, 
                'local_directories_deleted': local_deleted['directories'],
                'local_files_deleted': local_deleted['files']
            },
            'details': {
                'initial_counts': initial_counts,
                'deleted_counts': deleted_counts,
                'storage_deleted': storage_deleted,
                'local_deleted': local_deleted
            }
        }
        
        if all_errors:
            response_data['warnings'] = all_errors
            response_data['message'] += ' (with some warnings)'
            logger.warning(f"Completed with {len(all_errors)} warnings")
        else:
            logger.info("✅ All sessions deleted successfully without errors")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Critical error during delete all sessions: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Critical error occurred during deletion operation'
        }), 500


# ============================================================================
# EVALUATION TABLES API ENDPOINTS
# ============================================================================

@bp.route('/evaluation-tables/<session_id>', methods=['GET'])
def get_evaluation_tables(session_id):
    """
    Get evaluation metrics formatted as tables for display.
    Returns df_eval and df_eval_ts structures matching the original training output.
    """
    try:
        # Get UUID for session
        from utils.database import get_supabase_client, create_or_get_session_uuid
        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)

        # Get latest training results
        response = supabase.table('training_results') \
            .select('results') \
            .eq('session_id', uuid_session_id) \
            .order('created_at.desc') \
            .limit(1) \
            .execute()

        if not response.data:
            return jsonify({
                'success': False,
                'error': 'No training results found for this session',
                'session_id': session_id
            }), 404

        results = response.data[0]['results']

        # Check for evaluation metrics
        eval_metrics = results.get('evaluation_metrics', {})
        if not eval_metrics or eval_metrics.get('error'):
            # Try alternative location
            eval_metrics = results.get('metrics', {})

        # Debug: Log what we found
        logger.info(f"Found eval_metrics keys: {eval_metrics.keys() if eval_metrics else 'None'}")

        # If we have metrics but they contain an error, still try to use them if test/val metrics exist
        if eval_metrics and eval_metrics.get('test_metrics_scaled'):
            # We have some metrics, proceed
            pass
        elif not eval_metrics or (eval_metrics.get('error') and not eval_metrics.get('test_metrics_scaled')):
            return jsonify({
                'success': False,
                'error': f"No valid evaluation metrics found. Metrics: {eval_metrics}",
                'session_id': session_id
            }), 404

        # Format metrics to match original training_original.py structure exactly
        # The original uses nested dictionaries with output feature names as keys

        # Get output feature names (default to "Netzlast [kW]" if not available)
        output_features = results.get('output_features', ['Netzlast [kW]'])
        if not output_features:
            output_features = ['Netzlast [kW]']  # Default from original

        # df_eval - Dictionary with output features as keys
        # Each contains a DataFrame with metrics for different time deltas
        df_eval = {}

        # df_eval_ts - Dictionary with output features as keys
        # Each contains another dict with time deltas as keys
        df_eval_ts = {}

        # Time deltas in minutes (matching original n_max=12)
        time_deltas = [15, 30, 45, 60, 90, 120, 180, 240, 300, 360, 420, 480]

        # Process each output feature
        for feature_name in output_features:
            # Initialize lists for df_eval metrics
            delt_list = []
            mae_list = []
            mape_list = []
            mse_list = []
            rmse_list = []
            nrmse_list = []
            wape_list = []
            smape_list = []
            mase_list = []

            # Initialize df_eval_ts for this feature
            df_eval_ts[feature_name] = {}

            # Get test metrics if available
            test_metrics = eval_metrics.get('test_metrics_scaled', {})
            val_metrics = eval_metrics.get('val_metrics_scaled', {})

            # For each time delta, calculate or use available metrics
            for delta in time_deltas:
                # Add delta to list
                delt_list.append(float(delta))

                # For now, use the same test metrics for all deltas
                # In production, these would be calculated for each time averaging
                mae_list.append(float(test_metrics.get('MAE', 0.0)))
                mape_list.append(float(test_metrics.get('MAPE', 0.0)))
                mse_list.append(float(test_metrics.get('MSE', 0.0)))
                rmse_list.append(float(test_metrics.get('RMSE', 0.0)))
                nrmse_list.append(float(test_metrics.get('NRMSE', 0.0)))
                wape_list.append(float(test_metrics.get('WAPE', 0.0)))
                smape_list.append(float(test_metrics.get('sMAPE', 0.0)))
                mase_list.append(float(test_metrics.get('MASE', 0.0)))

                # Create df_eval_ts entry for this delta
                # This should contain per-timestep metrics
                timestep_metrics = []
                n_timesteps = results.get('n_timesteps', 96)  # Default 96 timesteps

                for ts in range(n_timesteps):
                    # In production, these would be actual per-timestep metrics
                    timestep_metrics.append({
                        'MAE': float(test_metrics.get('MAE', 0.0)),
                        'MAPE': float(test_metrics.get('MAPE', 0.0)),
                        'MSE': float(test_metrics.get('MSE', 0.0)),
                        'RMSE': float(test_metrics.get('RMSE', 0.0)),
                        'NRMSE': float(test_metrics.get('NRMSE', 0.0)),
                        'WAPE': float(test_metrics.get('WAPE', 0.0)),
                        'sMAPE': float(test_metrics.get('sMAPE', 0.0)),
                        'MASE': float(test_metrics.get('MASE', 0.0))
                    })

                df_eval_ts[feature_name][float(delta)] = timestep_metrics

            # Create DataFrame-like structure for df_eval
            df_eval[feature_name] = {
                "delta [min]": delt_list,
                "MAE": mae_list,
                "MAPE": mape_list,
                "MSE": mse_list,
                "RMSE": rmse_list,
                "NRMSE": nrmse_list,
                "WAPE": wape_list,
                "sMAPE": smape_list,
                "MASE": mase_list
            }

        return jsonify({
            'success': True,
            'session_id': session_id,
            'df_eval': df_eval,
            'df_eval_ts': df_eval_ts,
            'model_type': eval_metrics.get('model_type', 'Unknown')
        })

    except Exception as e:
        logger.error(f"Error getting evaluation tables: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'session_id': session_id
        }), 500


@bp.route('/save-evaluation-tables/<session_id>', methods=['POST'])
def save_evaluation_tables(session_id):
    """
    Save evaluation tables (df_eval and df_eval_ts) to database
    """
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        from datetime import datetime

        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        df_eval = data.get('df_eval', {})
        df_eval_ts = data.get('df_eval_ts', {})
        model_type = data.get('model_type', 'Unknown')

        if not df_eval and not df_eval_ts:
            return jsonify({
                'success': False,
                'error': 'No evaluation tables provided'
            }), 400

        # Get UUID for session
        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)

        # Prepare data for database
        evaluation_data = {
            'session_id': uuid_session_id,
            'df_eval': df_eval,
            'df_eval_ts': df_eval_ts,
            'model_type': model_type,
            'created_at': datetime.now().isoformat(),
            'table_format': 'original_training_format'
        }

        # Save to evaluation_tables table (create if doesn't exist)
        response = supabase.table('evaluation_tables').upsert(evaluation_data).execute()

        if response.data:
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'Evaluation tables saved successfully',
                'saved_at': evaluation_data['created_at']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to save to database'
            }), 500

    except Exception as e:
        logger.error(f"Error saving evaluation tables: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'session_id': session_id
        }), 500


# ============================================================================
# SCALER MANAGEMENT API ENDPOINTS
# ============================================================================

@bp.route('/scalers/<session_id>', methods=['GET'])
def get_scalers(session_id):
    """
    Retrieve saved scalers from database for a specific session.
    Returns input and output scalers that can be used for data normalization.
    """
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        import pickle
        import base64

        # Get the UUID session ID
        uuid_session_id = create_or_get_session_uuid(session_id)
        if not uuid_session_id:
            return jsonify({
                'success': False,
                'error': f'Session {session_id} not found'
            }), 404

        # Get scalers from database
        supabase = get_supabase_client()
        response = supabase.table('training_results').select('results').eq('session_id', uuid_session_id).execute()

        if not response.data:
            return jsonify({
                'success': False,
                'error': f'No training results found for session {session_id}'
            }), 404

        training_results = response.data[0]['results']
        scalers = training_results.get('scalers', {})

        if not scalers:
            return jsonify({
                'success': False,
                'error': f'No scalers found for session {session_id}'
            }), 404

        # Return scalers in serialized format (JSON-safe)
        input_scalers = scalers.get('input', {})
        output_scalers = scalers.get('output', {})

        return jsonify({
            'success': True,
            'session_id': session_id,
            'scalers': {
                'input': input_scalers,
                'output': output_scalers,
                'metadata': {
                    'input_features': len(input_scalers),
                    'output_features': len(output_scalers),
                    'input_features_scaled': sum(1 for s in input_scalers.values() if s is not None),
                    'output_features_scaled': sum(1 for s in output_scalers.values() if s is not None)
                }
            }
        })

    except Exception as e:
        logger.error(f"Error retrieving scalers for session {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to retrieve scalers from database'
        }), 500


@bp.route('/scalers/<session_id>/download', methods=['GET'])
def download_scalers_as_save_files(session_id):
    """Download scalers as .save files identical to original training_original.py format."""
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        import pickle, base64, os, zipfile, tempfile
        from datetime import datetime
        from flask import send_file

        uuid_session_id = create_or_get_session_uuid(session_id)
        if not uuid_session_id:
            return jsonify({'success': False, 'error': f'Session {session_id} not found'}), 404

        supabase = get_supabase_client()
        response = supabase.table('training_results').select('results').eq('session_id', uuid_session_id).execute()

        if not response.data:
            return jsonify({'success': False, 'error': f'No training results found for session {session_id}'}), 404

        scalers = response.data[0]['results'].get('scalers', {})
        if not scalers:
            return jsonify({'success': False, 'error': f'No scalers found for session {session_id}'}), 404

        # Deserialize scalers back to original format
        def deserialize_scalers_dict(scaler_dict):
            result = {}
            for key, scaler_data in scaler_dict.items():
                if scaler_data and isinstance(scaler_data, dict) and '_model_type' in scaler_data:
                    try:
                        scaler = pickle.loads(base64.b64decode(scaler_data['_model_data']))
                        result[int(key)] = scaler  # Convert key to int for original format
                    except Exception as e:
                        logger.error(f"Error deserializing scaler {key}: {str(e)}")
                        result[int(key)] = None
                else:
                    result[int(key)] = None
            return result

        input_scalers = deserialize_scalers_dict(scalers.get('input', {}))
        output_scalers = deserialize_scalers_dict(scalers.get('output', {}))

        # Create temporary directory for files
        temp_dir = tempfile.mkdtemp()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create .save files identical to original format
        i_scale_file = os.path.join(temp_dir, f'i_scale_{timestamp}.save')
        o_scale_file = os.path.join(temp_dir, f'o_scale_{timestamp}.save')

        # Save input scalers
        with open(i_scale_file, 'wb') as f:
            pickle.dump(input_scalers, f)

        # Save output scalers
        with open(o_scale_file, 'wb') as f:
            pickle.dump(output_scalers, f)

        # Create ZIP file with both .save files
        zip_file = os.path.join(temp_dir, f'scalers_{session_id}_{timestamp}.zip')
        with zipfile.ZipFile(zip_file, 'w') as zipf:
            zipf.write(i_scale_file, f'i_scale_{timestamp}.save')
            zipf.write(o_scale_file, f'o_scale_{timestamp}.save')

        logger.info(f"Created scaler files for session {session_id}: {zip_file}")

        # Send ZIP file as download
        return send_file(
            zip_file,
            as_attachment=True,
            download_name=f'scalers_{session_id}_{timestamp}.zip',
            mimetype='application/zip'
        )

    except Exception as e:
        logger.error(f"Error creating scaler download for session {session_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/scale-data/<session_id>', methods=['POST'])
def scale_input_data(session_id):
    """
    Scale input data using saved scalers (Skalierung Eingabedaten speichern).
    Takes raw input data and returns scaled data ready for model prediction.
    """
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        import numpy as np
        import pandas as pd
        import pickle
        import base64

        # Get request data
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        input_data = data.get('input_data')
        if input_data is None:
            return jsonify({'success': False, 'error': 'input_data field is required'}), 400

        # Convert input_data to numpy array
        try:
            if isinstance(input_data, list):
                input_array = np.array(input_data)
            elif isinstance(input_data, dict):
                # Assume it's a pandas DataFrame-like structure
                input_array = np.array(list(input_data.values())).T
            else:
                input_array = np.array(input_data)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to convert input_data to array: {str(e)}'
            }), 400

        # Get the UUID session ID
        uuid_session_id = create_or_get_session_uuid(session_id)
        if not uuid_session_id:
            return jsonify({
                'success': False,
                'error': f'Session {session_id} not found'
            }), 404

        # Get scalers from database
        supabase = get_supabase_client()
        response = supabase.table('training_results').select('results').eq('session_id', uuid_session_id).execute()

        if not response.data:
            return jsonify({
                'success': False,
                'error': f'No training results found for session {session_id}'
            }), 404

        training_results = response.data[0]['results']
        scalers = training_results.get('scalers', {})
        input_scalers = scalers.get('input', {})

        if not input_scalers:
            return jsonify({
                'success': False,
                'error': f'No input scalers found for session {session_id}'
            }), 404

        # Helper function to deserialize scalers
        def deserialize_scaler(scaler_data):
            """Convert serialized scaler back to usable object"""
            if scaler_data is None:
                return None
            elif isinstance(scaler_data, dict) and '_model_type' in scaler_data:
                # Deserialize pickled scaler
                try:
                    model_b64 = scaler_data['_model_data']
                    model_bytes = base64.b64decode(model_b64)
                    scaler = pickle.loads(model_bytes)
                    return scaler
                except Exception as e:
                    logger.error(f"Error deserializing scaler: {str(e)}")
                    return None
            else:
                return scaler_data

        # Scale the input data
        scaled_data = input_array.copy()
        scaling_info = {}

        for i in range(input_array.shape[1]):
            if str(i) in input_scalers:
                scaler = deserialize_scaler(input_scalers[str(i)])
                if scaler is not None:
                    try:
                        # Scale the column
                        original_data = input_array[:, i].reshape(-1, 1)
                        scaled_column = scaler.transform(original_data)
                        scaled_data[:, i] = scaled_column.flatten()

                        scaling_info[f'feature_{i}'] = {
                            'scaled': True,
                            'original_range': [float(np.min(original_data)), float(np.max(original_data))],
                            'scaled_range': [float(np.min(scaled_column)), float(np.max(scaled_column))],
                            'feature_range': scaler.feature_range
                        }
                    except Exception as e:
                        logger.error(f"Error scaling feature {i}: {str(e)}")
                        scaling_info[f'feature_{i}'] = {'scaled': False, 'error': str(e)}
                else:
                    scaling_info[f'feature_{i}'] = {'scaled': False, 'reason': 'no_scaler'}
            else:
                scaling_info[f'feature_{i}'] = {'scaled': False, 'reason': 'scaler_not_found'}

        # Optionally save scaled data
        save_scaled = data.get('save_scaled', False)
        saved_file_path = None

        if save_scaled:
            try:
                import os
                from datetime import datetime

                # Create scaled data directory if it doesn't exist
                scaled_dir = f"temp_uploads/scaled_data_{session_id}"
                os.makedirs(scaled_dir, exist_ok=True)

                # Save as CSV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"scaled_input_data_{timestamp}.csv"
                file_path = os.path.join(scaled_dir, file_name)

                # Create DataFrame and save
                scaled_df = pd.DataFrame(scaled_data, columns=[f'feature_{i}' for i in range(scaled_data.shape[1])])
                scaled_df.to_csv(file_path, index=False)
                saved_file_path = file_path

                logger.info(f"Scaled data saved to: {file_path}")

            except Exception as e:
                logger.error(f"Error saving scaled data: {str(e)}")

        return jsonify({
            'success': True,
            'session_id': session_id,
            'scaled_data': scaled_data.tolist(),
            'scaling_info': scaling_info,
            'metadata': {
                'original_shape': input_array.shape,
                'scaled_shape': scaled_data.shape,
                'features_scaled': sum(1 for info in scaling_info.values() if info.get('scaled', False)),
                'total_features': len(scaling_info),
                'saved_file_path': saved_file_path
            }
        })

    except Exception as e:
        logger.error(f"Error scaling data for session {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to scale input data'
        }), 500


def cleanup_incomplete_uploads(upload_base_dir: str, max_age_hours: int = 24) -> int:
    """
    Clean up incomplete or old upload sessions.

    Args:
        upload_base_dir: Base directory for uploads
        max_age_hours: Maximum age for incomplete uploads

    Returns:
        int: Number of cleaned up sessions
    """
    cleaned_count = 0
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    try:
        for session_dir in Path(upload_base_dir).iterdir():
            if not session_dir.is_dir():
                continue

            # Check if session is old
            dir_age = current_time - session_dir.stat().st_mtime
            if dir_age > max_age_seconds:
                # Check if session is incomplete (no finalized marker)
                finalized_marker = session_dir / '.finalized'
                if not finalized_marker.exists():
                    try:
                        shutil.rmtree(session_dir)
                        cleaned_count += 1
                        logger.info(f"Cleaned up old session: {session_dir.name}")
                    except Exception as e:
                        logger.error(f"Error cleaning up {session_dir}: {e}")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

    return cleaned_count



@bp.route('/session-name-change', methods=['POST'])
def change_session_name():
    """
    Update session name in the database.

    Request body:
    {
        "sessionId": "...",
        "sessionName": "novo ime"
    }
    """
    try:
        from utils.database import update_session_name, ValidationError, ConfigurationError, SessionNotFoundError, DatabaseError

        # Get request data
        data = request.get_json()
        if not data:
            return create_error_response('No data provided', 400)

        session_id = data.get('sessionId')
        session_name = data.get('sessionName')

        # Validate inputs
        if not session_id:
            return create_error_response('sessionId is required', 400)

        if not session_name:
            return create_error_response('sessionName is required', 400)

        if not isinstance(session_name, str):
            return create_error_response('sessionName must be a string', 400)

        # Trim and validate session name
        session_name = session_name.strip()
        if len(session_name) == 0:
            return create_error_response('sessionName cannot be empty', 400)

        if len(session_name) > 255:
            return create_error_response('sessionName too long (max 255 characters)', 400)

        logger.info(f"Updating session name for {session_id} to: {session_name}")

        # Update session name in database
        try:
            success = update_session_name(session_id, session_name)

            if success:
                return create_success_response(
                    data={
                        'sessionId': session_id,
                        'sessionName': session_name
                    },
                    message='Session name updated successfully'
                )
            else:
                return create_error_response('Failed to update session name', 500)

        except ValidationError as e:
            logger.warning(f"Validation error: {str(e)}")
            return create_error_response(str(e), 400)
        except SessionNotFoundError as e:
            logger.warning(f"Session not found: {str(e)}")
            return create_error_response(str(e), 404)
        except ConfigurationError as e:
            logger.error(f"Configuration error: {str(e)}")
            return create_error_response('Database connection not available', 500)
        except DatabaseError as e:
            logger.error(f"Database error: {str(e)}")
            return create_error_response(f'Database error: {str(e)}', 500)

    except Exception as e:
        logger.error(f"Error changing session name: {str(e)}")
        return create_error_response(f'Internal server error: {str(e)}', 500)
