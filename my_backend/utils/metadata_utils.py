"""
Metadata extraction utilities
Handles extraction and standardization of file metadata from sessions
"""

import os
import json
import logging

logger = logging.getLogger(__name__)


def extract_file_metadata_fields(file_metadata):
    """
    Extract standardized file metadata fields from a file metadata dictionary.

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
        'type': file_metadata.get('type', '')
    }


def extract_file_metadata(session_id, upload_base_dir):
    """
    Extract file metadata from session metadata.

    Args:
        session_id: ID of the session to extract metadata from
        upload_base_dir: Base directory for file uploads

    Returns:
        dict: Dictionary containing file metadata fields or None if not found
    """
    try:
        upload_dir = os.path.join(upload_base_dir, session_id)
        metadata_path = os.path.join(upload_dir, 'metadata.json')

        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file not found for session {session_id}")
            return None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        for chunk in metadata:
            if 'params' in chunk and 'fileMetadata' in chunk['params']:
                file_metadata = chunk['params']['fileMetadata']
                return extract_file_metadata_fields(file_metadata)

        logger.error(f"No file metadata found for session {session_id}")
        return None
    except Exception as e:
        logger.error(f"Error extracting file metadata: {str(e)}")
        return None
