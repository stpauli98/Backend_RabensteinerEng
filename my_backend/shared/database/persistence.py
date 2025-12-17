"""
Data persistence functions for database operations.

This module handles saving time info, zeitschritte, and file info to the database.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from .config import DomainDefaults, TableNames
from .exceptions import (
    DatabaseError,
    SessionNotFoundError,
    ValidationError,
    ConfigurationError
)
from .validators import validate_session_id, validate_time_info, validate_file_info, sanitize_filename
from .session import get_supabase_client, get_session_uuid, create_or_get_session_uuid


logger = logging.getLogger(__name__)


def save_time_info(session_id: str, time_info: dict) -> bool:
    """Save time information to the time_info table.

    Args:
        session_id: ID of the session (can be string or UUID)
        time_info: Dictionary containing time information

    Returns:
        bool: True if successful

    Raises:
        ValidationError: If input parameters are invalid
        ConfigurationError: If Supabase client is not available
        SessionNotFoundError: If session cannot be found
        DatabaseError: If database operations fail
    """
    if not validate_session_id(session_id):
        raise ValidationError(f"Invalid session_id format: {session_id}")

    if not validate_time_info(time_info):
        raise ValidationError(f"Invalid time_info structure: {time_info}")

    supabase = get_supabase_client()
    database_session_id = get_session_uuid(session_id)

    data = {
        "session_id": database_session_id,
        "jahr": time_info.get("jahr", False),
        "woche": time_info.get("woche", False),
        "monat": time_info.get("monat", False),
        "feiertag": time_info.get("feiertag", False),
        "tag": time_info.get("tag", False),
        "zeitzone": time_info.get("zeitzone", DomainDefaults.ZEITZONE),
        "category_data": time_info.get("category_data", {})
    }

    if isinstance(data['category_data'], dict):
        category_data_str = json.dumps(data['category_data'], ensure_ascii=False)
        data['category_data'] = json.loads(category_data_str)

    try:
        existing = supabase.table(TableNames.TIME_INFO)\
            .select("*")\
            .eq("session_id", database_session_id)\
            .execute()

        if existing.data and len(existing.data) > 0:
            response = supabase.table(TableNames.TIME_INFO)\
                .update(data)\
                .eq("session_id", database_session_id)\
                .execute()
        else:
            response = supabase.table(TableNames.TIME_INFO).insert(data).execute()

        if hasattr(response, 'error') and response.error:
            raise DatabaseError(f"Error saving time_info: {response.error}")

        return True

    except (DatabaseError, ValidationError, SessionNotFoundError, ConfigurationError):
        raise
    except Exception as e:
        raise DatabaseError(f"Error saving time_info: {str(e)}")


def save_zeitschritte(session_id: str, zeitschritte: dict, user_id: str = None) -> bool:
    """Save zeitschritte information to the zeitschritte table.

    Args:
        session_id: ID of the session (can be string or UUID)
        zeitschritte: Dictionary containing zeitschritte information
        user_id: User ID (required for creating new sessions)

    Returns:
        bool: True if successful

    Raises:
        ValidationError: If input parameters are invalid
        ConfigurationError: If Supabase client is not available
        SessionNotFoundError: If session cannot be found
        DatabaseError: If database operations fail
    """
    if not validate_session_id(session_id):
        raise ValidationError(f"Invalid session_id format: {session_id}")

    supabase = get_supabase_client()

    try:
        uuid.UUID(session_id)
        database_session_id = session_id
    except (ValueError, TypeError):
        database_session_id = create_or_get_session_uuid(session_id, user_id=user_id)

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

    try:
        existing = supabase.table(TableNames.ZEITSCHRITTE)\
            .select("*")\
            .eq("session_id", database_session_id)\
            .execute()

        if existing.data and len(existing.data) > 0:
            response = supabase.table(TableNames.ZEITSCHRITTE)\
                .update(data)\
                .eq("session_id", database_session_id)\
                .execute()
        else:
            response = supabase.table(TableNames.ZEITSCHRITTE).insert(data).execute()

        if hasattr(response, 'error') and response.error:
            raise DatabaseError(f"Error saving zeitschritte: {response.error}")

        return True

    except (DatabaseError, ValidationError, SessionNotFoundError, ConfigurationError):
        raise
    except Exception as e:
        raise DatabaseError(f"Error saving zeitschritte: {str(e)}")


def save_file_info(session_id: str, file_info: dict) -> Tuple[bool, Optional[str]]:
    """Save file information to the files table.

    Args:
        session_id: ID of the session (can be string or UUID)
        file_info: Dictionary containing file information

    Returns:
        tuple: (success, valid_uuid) where success is True if successful,
               and valid_uuid is the UUID used for the file

    Raises:
        ValidationError: If input parameters are invalid
        ConfigurationError: If Supabase client is not available
        SessionNotFoundError: If session cannot be found
        DatabaseError: If database operations fail
    """
    if not validate_file_info(file_info):
        raise ValidationError(f"Invalid file_info structure: missing required fields")

    supabase = get_supabase_client()

    try:
        uuid.UUID(session_id)
        database_session_id = session_id
    except (ValueError, TypeError):
        database_session_id = create_or_get_session_uuid(session_id)

    file_id = file_info.get("id")
    try:
        valid_uuid = str(uuid.UUID(file_id))
    except (ValueError, TypeError, AttributeError):
        valid_uuid = str(uuid.uuid4())

    storage_path = file_info.get("storagePath", "")
    if not storage_path:
        file_name = file_info.get("fileName", "")
        bezeichnung = file_info.get("bezeichnung", "")
        if file_name:
            # Include bezeichnung in storage path to prevent overwrites
            if bezeichnung:
                safe_bezeichnung = sanitize_filename(bezeichnung)
                safe_filename = sanitize_filename(file_name)
                storage_path = f"{database_session_id}/{safe_bezeichnung}_{safe_filename}"
            else:
                storage_path = f"{database_session_id}/{sanitize_filename(file_name)}"

    # Get zeitschrittweite values from file_info
    zeitschrittweite_mittelwert = file_info.get(
        "zeitschrittweiteAvgValue",
        file_info.get("zeitschrittweiteMittelwert")
    ) or None
    zeitschrittweite_min_val = file_info.get(
        "zeitschrittweiteMinValue",
        file_info.get("zeitschrittweiteMin")
    ) or None

    # For output files without zeitschrittweite values, try to copy from input file
    file_type = file_info.get("type", "")
    if file_type == "output" and (zeitschrittweite_mittelwert is None or zeitschrittweite_min_val is None):
        try:
            input_file = supabase.table(TableNames.FILES).select(
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
        "skalierung": file_info.get("skalierung", DomainDefaults.SKALIERUNG),
        "skalierung_max": str(file_info.get("skalierungMax", "")),
        "skalierung_min": str(file_info.get("skalierungMin", "")),
        "zeithorizont_start": file_info.get("zeithorizontStart", ""),
        "zeithorizont_end": file_info.get("zeithorizontEnd", ""),
        "zeitschrittweite_transferierten_daten": str(file_info.get("zeitschrittweiteTransferiertenDaten", "")),
        "offset_transferierten_daten": str(file_info.get("offsetTransferiertenDaten", "")),
        "mittelwertbildung_uber_den_zeithorizont": file_info.get(
            "mittelwertbildungÜberDenZeithorizont",
            DomainDefaults.MITTELWERTBILDUNG
        ),
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

    try:
        response = supabase.table(TableNames.FILES).insert(data).execute()

        if hasattr(response, 'error') and response.error:
            raise DatabaseError(f"Error saving file info: {response.error}")

        return True, valid_uuid

    except (DatabaseError, ValidationError, SessionNotFoundError, ConfigurationError):
        raise
    except Exception as e:
        raise DatabaseError(f"Error saving file info: {str(e)}")


def transform_time_info_to_jsonb(time_info: Dict[str, Any]) -> Dict[str, Any]:
    """Transform old time information format to the new JSONB structure format.

    Args:
        time_info: Dictionary containing time information in the old format

    Returns:
        dict: Time information in the new format with category_data JSONB structure
    """
    new_time_info = {
        "jahr": time_info.get("jahr", False),
        "monat": time_info.get("monat", False),
        "woche": time_info.get("woche", False),
        "feiertag": time_info.get("feiertag", False),
        "tag": time_info.get("tag", False),
        "zeitzone": time_info.get("zeitzone", "")
    }

    if "category_data" in time_info:
        new_time_info["category_data"] = time_info["category_data"]
        return new_time_info

    category_data = {}

    detaillierte_berechnung = time_info.get("detaillierteBerechnung", False)
    datenform = time_info.get("datenform", "")
    zeithorizont_start = time_info.get("zeithorizontStart", "")
    zeithorizont_end = time_info.get("zeithorizontEnd", "")
    skalierung = time_info.get("skalierung", DomainDefaults.SKALIERUNG)
    skalierung_min = time_info.get("skalierungMin", "")
    skalierung_max = time_info.get("skalierungMax", "")

    category_template = {
        "detaillierteBerechnung": detaillierte_berechnung,
        "datenform": datenform,
        "zeithorizontStart": zeithorizont_start,
        "zeithorizontEnd": zeithorizont_end,
        "skalierung": skalierung,
        "skalierungMin": skalierung_min,
        "skalierungMax": skalierung_max
    }

    if new_time_info["jahr"]:
        category_data["jahr"] = category_template.copy()

    if new_time_info["monat"]:
        category_data["monat"] = category_template.copy()

    if new_time_info["woche"]:
        category_data["woche"] = category_template.copy()

    if new_time_info["feiertag"]:
        feiertag_data = category_template.copy()
        feiertag_data["land"] = time_info.get("land", DomainDefaults.LAND)
        category_data["feiertag"] = feiertag_data

    new_time_info["category_data"] = category_data

    return new_time_info


# Alias for backward compatibility
_transform_time_info_to_jsonb = transform_time_info_to_jsonb
