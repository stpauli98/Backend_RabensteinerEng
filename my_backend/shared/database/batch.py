"""
Batch file operations for database.

This module handles batch insert and upsert operations for file data,
providing efficient bulk operations with proper error handling.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List

from supabase import Client

from .config import DomainDefaults, TableNames
from .exceptions import DatabaseError
from .validators import validate_file_info, sanitize_filename


logger = logging.getLogger(__name__)


def _map_file_info_to_db_record(
    file_info: Dict[str, Any],
    database_session_id: str,
    zeitschrittweite_mittelwert: Any = None,
    zeitschrittweite_min: Any = None
) -> Dict[str, Any]:
    """Map frontend file_info to database record format.

    This is a shared helper that both save_file_info and _prepare_file_batch_data use.

    Args:
        file_info: File info from frontend
        database_session_id: UUID session ID
        zeitschrittweite_mittelwert: Fallback value from input file
        zeitschrittweite_min: Fallback value from input file

    Returns:
        dict: Database record ready for insert/update
    """
    file_id = file_info.get("id")
    try:
        valid_uuid = str(uuid.UUID(file_id))
    except (ValueError, TypeError, AttributeError):
        valid_uuid = str(uuid.uuid4())

    storage_path = file_info.get("storagePath", "")
    if not storage_path:
        file_name = file_info.get("fileName", "")
        if file_name:
            storage_path = f"{database_session_id}/{sanitize_filename(file_name)}"

    # Get zeitschrittweite values, with fallbacks
    zw_mittelwert = file_info.get(
        "zeitschrittweiteAvgValue",
        file_info.get("zeitschrittweiteMittelwert")
    ) or None
    zw_min = file_info.get(
        "zeitschrittweiteMinValue",
        file_info.get("zeitschrittweiteMin")
    ) or None

    # For output files, use fallback values from input
    file_type = file_info.get("type", "")
    if file_type == "output":
        if zw_mittelwert is None and zeitschrittweite_mittelwert:
            zw_mittelwert = zeitschrittweite_mittelwert
        if zw_min is None and zeitschrittweite_min:
            zw_min = zeitschrittweite_min

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
        "zeitschrittweite_mittelwert": zw_mittelwert,
        "zeitschrittweite_min": zw_min,
        "skalierung": file_info.get("skalierung", DomainDefaults.SKALIERUNG),
        "skalierung_max": str(file_info.get("skalierungMax", "")),
        "skalierung_min": str(file_info.get("skalierungMin", "")),
        "zeithorizont_start": file_info.get("zeithorizontStart", ""),
        "zeithorizont_end": file_info.get("zeithorizontEnd", ""),
        "zeitschrittweite_transferierten_daten": str(
            file_info.get("zeitschrittweiteTransferiertenDaten", "")
        ),
        "offset_transferierten_daten": str(file_info.get("offsetTransferiertenDaten", "")),
        "mittelwertbildung_uber_den_zeithorizont": file_info.get(
            "mittelwertbildungÜberDenZeithorizont",
            DomainDefaults.MITTELWERTBILDUNG
        ),
        "storage_path": storage_path,
        "type": file_info.get("type", DomainDefaults.FILE_TYPE)
    }

    # Handle UTC timestamps
    for field_name, value in [("utcMin", file_info.get("utcMin")), ("utcMax", file_info.get("utcMax"))]:
        if value:
            try:
                dt_obj = datetime.fromisoformat(value)
                db_field_name = "utc_min" if field_name == "utcMin" else "utc_max"
                data[db_field_name] = dt_obj.isoformat(sep=' ', timespec='seconds')
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid {field_name} format: {value}, error: {str(e)}")

    return data


def prepare_file_batch_data(
    database_session_id: str,
    files_list: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Prepare batch data for file insertion.

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
            input_zeitschrittweite_mittelwert = file_info.get(
                "zeitschrittweiteAvgValue",
                file_info.get("zeitschrittweiteMittelwert")
            ) or None
            input_zeitschrittweite_min = file_info.get(
                "zeitschrittweiteMinValue",
                file_info.get("zeitschrittweiteMin")
            ) or None
            if input_zeitschrittweite_mittelwert or input_zeitschrittweite_min:
                break

    for file_info in files_list:
        if not validate_file_info(file_info):
            logger.warning(f"Skipping invalid file info: {file_info}")
            continue

        data = _map_file_info_to_db_record(
            file_info,
            database_session_id,
            input_zeitschrittweite_mittelwert,
            input_zeitschrittweite_min
        )
        batch_data.append(data)

    return batch_data


def batch_upsert_files(
    supabase: Client,
    database_session_id: str,
    batch_data: List[Dict[str, Any]]
) -> List[str]:
    """Perform smart UPSERT of file data - INSERT new, UPDATE existing (no deletions).

    Args:
        supabase: Supabase client
        database_session_id: UUID format session ID
        batch_data: List of file data to upsert

    Returns:
        list: List of successfully upserted file UUIDs

    Raises:
        DatabaseError: If batch operation fails
    """
    if not batch_data:
        return []

    try:
        # 1. Get existing files for this session
        existing_response = supabase.table(TableNames.FILES)\
            .select("*")\
            .eq("session_id", database_session_id)\
            .execute()
        # Build lookup by ID (primary) for proper UPSERT matching
        existing_files_by_id = {
            f['id']: f
            for f in (existing_response.data or [])
            if f.get('id')
        }

        # 2. Classify files for INSERT or UPDATE (NO DELETE - preserve existing)
        files_to_insert = []
        files_to_update = []
        upserted_uuids = []

        # 3. Classify each file by ID (not bezeichnung)
        for file_data in batch_data:
            file_id = file_data.get('id')
            if file_id and file_id in existing_files_by_id:
                # File with same ID exists - UPDATE
                files_to_update.append(file_data)
                upserted_uuids.append(file_id)
            else:
                # New file - INSERT
                files_to_insert.append(file_data)

        # 4. Batch INSERT new files
        if files_to_insert:
            insert_response = supabase.table(TableNames.FILES)\
                .insert(files_to_insert)\
                .execute()

            if hasattr(insert_response, 'error') and insert_response.error:
                raise DatabaseError(f"Batch file insert failed: {insert_response.error}")

            inserted_uuids = [
                item['id'] for item in insert_response.data if 'id' in item
            ]
            upserted_uuids.extend(inserted_uuids)

        # 5. Update existing files one by one
        if files_to_update:
            for file_data in files_to_update:
                file_id = file_data.get('id')
                # Create update payload without id (can't update primary key)
                update_payload = {k: v for k, v in file_data.items() if k != 'id'}
                update_response = supabase.table(TableNames.FILES)\
                    .update(update_payload)\
                    .eq("id", file_id)\
                    .execute()

                if hasattr(update_response, 'error') and update_response.error:
                    logger.error(
                        f"Failed to update file {file_id} ({file_data.get('bezeichnung')}): "
                        f"{update_response.error}"
                    )

        return upserted_uuids

    except DatabaseError:
        raise
    except Exception as e:
        raise DatabaseError(f"Batch file upsert failed: {str(e)}")


# Aliases for backward compatibility
_prepare_file_batch_data = prepare_file_batch_data
_batch_upsert_files = batch_upsert_files
