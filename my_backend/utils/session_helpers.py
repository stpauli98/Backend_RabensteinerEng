"""Pomoćne funkcije za rad sa session ID-jevima"""
import uuid as uuid_lib
import logging
from typing import Tuple, Optional

from shared.database.operations import get_string_id_from_uuid, create_or_get_session_uuid

logger = logging.getLogger(__name__)


def is_uuid_format(value: str) -> bool:
    """
    Provjeri je li vrijednost u UUID formatu.

    Args:
        value: String za provjeru

    Returns:
        bool: True ako je UUID format, False inače
    """
    if not value or not isinstance(value, str):
        return False

    try:
        uuid_lib.UUID(value)
        return True
    except (ValueError, TypeError):
        return False


def resolve_session_id(session_id: str, user_id: Optional[str] = None) -> Tuple[str, str]:
    """
    Razriješi session ID u oba formata (string i UUID).

    Ova funkcija eliminira duplicirani kod za pretvaranje između
    string session ID-a i UUID session ID-a koji se koristi u bazi.

    Args:
        session_id: Session ID (može biti UUID ili string format)
        user_id: User ID za validaciju vlasništva (opcionalno, potrebno za kreiranje)

    Returns:
        Tuple[str, str]: (string_session_id, uuid_session_id)

    Raises:
        ValueError: Ako mapiranje ne postoji i ne može se kreirati
    """
    if not session_id:
        raise ValueError("Session ID je obavezan")

    if is_uuid_format(session_id):
        # Input je UUID format - pronađi odgovarajući string ID
        string_id = get_string_id_from_uuid(session_id)
        if not string_id:
            raise ValueError(f'Session mapping not found for UUID: {session_id}')
        return string_id, session_id
    else:
        # Input je string format - pronađi ili kreiraj UUID
        uuid_id = create_or_get_session_uuid(session_id, user_id)
        if not uuid_id:
            raise ValueError(f'Could not resolve session: {session_id}')
        return session_id, uuid_id


def get_string_session_id(session_id: str) -> str:
    """
    Osiguraj da imaš string session ID (lokalni format).

    Args:
        session_id: Session ID u bilo kojem formatu

    Returns:
        str: String session ID

    Raises:
        ValueError: Ako session ne postoji
    """
    if is_uuid_format(session_id):
        string_id = get_string_id_from_uuid(session_id)
        if not string_id:
            raise ValueError(f'Session not found for UUID: {session_id}')
        return string_id
    return session_id


def get_uuid_session_id(session_id: str, user_id: Optional[str] = None) -> str:
    """
    Osiguraj da imaš UUID session ID (database format).

    Args:
        session_id: Session ID u bilo kojem formatu
        user_id: User ID za kreiranje ako ne postoji

    Returns:
        str: UUID session ID

    Raises:
        ValueError: Ako session ne postoji i ne može se kreirati
    """
    if is_uuid_format(session_id):
        return session_id

    uuid_id = create_or_get_session_uuid(session_id, user_id)
    if not uuid_id:
        raise ValueError(f'Could not get UUID for session: {session_id}')
    return uuid_id


def validate_session_ownership(session_id: str, user_id: str) -> bool:
    """
    Validiraj da korisnik ima pristup sesiji.

    Args:
        session_id: Session ID
        user_id: User ID za provjeru

    Returns:
        bool: True ako korisnik ima pristup

    Note:
        Trenutno vraća True jer se vlasništvo provjerava putem Supabase RLS policy-a
        koji osigurava da korisnik može pristupiti samo svojim sesijama.
        Ova funkcija je placeholder za dodatnu aplikacijsku provjeru ako bude potrebna.
    """
    # Ownership is currently enforced via Supabase RLS policies on sessions table.
    # This function can be extended if additional application-level checks are needed.
    return True
