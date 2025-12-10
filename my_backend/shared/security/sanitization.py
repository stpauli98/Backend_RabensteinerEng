"""Sigurnosne utility funkcije za sanitizaciju korisničkog unosa"""
import re
import os
import unicodedata
from werkzeug.utils import secure_filename

# Regex za provjeru sigurnih karaktera u imenima datoteka
SAFE_CHARS = re.compile(r'^[a-zA-Z0-9_\-][a-zA-Z0-9_\-\.]*$')

# Maksimalna duljina imena datoteke
MAX_FILENAME_LENGTH = 255

# Blokirane ekstenzije (potencijalno opasne)
BLOCKED_EXTENSIONS = {
    '.exe', '.bat', '.cmd', '.sh', '.ps1', '.vbs', '.php',
    '.pif', '.application', '.gadget', '.msi', '.msp', '.com',
    '.scr', '.hta', '.cpl', '.msc', '.jar', '.ws', '.wsf',
    '.wsc', '.wsh', '.scf', '.lnk', '.inf', '.reg'
}

# Dozvoljene ekstenzije za upload
ALLOWED_EXTENSIONS = {
    '.csv', '.txt', '.json', '.xml', '.xlsx', '.xls',
    '.h5', '.pkl', '.pickle', '.npy', '.npz',
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.pdf'
}


def sanitize_filename(filename: str, strict: bool = False) -> str:
    """
    Sanitiziraj ime datoteke za siguran pristup filesystem-u.

    Uklanja:
    - Path traversal pokušaje (../)
    - Null bajtove
    - Kontrolne karaktere
    - Opasne ekstenzije

    Args:
        filename: Originalno ime datoteke
        strict: Ako True, dozvoljava samo ALLOWED_EXTENSIONS

    Returns:
        str: Sanitizirano ime datoteke

    Raises:
        ValueError: Ako ime nije validno ili sadrži opasne elemente
    """
    if not filename or not isinstance(filename, str):
        raise ValueError("Ime datoteke je obavezno")

    # Ukloni null bajtove - može se koristiti za bypass
    filename = filename.replace('\x00', '').strip()

    # Ukloni kontrolne karaktere (Unicode kategorija 'Cc')
    filename = ''.join(c for c in filename if unicodedata.category(c) != 'Cc')

    # Uzmi samo bazno ime (uklanja path komponente kao ../ ili /etc/)
    filename = os.path.basename(filename)

    # Koristi Werkzeug secure_filename za dodatnu sanitizaciju
    # Ovo uklanja specijalne karaktere i normalizira unicode
    filename = secure_filename(filename)

    if not filename:
        raise ValueError("Ime datoteke sadrži samo nevažeće karaktere")

    # Provjeri duljinu
    if len(filename) > MAX_FILENAME_LENGTH:
        name, ext = os.path.splitext(filename)
        # Skrati ime ali zadrži ekstenziju
        max_name_length = MAX_FILENAME_LENGTH - len(ext)
        filename = name[:max_name_length] + ext

    # Provjeri ekstenziju
    _, ext = os.path.splitext(filename.lower())

    # Blokiraj opasne ekstenzije
    if ext in BLOCKED_EXTENSIONS:
        raise ValueError(f"Ekstenzija '{ext}' nije dozvoljena iz sigurnosnih razloga")

    # U strogom modu, provjeri je li ekstenzija u dozvoljenoj listi
    if strict and ext and ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Ekstenzija '{ext}' nije podržana. Dozvoljene: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    return filename


def validate_session_path(session_id: str, base_dir: str) -> str:
    """
    Validiraj i konstruiraj siguran path za sesiju.

    Sprječava path traversal napade osiguravajući da generirani
    path ostaje unutar base_dir direktorija.

    Args:
        session_id: Session ID (može sadržavati samo alfanumeričke karaktere, _ i -)
        base_dir: Bazni direktorij unutar kojeg mora biti rezultat

    Returns:
        str: Siguran apsolutni path za sesiju

    Raises:
        ValueError: Ako session_id sadrži nevažeće karaktere ili path izlazi iz base_dir
    """
    if not session_id or not isinstance(session_id, str):
        raise ValueError("Session ID je obavezan")

    # Dozvoli samo alfanumeričke karaktere, _ i -
    safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '', session_id)

    if not safe_id:
        raise ValueError("Session ID sadrži samo nevažeće karaktere")

    if safe_id != session_id:
        raise ValueError(f"Session ID sadrži nedozvoljene karaktere. Dozvoljen: {safe_id}")

    # Konstruiraj apsolutni path
    target = os.path.abspath(os.path.join(base_dir, safe_id))
    base = os.path.abspath(base_dir)

    # Provjeri je li target path unutar base direktorija
    # Dodaj os.sep da spriječimo base="/uploads" target="/uploads_evil"
    if not target.startswith(base + os.sep) and target != base:
        raise ValueError("Path traversal detektiran - pristup odbijen")

    return target


def sanitize_user_input(value: str, max_length: int = 1000, allow_html: bool = False) -> str:
    """
    Sanitiziraj korisnički tekstualni unos.

    Args:
        value: Korisnički unos
        max_length: Maksimalna dozvoljena duljina
        allow_html: Ako False, uklanja HTML tagove

    Returns:
        str: Sanitizirani tekst
    """
    if not value or not isinstance(value, str):
        return ""

    # Ukloni null bajtove
    value = value.replace('\x00', '')

    # Ukloni kontrolne karaktere osim newline i tab
    value = ''.join(c for c in value if c in '\n\t' or unicodedata.category(c) != 'Cc')

    # Ukloni HTML tagove ako nije dozvoljeno
    if not allow_html:
        value = re.sub(r'<[^>]+>', '', value)

    # Skrati na maksimalnu duljinu
    if len(value) > max_length:
        value = value[:max_length]

    return value.strip()
