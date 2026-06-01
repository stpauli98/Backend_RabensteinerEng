"""Shared helpers for classifying Supabase storage exceptions.

When a storage download fails, we need to distinguish "object does not
exist" (404 to the client) from any other failure mode (500). Supabase's
``storage3`` library raises ``StorageException`` / ``StorageApiError``
with a numeric ``.status`` attribute; older versions or different paths
may surface the error as a string. This helper checks both.
"""


def is_storage_not_found(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like a Supabase storage "object not found".

    Detects:
    - ``FileNotFoundError`` (local-FS fallback path)
    - ``storage3.utils.StorageException`` (or subclass) whose ``.status``
      attribute is 404
    - Any exception whose stringified form contains "not found",
      "no such", or "object not found" (defence in depth for version
      drift in storage3's error contract)

    Use in route handlers to decide between 404 + ``MODEL_NOT_FOUND`` /
    ``SCALER_NOT_FOUND`` / ``RESULTS_NOT_FOUND`` and 500 + ``INTERNAL_ERROR``.

    Args:
        exc: The caught exception.

    Returns:
        ``True`` if the exception means the storage object is missing,
        ``False`` otherwise.
    """
    if isinstance(exc, FileNotFoundError):
        return True

    # storage3-specific check, wrapped so the helper stays importable
    # if storage3's layout changes.
    try:
        from storage3.utils import StorageException
        if isinstance(exc, StorageException):
            status = getattr(exc, 'status', None)
            if status in (404, '404'):
                return True
    except ImportError:
        pass

    # Defence in depth: substring match on exception message.
    msg = str(exc).lower()
    return any(token in msg for token in ('not found', 'no such', 'object not found'))
