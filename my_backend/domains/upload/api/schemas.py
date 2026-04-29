"""
Marshmallow schemas for upload endpoint JSON payloads.

Each schema mirrors the shape that a specific handler in
``load_data.py`` consumes. Validation runs at the top of the handler;
rejection produces a 400 response with the malformed-fields list.
The schema's ``load()`` returns a plain ``dict`` so existing handler
code that does ``data['key']`` / ``data.get('key')`` keeps working.

We deliberately do not widen any field constraint to be permissive --
better to reject and have the client correct the payload than to
silently accept malformed shapes that crash downstream code as 500.
"""
from marshmallow import Schema, fields, validate

# Mirrors MAX_CONTENT_LENGTH (100MB) for any JSON-encoded byte-size
# bounded fields. Per-handler caps tighten this when known.
MAX_TOTAL_BYTES = 100 * 1024 * 1024

# Conservative caps to keep payloads bounded.
MAX_FILE_NAME_LEN = 512
MAX_FILE_ID_LEN = 512
MAX_FILE_IDS = 1000
MAX_PREPARE_SAVE_ROWS = 10_000_000  # rows in the embedded CSV-as-list
MAX_PREPARE_SAVE_COLS = 1024
MAX_CELL_LEN = 100_000  # individual cell string length cap


class UploadIdSchema(Schema):
    """Shape used by /finalize-upload and /cancel-upload.

    Both handlers only read ``uploadId`` from the body.
    """
    uploadId = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=256),
    )


class PrepareSaveSchema(Schema):
    """Shape used by /prepare-save.

    Wire format observed in the handler:
        {"data": {"data": [[...row...], ...], "fileName": "..."}}

    The outer ``data`` key holds a wrapper object; the wrapper carries
    the actual rows under another ``data`` key plus an optional
    ``fileName``. Each row is a list of CSV cells (strings/numbers).
    """

    class _Wrapper(Schema):
        data = fields.List(
            fields.List(
                fields.Raw(),
                validate=validate.Length(max=MAX_PREPARE_SAVE_COLS),
            ),
            required=True,
            validate=validate.Length(min=1, max=MAX_PREPARE_SAVE_ROWS),
        )
        fileName = fields.Str(
            required=False,
            load_default="",
            validate=validate.Length(max=MAX_FILE_NAME_LEN),
        )

    data = fields.Nested(_Wrapper, required=True)


class MergeAndPrepareSchema(Schema):
    """Shape used by /merge-and-prepare.

    Reads ``fileIds`` (list of file_id strings) and an optional
    ``fileName`` for the merged output. The handler accepts a
    single-element list as a no-op short-circuit, but still requires
    a non-empty list.
    """
    fileIds = fields.List(
        fields.Str(validate=validate.Length(min=1, max=MAX_FILE_ID_LEN)),
        required=True,
        validate=validate.Length(min=1, max=MAX_FILE_IDS),
    )
    fileName = fields.Str(
        required=False,
        load_default="merged_data.csv",
        validate=validate.Length(min=1, max=MAX_FILE_NAME_LEN),
    )


class CleanupFilesSchema(Schema):
    """Shape used by /cleanup-files.

    Only consumes ``fileIds``. The current handler treats an empty
    list as a no-op success, so we allow ``min=0`` here -- this is the
    one place where permissiveness matches existing handler intent.
    """
    fileIds = fields.List(
        fields.Str(validate=validate.Length(min=1, max=MAX_FILE_ID_LEN)),
        required=False,
        load_default=list,
        validate=validate.Length(max=MAX_FILE_IDS),
    )
