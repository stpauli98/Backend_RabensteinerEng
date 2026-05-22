"""
Marshmallow schemas for first_processing endpoint JSON payloads.

Domain-local schemas. We deliberately mirror the caps defined in
domains.upload.api.schemas (MAX_FILE_IDS=1000, MAX_FILE_ID_LEN=512,
MAX_FILE_NAME_LEN=512, MAX_PREPARE_SAVE_ROWS=10M, MAX_PREPARE_SAVE_COLS=1024)
so the two pipelines have identical guardrails. We import these caps
from the upload schemas module rather than re-declaring them so a future
tightening in one place propagates.
"""
from marshmallow import Schema, fields, validate

from domains.upload.api.schemas import (
    MAX_FILE_ID_LEN,
    MAX_FILE_IDS,
    MAX_FILE_NAME_LEN,
    MAX_PREPARE_SAVE_ROWS,
    MAX_PREPARE_SAVE_COLS,
)


class FirstProcessingCleanupSchema(Schema):
    """Shape used by /api/firstProcessing/cleanup-files.

    Mirrors W6 CleanupFilesSchema: optional fileIds list of strings,
    each 1-512 chars, list capped at MAX_FILE_IDS (1000) entries.
    """
    fileIds = fields.List(
        fields.Str(validate=validate.Length(min=1, max=MAX_FILE_ID_LEN)),
        required=False,
        load_default=list,
        validate=validate.Length(max=MAX_FILE_IDS),
    )


class FirstProcessingPrepareSaveSchema(Schema):
    """Shape used by /api/firstProcessing/prepare-save.

    Wire format:
        {"data": {"data": [[...row...], ...], "fileName": "..."}}
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
