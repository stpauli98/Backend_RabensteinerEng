"""
CSV injection sanitizer for cells exported via csv.writer.

Spreadsheet apps (Excel, Sheets, Numbers) interpret cell content beginning
with =, +, -, @ as a formula. Whitespace before those chars is stripped by
the interpreter, so cells like "  =SUM(A1)" are equally dangerous.

We prefix such cells with a single quote so the spreadsheet treats them as
literal text. Non-string values (numbers, None, bool) pass through unchanged
— csv.writer handles their formatting.
"""

_FORMULA_CHARS = ("=", "+", "-", "@")
_WHITESPACE = (" ", "\t", "\r", "\n")


def csv_safe_cell(value):
    """
    Neutralize CSV-formula-injection chars in a cell value.

    If value is a string whose first non-whitespace character is one of
    = + - @, return "'" + value. Otherwise return value unchanged.
    """
    if not isinstance(value, str) or not value:
        return value

    # Find first non-whitespace char
    for ch in value:
        if ch in _WHITESPACE:
            continue
        if ch in _FORMULA_CHARS:
            return "'" + value
        return value

    # All-whitespace string is safe (no formula char ever encountered)
    return value
