"""middleman honors the per-file 'average over horizon' flag instead of forcing
avg=False. The UI toggle persists 'mittelwertbildung_uber_den_zeithorizont' =
'ja'|'nein'; avg_from_metadata maps it to the bool the transformer consumes.
"""
from domains.training.services.middleman import avg_from_metadata


def test_ja_enables_averaging():
    assert avg_from_metadata({"mittelwertbildung_uber_den_zeithorizont": "ja"}) is True


def test_nein_disables_averaging():
    assert avg_from_metadata({"mittelwertbildung_uber_den_zeithorizont": "nein"}) is False


def test_missing_defaults_to_false():
    assert avg_from_metadata({}) is False


def test_case_and_whitespace_insensitive():
    assert avg_from_metadata({"mittelwertbildung_uber_den_zeithorizont": " JA "}) is True
    assert avg_from_metadata({"mittelwertbildung_uber_den_zeithorizont": "Nein"}) is False


def test_none_value_defaults_to_false():
    assert avg_from_metadata({"mittelwertbildung_uber_den_zeithorizont": None}) is False
