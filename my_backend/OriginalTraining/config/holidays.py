"""
Holidays (HOL) konfiguracija - Praznici po zemljama
Ekstrahirano iz training_original.py linije 633-692

Sadrži datume praznika za Austriju, Njemačku i Švicarsku.
Datumi se pretvaraju iz string formata u datetime objekte.
"""

import datetime

###############################################################################
# INFORMATIONEN ZU DEN FEIERTAGEN (DIE KEINE SONNTAGE SIND) ###################

HOL = {
        "Österreich": [
            "2022-01-01",   # Neujahrstag (SA)
            "2022-01-06",   # Heilige Drei Könige (DO)
            "2022-04-18",   # Ostermontag (MO)
            "2022-05-26",   # Christi Himmelfahrt (DO)
            "2022-06-06",   # Pfingsmontag (MO)
            "2022-06-16",   # Fronleichnam (DO)
            "2022-08-15",   # Mariä Himmelfahrt (MO)
            "2022-10-26",   # Nationalfeiertag (MI)
            "2022-11-01",   # Allerheiligen (DI)
            "2022-12-08",   # Mariä Empfängnis (DO)
            "2022-12-26",   # Stefanitag (MO)
            "2023-01-06",   # Heilige Drei Könige (FR)
            "2023-04-10",   # Ostermontag (MO)
            "2023-05-01",   # Tag der Arbeit (MO)
            "2023-05-18",   # Christi Himmelfahrt (DO)
            "2023-05-29",   # Pfingsmontag (MO)
            "2023-06-08",   # Fronleichnam (DO)
            "2023-08-15",   # Mariä Himmelfahrt (DI)
            "2023-10-26",   # Nationalfeiertag (DO)
            "2023-11-01",   # Allerheiligen (MI)
            "2023-12-08",   # Mariä Empfängnis (FR)
            "2023-12-25",   # Christtag (MO)
            "2023-12-26",   # Stefanitag (DI)
            "2024-01-01",   # Neujahrstag (MO)
            "2024-01-06",   # Heilige Drei Könige (SA)
            "2024-04-01",   # Ostermontag (MO)
            "2024-05-01",   # Tag der Arbeit (MI)
            "2024-05-09",   # Christi Himmelfahrt (DO)
            "2025-05-20",   # Pfingsmontag (MO)
            "2024-05-30",   # Fronleichnam (DO)
            "2024-08-15",   # Mariä Himmelfahrt (DO)
            "2024-10-26",   # Nationalfeiertag (SA)
            "2024-11-01",   # Allerheiligen (FR)
            "2024-12-25",   # Christtag (MI)
            "2024-12-26",   # Stefanitag (DO)
            "2025-01-01",   # Neujahrstag (MI)
            "2025-01-06",   # Heilige Drei Könige (MO)
            "2025-04-21",   # Ostermontag (MO)
            "2025-05-01",   # Tag der Arbeit (DO)
            "2025-05-29",   # Christi Himmelfahrt (DO)
            "2025-06-09",   # Pfingsmontag (MO)
            "2025-06-19",   # Fronleichnam (DO)
            "2025-08-15",   # Mariä Himmelfahrt (FR)
            "2025-11-01",   # Allerheiligen (SA)
            "2025-12-08",   # Mariä Empfängnis (MO)
            "2025-12-25",   # Christtag (DO)
            "2025-12-26"    # Stefanitag (FR)
        ],
        "Deutschland":  [],
        "Schweiz":      []
        }

# Konvertiranje stringova u datetime objekte
HOL = {
    land: [datetime.datetime.strptime(datum, "%Y-%m-%d") for datum in daten]
    for land, daten in HOL.items()
}
