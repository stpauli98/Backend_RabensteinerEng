"""
Time Features (T) konfiguracija - Vremenske značajke
Ekstrahirano iz training_original.py linije 798-954

Sadrži konfiguraciju za cikličke vremenske značajke:
- Y: Godišnja sinus/kosinus komponenta
- M: Mjesečna sinus/kosinus komponenta
- W: Tjedna sinus/kosinus komponenta
- D: Dnevna sinus/kosinus komponenta
- H: Praznici
"""

from .mts import MTS

###############################################################################
# ZEITINFORMATION #############################################################
###############################################################################

class T:

    # Jahreszeitliche Sinus-/Cosinus-Komponente
    class Y:

        # Anwendung
        IMP = False

        # Bezug auf lokale Zeit (Winter-/Sommerzeit)
        LT = False

        # Datenform
        SPEC = "Zeithorizont"

        # Zeithorizont Start [h]
        TH_STRT = -24

        # Zeithorizont Ende [h]
        TH_END = 0

        # Skalierung
        SCAL = True

        # Skalierung max [-]
        SCAL_MAX = 1

        # Skalierung min [-]
        SCAL_MIN = 0

        # Zeitschrittweite [min]
        DELT = (TH_END-TH_STRT)*60/(MTS.I_N-1)

    # Monatliche Sinus-/Cosinus-Komponente
    class M:

        # Anwendung
        IMP = False

        # Bezug auf lokale Zeit (Winter-/Sommerzeit)
        LT = False

        # Datenform
        SPEC = "Zeithorizont"

        # Zeithorizont Start [h]
        TH_STRT = -1

        # Zeithorizont Ende [h]
        TH_END = 0

        # Skalierung
        SCAL = True

        # Skalierung max [-]
        SCAL_MAX = 1

        # Skalierung min [-]
        SCAL_MIN = 0

        # Zeitschrittweite [min]
        DELT = (TH_END-TH_STRT)*60/(MTS.I_N-1)

    # Wöchentliche Sinus-/Cosinus-Komponente
    class W:

        # Anwendung
        IMP = False

        # Bezug auf lokale Zeit (Winter-/Sommerzeit)
        LT = False

        # Datenform
        SPEC = "Aktuelle Zeit"

        # Zeithorizont Start [h]
        TH_STRT = -24

        # Zeithorizont Ende [h]
        TH_END = 0

        # Skalierung
        SCAL = True

        # Skalierung max [-]
        SCAL_MAX = 1

        # Skalierung min [-]
        SCAL_MIN = 0

        # Zeitschrittweite [min]
        DELT = (TH_END-TH_STRT)*60/(MTS.I_N-1)

    # Tägliche Sinus-/Cosinus-Komponente
    class D:

        # Anwendung
        IMP = False

        # Bezug auf lokale Zeit (Winter-/Sommerzeit)
        LT = True

        # Datenform
        SPEC = "Zeithorizont"

        # Zeithorizont Start [h]
        TH_STRT = -24

        # Zeithorizont Ende [h]
        TH_END = 0

        # Skalierung
        SCAL = True

        # Skalierung max [-]
        SCAL_MAX = 1

        # Skalierung min [-]
        SCAL_MIN = 0

        # Zeitschrittweite [min]
        DELT = (TH_END-TH_STRT)*60/(MTS.I_N-1)

    # Berücksichtigung von Feiertagen
    class H:

        # Anwendung
        IMP = False

        # Bezug auf lokale Zeit (Winter-/Sommerzeit)
        LT = False

        # Datenform
        SPEC = "Aktuelle Zeit"

        # Zeithorizont Start [h]
        TH_STRT = -100

        # Zeithorizont Ende [h]
        TH_END = 0

        # Skalierung
        SCAL = True

        # Skalierung max [-]
        SCAL_MAX = 1

        # Skalierung min [-]
        SCAL_MIN = 0

        # Land
        CNTRY   = "Österreich"

        # Zeitschrittweite [min]
        DELT = (TH_END-TH_STRT)*60/(MTS.I_N-1)

    # Zeitzone
    TZ      = "Europe/Vienna"
