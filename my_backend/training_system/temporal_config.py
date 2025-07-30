"""
Temporal features configuration
Contains the T class structure extracted from training_backend_test_2.py
This preserves the exact temporal features logic from the original code
"""

class T:
    """
    Temporal features configuration class
    Extracted from training_backend_test_2.py lines 798-949
    EXACT COPY from original code
    """
    
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
        
        # Zeitschrittweite [min] - Note: MTS.I_N would need to be defined
        # DELT = (TH_END-TH_STRT)*60/(MTS.I_N-1)
        
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
        
        # Zeitschrittweite [min] - Note: MTS.I_N would need to be defined
        # DELT = (TH_END-TH_STRT)*60/(MTS.I_N-1)
        
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
        
        # Zeitschrittweite [min] - Note: MTS.I_N would need to be defined
        # DELT = (TH_END-TH_STRT)*60/(MTS.I_N-1)
    
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
        
        # Zeitschrittweite [min] - Note: MTS.I_N would need to be defined
        # DELT = (TH_END-TH_STRT)*60/(MTS.I_N-1)
    
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
        CNTRY = "Österreich"


# Create default instance that can be imported
temporal_config = T()