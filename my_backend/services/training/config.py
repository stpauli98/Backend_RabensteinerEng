"""
Configuration module for training system
Contains all configuration classes and constants extracted from training_backend_test_2.py
EXACT COPY from original file to preserve functionality
"""

import datetime


class MTS:
    """
    Multivariate Time Series configuration class
    Extracted from training_backend_test_2.py lines 619-632
    """
    
    def __init__(self):
        # Anzahl der Zeitschritte der Eingabedaten
        self.I_N  = 13
        
        # Anzahl der Zeitschritte der Ausgabedaten
        self.O_N  = 13
        
        # Zeitschrittweite für die Bildung der finalen Datensätze [min]
        self.DELT = 3
        
        # Offset für die Bildung der finalen Datensätze [min]
        self.OFST = 0
        
        # Additional attributes for DataProcessor compatibility
        self.timezone = 'UTC'
        self.use_time_features = True
        self.interpolation = True
        self.outlier_removal = False
        self.scaling = True


class T:
    """
    Time features configuration class
    Extracted from training_backend_test_2.py lines 798-954
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
        
        # Zeitschrittweite [min]
        DELT = (TH_END-TH_STRT)*60/(13-1)  # Using default I_N=13
        
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
        DELT = (TH_END-TH_STRT)*60/(13-1)  # Using default I_N=13
        
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
        DELT = (TH_END-TH_STRT)*60/(13-1)  # Using default I_N=13
    
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
        DELT = (TH_END-TH_STRT)*60/(13-1)  # Using default I_N=13
    
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
        DELT = (TH_END-TH_STRT)*60/(13-1)  # Using default I_N=13
    
    # Zeitzone
    TZ      = "Europe/Vienna"


class MDL:
    """
    Model configuration class EXACTLY as in training_original.py lines 2046-2141
    Sets parameters conditionally based on MODE
    """
    
    def __init__(self, mode: str = "LIN"):
        # Set MODE (default is "LIN" as in original line 2055)
        self.MODE = mode
        
        # Initialize parameters based on MODE (exactly as in original)
        if self.MODE == "Dense":
            # Lines 2057-2069
            self.LAY = 3         # Anzahl an Layer
            self.N = 512         # Anzahl der Neuronen pro Layer
            self.EP = 20         # Max. Anzahl der Trainingsdurchläufe
            self.ACTF = "ReLU"   # Aktivierungsfunktion
            
        elif self.MODE == "CNN":
            # Lines 2071-2086
            self.LAY = 3         # Anzahl an Layer
            self.N = 512         # Anzahl der Filter pro Layer
            self.K = 3           # Kernelgröße
            self.EP = 20         # Max. Anzahl der Trainingsdurchläufe
            self.ACTF = "ReLU"   # Aktivierungsfunktion
            
        elif self.MODE == "LSTM":
            # Lines 2089-2101
            self.LAY = 3         # Anzahl an Layer
            self.N = 512         # Anzahl der Neuronen pro Layer
            self.EP = 20         # Max. Anzahl der Trainingsdurchläufe
            self.ACTF = "ReLU"   # Aktivierungsfunktion
            
        elif self.MODE == "AR LSTM":
            # Lines 2103-2115
            self.LAY = 3         # Anzahl an Layer
            self.N = 512         # Anzahl der Neuronen pro Layer
            self.EP = 20         # Max. Anzahl der Trainingsdurchläufe
            self.ACTF = "ReLU"   # Aktivierungsfunktion
            
        elif self.MODE == "SVR_dir":
            # Lines 2117-2126
            self.KERNEL = "poly" # Art der Modellierung von Nichtlinearitäten
            self.C = 1           # Regulationsparameter (Trade-off Bias vs. Varianz)
            self.EPSILON = 0.1   # Toleranz für Abweichungen
            
        elif self.MODE == "SVR_MIMO":
            # Lines 2128-2137
            self.KERNEL = "poly" # Art der Modellierung von Nichtlinearitäten
            self.C = 1           # Regulationsparameter (Trade-off Bias vs. Varianz)
            self.EPSILON = 0.1   # Toleranz für Abweichungen
            
        elif self.MODE == "LIN":
            # Line 2139-2140
            pass  # No specific parameters for LIN mode


# INFORMATIONEN ZU DEN FEIERTAGEN (DIE KEINE SONNTAGE SIND)
# Extracted from training_backend_test_2.py lines 635-692
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

# Convert holiday strings to datetime objects (exactly as in original)
HOL = {
    land: [datetime.datetime.strptime(datum, "%Y-%m-%d") for datum in daten]
    for land, daten in HOL.items()
}


# PLOT SETTINGS ################################################################
# Visualization configuration for matplotlib and seaborn plots
PLOT_SETTINGS = {
    'figure_size': (12, 8),
    'dpi': 100,
    'font_size': 12,
    'color_palette': 'tab20',
    'violin_plot': {
        'inner': 'quartile',
        'linewidth': 1.5
    },
    'line_plot': {
        'linewidth': 1,
        'markersize': 2,
        'marker_style': 'o',
        'forecast_marker': 'x',
        'forecast_linestyle': '--',
        'forecast_markersize': 4
    },
    'subplot_layout': {
        'constrained': True,
        'wspace': 0.3,
        'hspace': 0.4
    }
}