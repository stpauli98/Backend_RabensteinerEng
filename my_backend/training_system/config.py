"""
Configuration module for training system
Contains all configuration classes and constants extracted from training_backend_test_2.py
EXACT COPY from original file to preserve functionality
"""

import datetime


class MTS:
    """
    Multivariate Time Series configuration class
    Extracted EXACTLY from training_backend_test_2.py lines 619-632
    """
    
    # Anzahl der Zeitschritte der Eingabedaten
    I_N  = 13
    
    # Anzahl der Zeitschritte der Ausgabedaten
    O_N  = 13
    
    # Zeitschrittweite für die Bildung der finalen Datensätze [min]
    DELT = 3
    
    # Offset für die Bildung der finalen Datensätze [min]
    OFST = 0


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
    Model configuration class - Dynamic configuration from user input
    Receives parameters from frontend ModelConfiguration.tsx component
    """
    
    def __init__(self, user_config: dict = None):
        """
        Initialize model configuration from user input
        
        Args:
            user_config: Dictionary containing user-selected model parameters
                        from frontend ModelConfiguration component
        """
        if user_config:
            # Use user-provided configuration
            self.MODE = user_config.get('MODE', 'LIN')
            self.LAY = user_config.get('LAY')
            self.N = user_config.get('N') 
            self.EP = user_config.get('EP')
            self.ACTF = user_config.get('ACTF')
            self.K = user_config.get('K')  # CNN kernel size
            self.KERNEL = user_config.get('KERNEL')  # SVR kernel type
            self.C = user_config.get('C')  # SVR C parameter
            self.EPSILON = user_config.get('EPSILON')  # SVR epsilon
        else:
            # Default fallback configuration (for backwards compatibility)
            self.MODE = "LIN"
            self.LAY = None
            self.N = None
            self.EP = None
            self.ACTF = None
            self.K = None
            self.KERNEL = None
            self.C = None
            self.EPSILON = None
    
    def validate_config(self) -> bool:
        """
        Validate that all required parameters are provided for the selected model
        Returns True if valid, False otherwise
        """
        if self.MODE in ['Dense', 'CNN', 'LSTM', 'AR LSTM']:
            # Neural network models require LAY, N, EP, ACTF
            required = [self.LAY, self.N, self.EP, self.ACTF]
            if self.MODE == 'CNN':
                required.append(self.K)  # CNN also needs kernel size
            return all(param is not None and param != '' for param in required)
        
        elif self.MODE in ['SVR_dir', 'SVR_MIMO']:
            # SVR models require KERNEL, C, EPSILON
            required = [self.KERNEL, self.C, self.EPSILON]
            return all(param is not None and param != '' for param in required)
        
        elif self.MODE == 'LIN':
            # Linear model has minimal requirements
            return True
        
        return False
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary format"""
        return {
            'MODE': self.MODE,
            'LAY': self.LAY,
            'N': self.N,
            'EP': self.EP,
            'ACTF': self.ACTF,
            'K': self.K,
            'KERNEL': self.KERNEL,
            'C': self.C,
            'EPSILON': self.EPSILON
        }
    
    @classmethod
    def get_default_for_mode(cls, mode: str) -> 'MDL':
        """
        Get default configuration for a specific model type
        Used as fallback when user doesn't provide complete config
        """
        defaults = {
            'Dense': {'MODE': mode, 'LAY': 3, 'N': 512, 'EP': 20, 'ACTF': 'ReLU'},
            'CNN': {'MODE': mode, 'LAY': 3, 'N': 512, 'K': 3, 'EP': 20, 'ACTF': 'ReLU'},
            'LSTM': {'MODE': mode, 'LAY': 3, 'N': 512, 'EP': 20, 'ACTF': 'ReLU'},
            'AR LSTM': {'MODE': mode, 'LAY': 3, 'N': 512, 'EP': 20, 'ACTF': 'ReLU'},
            'SVR_dir': {'MODE': mode, 'KERNEL': 'poly', 'C': 1, 'EPSILON': 0.1},
            'SVR_MIMO': {'MODE': mode, 'KERNEL': 'poly', 'C': 1, 'EPSILON': 0.1},
            'LIN': {'MODE': mode}
        }
        
        return cls(defaults.get(mode, {'MODE': 'LIN'}))


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