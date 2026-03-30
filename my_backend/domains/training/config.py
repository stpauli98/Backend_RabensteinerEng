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
        # Fallback defaults: 0,0,0,0 when zeitschritte not in database
        # Actual values should come from zeitschritte table via session_data
        self.I_N  = 0   # Input timesteps (eingabe)
        
        self.O_N  = 0   # Output timesteps (ausgabe)
        
        self.DELT = 0   # Timestep width in minutes (zeitschrittweite)
        
        self.OFST = 0   # Offset (offset)
        
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
    
    class Y:
        
        IMP = False
        
        LT = False
        
        SPEC = "Zeithorizont"
        
        TH_STRT = -24
        
        TH_END = 0
        
        SCAL = True
        
        SCAL_MAX = 1
        
        SCAL_MIN = 0
        
        DELT = (TH_END-TH_STRT)*60/(13-1)
        
    class M:
        
        IMP = False
        
        LT = False
        
        SPEC = "Zeithorizont"
        
        TH_STRT = -1
        
        TH_END = 0
        
        SCAL = True
        
        SCAL_MAX = 1
        
        SCAL_MIN = 0
        
        DELT = (TH_END-TH_STRT)*60/(13-1)
        
    class W:

        IMP = False
        
        LT = False
        
        SPEC = "Aktuelle Zeit"
        
        TH_STRT = -24
        
        TH_END = 0
        
        SCAL = True
        
        SCAL_MAX = 1
        
        SCAL_MIN = 0
        
        DELT = (TH_END-TH_STRT)*60/(13-1)
    
    class D:

        IMP = False
        
        LT = False
        
        SPEC = "Zeithorizont"
        
        TH_STRT = -24
        
        TH_END = 0
        
        SCAL = True
        
        SCAL_MAX = 1
        
        SCAL_MIN = 0
        
        DELT = (TH_END-TH_STRT)*60/(13-1)
    
    class H:

        IMP = False
        
        LT = False
        
        SPEC = "Aktuelle Zeit"
        
        TH_STRT = -100
        
        TH_END = 0
        
        SCAL = True
        
        SCAL_MAX = 1
        
        SCAL_MIN = 0
        
        CNTRY   = "Österreich"
        
        DELT = (TH_END-TH_STRT)*60/(13-1)
    
    TZ      = "Europe/Vienna"


class MDL:
    """
    Model configuration class EXACTLY as in training_original.py lines 2046-2141
    Sets parameters conditionally based on MODE
    """
    
    def __init__(self, mode: str = "LIN"):
        self.MODE = mode
        
        if self.MODE == "Dense":
            self.LAY = 3
            self.N = 512
            self.EP = 20
            self.ACTF = "ReLU"
            
        elif self.MODE == "CNN":
            self.LAY = 3
            self.N = 512
            self.K = 3
            self.EP = 20
            self.ACTF = "ReLU"
            
        elif self.MODE == "LSTM":
            self.LAY = 3
            self.N = 512
            self.EP = 20
            self.ACTF = "ReLU"
            
        elif self.MODE == "AR LSTM":
            self.LAY = 3
            self.N = 512
            self.EP = 20
            self.ACTF = "ReLU"
            
        elif self.MODE == "SVR_dir":
            self.KERNEL = "poly"
            self.C = 1
            self.EPSILON = 0.1
            
        elif self.MODE == "SVR_MIMO":
            self.KERNEL = "poly"
            self.C = 1
            self.EPSILON = 0.1

        elif self.MODE == "LGBMR":
            self.N_ESTIMATORS = 100
            self.LEARNING_RATE = 0.1
            self.MAX_DEPTH = -1

        elif self.MODE == "LIN":
            pass


###############################################################################
# GESETZLICHE FEIERTAGE 2020–2030
# Keine Berücksichtigung von Feiertagen vor 2020.
#
# Easter Sunday dates used for moveable holidays:
# 2020: Apr 12, 2021: Apr 4, 2022: Apr 17, 2023: Apr 9, 2024: Mar 31,
# 2025: Apr 20, 2026: Apr 5, 2027: Mar 28, 2028: Apr 16, 2029: Apr 1, 2030: Apr 21
#
# AT fixed: Jan 1, Jan 6, May 1, Aug 15, Oct 26, Nov 1, Dec 8, Dec 25, Dec 26
# AT moveable: Ostermontag(+1), Christi Himmelfahrt(+39), Pfingstmontag(+50), Fronleichnam(+60)
#
# DE federal: Jan 1, May 1, Oct 3, Dec 25, Dec 26
# DE moveable: Karfreitag(-2), Ostermontag(+1), Chr. Himmelfahrt(+39), Pfingstmontag(+50)
#
# CH national: Jan 1, Aug 1, Dec 25
# CH moveable: Karfreitag(-2), Ostermontag(+1), Auffahrt(+39), Pfingstmontag(+50)
###############################################################################

HOL = {
    "Österreich": [
        # 2020 (Easter: Apr 12)
        "2020-01-01", "2020-01-06", "2020-04-13", "2020-05-01",
        "2020-05-21", "2020-06-01", "2020-06-11", "2020-08-15",
        "2020-10-26", "2020-11-01", "2020-12-08", "2020-12-25", "2020-12-26",
        # 2021 (Easter: Apr 4)
        "2021-01-01", "2021-01-06", "2021-04-05", "2021-05-01",
        "2021-05-13", "2021-05-24", "2021-06-03", "2021-08-15",
        "2021-10-26", "2021-11-01", "2021-12-08", "2021-12-25", "2021-12-26",
        # 2022 (Easter: Apr 17)
        "2022-01-01", "2022-01-06", "2022-04-18", "2022-05-01",
        "2022-05-26", "2022-06-06", "2022-06-16", "2022-08-15",
        "2022-10-26", "2022-11-01", "2022-12-08", "2022-12-25", "2022-12-26",
        # 2023 (Easter: Apr 9)
        "2023-01-01", "2023-01-06", "2023-04-10", "2023-05-01",
        "2023-05-18", "2023-05-29", "2023-06-08", "2023-08-15",
        "2023-10-26", "2023-11-01", "2023-12-08", "2023-12-25", "2023-12-26",
        # 2024 (Easter: Mar 31)
        "2024-01-01", "2024-01-06", "2024-04-01", "2024-05-01",
        "2024-05-09", "2024-05-20", "2024-05-30", "2024-08-15",
        "2024-10-26", "2024-11-01", "2024-12-08", "2024-12-25", "2024-12-26",
        # 2025 (Easter: Apr 20)
        "2025-01-01", "2025-01-06", "2025-04-21", "2025-05-01",
        "2025-05-29", "2025-06-09", "2025-06-19", "2025-08-15",
        "2025-10-26", "2025-11-01", "2025-12-08", "2025-12-25", "2025-12-26",
        # 2026 (Easter: Apr 5)
        "2026-01-01", "2026-01-06", "2026-04-06", "2026-05-01",
        "2026-05-14", "2026-05-25", "2026-06-04", "2026-08-15",
        "2026-10-26", "2026-11-01", "2026-12-08", "2026-12-25", "2026-12-26",
        # 2027 (Easter: Mar 28)
        "2027-01-01", "2027-01-06", "2027-03-29", "2027-05-01",
        "2027-05-06", "2027-05-17", "2027-05-27", "2027-08-15",
        "2027-10-26", "2027-11-01", "2027-12-08", "2027-12-25", "2027-12-26",
        # 2028 (Easter: Apr 16)
        "2028-01-01", "2028-01-06", "2028-04-17", "2028-05-01",
        "2028-05-25", "2028-06-05", "2028-06-15", "2028-08-15",
        "2028-10-26", "2028-11-01", "2028-12-08", "2028-12-25", "2028-12-26",
        # 2029 (Easter: Apr 1)
        "2029-01-01", "2029-01-06", "2029-04-02", "2029-05-01",
        "2029-05-10", "2029-05-21", "2029-05-31", "2029-08-15",
        "2029-10-26", "2029-11-01", "2029-12-08", "2029-12-25", "2029-12-26",
        # 2030 (Easter: Apr 21)
        "2030-01-01", "2030-01-06", "2030-04-22", "2030-05-01",
        "2030-05-30", "2030-06-10", "2030-06-20", "2030-08-15",
        "2030-10-26", "2030-11-01", "2030-12-08", "2030-12-25", "2030-12-26",
    ],
    "Deutschland": [
        # Bundesweit einheitliche Feiertage
        # 2020 (Easter: Apr 12)
        "2020-01-01", "2020-04-10", "2020-04-13", "2020-05-01",
        "2020-05-21", "2020-06-01", "2020-10-03", "2020-12-25", "2020-12-26",
        # 2021 (Easter: Apr 4)
        "2021-01-01", "2021-04-02", "2021-04-05", "2021-05-01",
        "2021-05-13", "2021-05-24", "2021-10-03", "2021-12-25", "2021-12-26",
        # 2022 (Easter: Apr 17)
        "2022-01-01", "2022-04-15", "2022-04-18", "2022-05-01",
        "2022-05-26", "2022-06-06", "2022-10-03", "2022-12-25", "2022-12-26",
        # 2023 (Easter: Apr 9)
        "2023-01-01", "2023-04-07", "2023-04-10", "2023-05-01",
        "2023-05-18", "2023-05-29", "2023-10-03", "2023-12-25", "2023-12-26",
        # 2024 (Easter: Mar 31)
        "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-01",
        "2024-05-09", "2024-05-20", "2024-10-03", "2024-12-25", "2024-12-26",
        # 2025 (Easter: Apr 20)
        "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-01",
        "2025-05-29", "2025-06-09", "2025-10-03", "2025-12-25", "2025-12-26",
        # 2026 (Easter: Apr 5)
        "2026-01-01", "2026-04-03", "2026-04-06", "2026-05-01",
        "2026-05-14", "2026-05-25", "2026-10-03", "2026-12-25", "2026-12-26",
        # 2027 (Easter: Mar 28)
        "2027-01-01", "2027-03-26", "2027-03-29", "2027-05-01",
        "2027-05-06", "2027-05-17", "2027-10-03", "2027-12-25", "2027-12-26",
        # 2028 (Easter: Apr 16)
        "2028-01-01", "2028-04-14", "2028-04-17", "2028-05-01",
        "2028-05-25", "2028-06-05", "2028-10-03", "2028-12-25", "2028-12-26",
        # 2029 (Easter: Apr 1)
        "2029-01-01", "2029-03-30", "2029-04-02", "2029-05-01",
        "2029-05-10", "2029-05-21", "2029-10-03", "2029-12-25", "2029-12-26",
        # 2030 (Easter: Apr 21)
        "2030-01-01", "2030-04-19", "2030-04-22", "2030-05-01",
        "2030-05-30", "2030-06-10", "2030-10-03", "2030-12-25", "2030-12-26",
    ],
    "Schweiz": [
        # National anerkannte Feiertage
        # 2020 (Easter: Apr 12)
        "2020-01-01", "2020-04-10", "2020-04-13",
        "2020-05-21", "2020-06-01", "2020-08-01", "2020-12-25",
        # 2021 (Easter: Apr 4)
        "2021-01-01", "2021-04-02", "2021-04-05",
        "2021-05-13", "2021-05-24", "2021-08-01", "2021-12-25",
        # 2022 (Easter: Apr 17)
        "2022-01-01", "2022-04-15", "2022-04-18",
        "2022-05-26", "2022-06-06", "2022-08-01", "2022-12-25",
        # 2023 (Easter: Apr 9)
        "2023-01-01", "2023-04-07", "2023-04-10",
        "2023-05-18", "2023-05-29", "2023-08-01", "2023-12-25",
        # 2024 (Easter: Mar 31)
        "2024-01-01", "2024-03-29", "2024-04-01",
        "2024-05-09", "2024-05-20", "2024-08-01", "2024-12-25",
        # 2025 (Easter: Apr 20)
        "2025-01-01", "2025-04-18", "2025-04-21",
        "2025-05-29", "2025-06-09", "2025-08-01", "2025-12-25",
        # 2026 (Easter: Apr 5)
        "2026-01-01", "2026-04-03", "2026-04-06",
        "2026-05-14", "2026-05-25", "2026-08-01", "2026-12-25",
        # 2027 (Easter: Mar 28)
        "2027-01-01", "2027-03-26", "2027-03-29",
        "2027-05-06", "2027-05-17", "2027-08-01", "2027-12-25",
        # 2028 (Easter: Apr 16)
        "2028-01-01", "2028-04-14", "2028-04-17",
        "2028-05-25", "2028-06-05", "2028-08-01", "2028-12-25",
        # 2029 (Easter: Apr 1)
        "2029-01-01", "2029-03-30", "2029-04-02",
        "2029-05-10", "2029-05-21", "2029-08-01", "2029-12-25",
        # 2030 (Easter: Apr 21)
        "2030-01-01", "2030-04-19", "2030-04-22",
        "2030-05-30", "2030-06-10", "2030-08-01", "2030-12-25",
    ],
}

HOL = {
    land: [datetime.datetime.strptime(datum, "%Y-%m-%d") for datum in daten]
    for land, daten in HOL.items()
}


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
