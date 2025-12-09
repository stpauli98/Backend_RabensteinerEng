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
        self.I_N  = 13
        
        self.O_N  = 13
        
        self.DELT = 3
        
        self.OFST = 0
        
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
        
        LT = True
        
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
            
        elif self.MODE == "LIN":
            pass


HOL = {
        "Österreich": [
            "2022-01-01",
            "2022-01-06",
            "2022-04-18",
            "2022-05-26",
            "2022-06-06",
            "2022-06-16",
            "2022-08-15",
            "2022-10-26",
            "2022-11-01",
            "2022-12-08",
            "2022-12-26",
            "2023-01-06",
            "2023-04-10",
            "2023-05-01",
            "2023-05-18",
            "2023-05-29",
            "2023-06-08",
            "2023-08-15",
            "2023-10-26",
            "2023-11-01",
            "2023-12-08",
            "2023-12-25",
            "2023-12-26",
            "2024-01-01",
            "2024-01-06",
            "2024-04-01",
            "2024-05-01",
            "2024-05-09",
            "2025-05-20",
            "2024-05-30",
            "2024-08-15",
            "2024-10-26",
            "2024-11-01",
            "2024-12-25",
            "2024-12-26",
            "2025-01-01",
            "2025-01-06",
            "2025-04-21",
            "2025-05-01",
            "2025-05-29",
            "2025-06-09",
            "2025-06-19",
            "2025-08-15",
            "2025-11-01",
            "2025-12-08",
            "2025-12-25",
            "2025-12-26"
        ],
        "Deutschland":  [],
        "Schweiz":      []
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
