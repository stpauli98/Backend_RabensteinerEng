"""
Model (MDL) konfiguracija - Parametri modela
Ekstrahirano iz training_original.py linije 2046-2156

Sadrži konfiguraciju za različite tipove modela:
- Dense: Fully connected neuronska mreža
- CNN: Konvolucijska neuronska mreža
- LSTM: Long Short-Term Memory mreža
- AR LSTM: Autoregressive LSTM
- SVR_dir: Support Vector Regression (direktna)
- SVR_MIMO: Support Vector Regression (Multi-Input Multi-Output)
- LIN: Linearna regresija
"""

###############################################################################
# INFORMATIONEN ZUM MODELL ####################################################
###############################################################################

class MDL:

    # Ausgewähltes Verfahren
    #MODE = "Dense"
    #MODE = "CNN"
    #MODE = "LSTM"
    #MODE = "AR LSTM"
    #MODE = "SVR_dir"
    #MODE = "SVR_MIMO"
    MODE = "LIN"

    if MODE == "Dense":

        # Anzahl an Layer [-]
        LAY = 3

        # Anzahl der Neuronen pro Layer [-]
        N = 512

        # Max. Anzahl der Trainingsdurchläufe [-]
        EP = 20

        # Aktivierungsfunktion
        ACTF = "ReLU"

    elif MODE == "CNN":

        # Anzahl an Layer [-]
        LAY = 3

        # Anzahl der Filter pro Layer [-]
        N = 512

        # Kernelgröße [-]
        K = 3

        # Max. Anzahl der Trainingsdurchläufe [-]
        EP = 20

        # Aktivierungsfunktion
        ACTF = "ReLU"


    elif MODE == "LSTM":

        # Anzahl an Layer [-]
        LAY = 3

        # Anzahl der Neuronen pro Layer [-]
        N = 512

        # Max. Anzahl der Trainingsdurchläufe [-]
        EP = 20

        # Aktivierungsfunktion
        ACTF = "ReLU"

    elif MODE == "AR LSTM":

        # Anzahl an Layer [-]
        LAY = 3

        # Anzahl der Neuronen pro Layer [-]
        N = 512

        # Max. Anzahl der Trainingsdurchläufe [-]
        EP = 20

        # Aktivierungsfunktion
        ACTF = "ReLU"

    elif MODE == "SVR_dir":

        # Art der Modellierung von Nichtlinearitäten
        KERNEL = "poly"

        # Regulationsparameter (Trade-off Bias vs. Varianz) [-]
        C = 1

        # Toleranz für Abweichungen [-]
        EPSILON = 0.1

    elif MODE == "SVR_MIMO":

        # Art der Modellierung von Nichtlinearitäten
        KERNEL = "poly"

        # Regulationsparameter (Trade-off Bias vs. Varianz) [-]
        C = 1

        # Toleranz für Abweichungen [-]
        EPSILON = 0.1

    elif MODE == "LIN":
        a = 5


# Mapiranje aktivacijskih funkcija za Frontend->Backend konverziju
ACTIVATION_MAPPING = {
    'ReLU':     'relu',
    'Sigmoid':  'sigmoid',
    'Tanh':     'tanh',
    'Linear':   'linear',
    'Softmax':  'softmax',
    'Keine':    None,
}
