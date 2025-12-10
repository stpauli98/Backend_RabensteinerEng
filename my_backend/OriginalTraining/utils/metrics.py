"""
Metrics - Funkcije za evaluaciju modela
Ekstrahirano iz training_original.py linije 553-608

Sadrži metrike za evaluaciju:
- wape: Weighted Absolute Percentage Error
- smape: Symmetric Mean Absolute Percentage Error
- mase: Mean Absolute Scaled Error
"""

import numpy as np

###############################################################################
# FUNKTION ZUR BERECHNUNG DES GEWICHTETEN ABSOLUTEN PROZENTUALEN FEHLERS ######
###############################################################################

def wape(y_true, y_pred):

    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return np.nan

    return (numerator/denominator)*100

###############################################################################
# FUNKTION ZUR BERECHNUNG DES SYMMETRISCHEN MITTLEREN ABSOLUTEN PROZENTUALEN ##
# FEHLERS #####################################################################
###############################################################################

def smape(y_true, y_pred):

    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    n = len(y_true)
    smape_values = []

    for yt, yp in zip(y_true, y_pred):
        denominator = (abs(yt)+abs(yp))/2
        if denominator == 0:
            smape_values.append(0)
        else:
            smape_values.append(abs(yp-yt)/denominator)

    return sum(smape_values)/n*100

###############################################################################
# FUNKTION ZUR BERECHNUNG DES MITTLEREN ABSOLUTEN FEHLERS #####################
###############################################################################

def mase(y_true, y_pred, m = 1):

    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    n = len(y_true)

    # Vorhersagefehler (MAE der Prognose)
    mae_forecast = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n

    # MAE des Naive-m-Modells (Baseline)
    if n <= m:
        raise ValueError("Zu wenig Daten für gewählte Saisonalität m.")

    naive_errors = [abs(y_true[t] - y_true[t - m]) for t in range(m, n)]
    mae_naive = sum(naive_errors) / len(naive_errors)

    if mae_naive == 0:
        raise ZeroDivisionError("Naive MAE ist 0 – MASE nicht definiert.")

    return mae_forecast/mae_naive
