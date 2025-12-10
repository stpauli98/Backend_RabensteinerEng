"""
Error Calculation - Berechnung aller Fehlermetriken
Ekstrahirano iz training_original.py linije 3294-3401

Sadrži:
- Izračun ukupnih metrika grešaka (MAE, MAPE, MSE, RMSE, NRMSE, WAPE, sMAPE, MASE)
- Izračun metrika po vremenskim koracima (_TS verzije)
"""

import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import root_mean_squared_error as rmse

from utils.metrics import wape, smape, mase

###############################################################################
# FEHLERBERECHNUNG ############################################################
###############################################################################

# Napomena: Ovaj kod zahtijeva prethodno definirane varijable:
# n_max, num_feat, y_all, fcst_all, dat_eval

# GESAMT ######################################################################

# Schleife über alle Mittelungen
for i in range(n_max):

    # Initialisierung
    mae_int, mape_int, mse_int, rmse_int, nrmse_int, wape_int, \
        smape_int, mase_int = ([] for _ in range(8))

    # Durchlauf aller Merkmale
    for i_feat in range(num_feat):

        v_true = y_all[i,:,:,i_feat]
        v_fcst = fcst_all[i,:,:,i_feat]

        mask = ~np.isnan(v_true) & ~np.isnan(v_fcst)
        mask_1 = ~np.isnan(v_true)

        try:
            mae_int.append(mae(v_true[mask], v_fcst[mask]))
            mape_int.append(100*mape(v_true[mask], v_fcst[mask]))
            mse_int.append(mse(v_true[mask], v_fcst[mask]))
            rmse_int.append(rmse(v_true[mask], v_fcst[mask]))
            nrmse_int.append(rmse(v_true[mask], v_fcst[mask])/np.mean(v_true[mask_1]))
            wape_int.append(wape(v_true[mask], v_fcst[mask]))
            smape_int.append(smape(v_true[mask], v_fcst[mask]))
            mase_int.append(mase(v_true[mask], v_fcst[mask]))
        except:

            pass

    # Mittlerer absoluter Fehler (Mean Absolute Error, MAE)
    dat_eval[i+1]["MAE"]    = np.array(mae_int)

    # Mittlerer absoluter prozentualer Fehler (Mean Absolute Percentage Error, MAPE)
    dat_eval[i+1]["MAPE"]   = np.array(mape_int)

    # Mittlerer quatratischer Fehler (Mean Squared Error, MSE)
    dat_eval[i+1]["MSE"]    = np.array(mse_int)

    # Wurzel des mittleren quatratischen Fehlers (Root Mean Squared Error, RMSE)
    dat_eval[i+1]["RMSE"]    = np.array(rmse_int)

    # Wurzel des mittleren quatratischen Fehlers, normalisiert (Normalized Root Mean Squared Error, NRMSE)
    dat_eval[i+1]["NRMSE"]    = np.array(nrmse_int)

    # Gewichteter absoluter prozentualer Fehler (Weighted Average Percentage Error, WAPE)
    dat_eval[i+1]["WAPE"]    = np.array(wape_int)

    # Symmetrischer mittlerer absoluter prozentualer Fehler (Symmetric Mean Absolute Percentage Error, sMAPE)
    dat_eval[i+1]["sMAPE"]    = np.array(smape_int)

    # Skalierter mittlerer absoluter Fehler (Mean Absolute Scaled Error, MASE)
    dat_eval[i+1]["MASE"]    = np.array(mase_int)

# ZEITSCHRITTE ################################################################

# Schleife über alle Mittelungen
for i in range(n_max):

    # Initialisierung
    mae_ts, mape_ts, mse_ts, rmse_ts, nrmse_ts, wape_ts, \
        smape_ts, mase_ts = ([] for _ in range(8))

    # Durchlauf aller Merkmale
    for i_feat in range(num_feat):

        # Initialisierung
        mae_int, mape_int, mse_int, rmse_int, nrmse_int, wape_int, \
            smape_int, mase_int = ([] for _ in range(8))

        # Durchlauf aller Zeitschritte
        for i_ts in range(dat_eval[i+1]["y"].shape[1]):

            v_true = y_all[i,:,i_ts,i_feat]
            v_fcst = fcst_all[i,:,i_ts,i_feat]

            mae_int.append(mae(v_true, v_fcst))
            mape_int.append(100*mape(v_true, v_fcst))
            mse_int.append(mse(v_true, v_fcst))
            rmse_int.append(rmse(v_true, v_fcst))
            nrmse_int.append(rmse(v_true, v_fcst)/np.mean(v_true))
            wape_int.append(wape(v_true, v_fcst))
            smape_int.append(smape(v_true, v_fcst))
            mase_int.append(mase(v_true, v_fcst))

        mae_ts.append(mae_int)
        mape_ts.append(mape_int)
        mse_ts.append(mse_int)
        rmse_ts.append(rmse_int)
        nrmse_ts.append(nrmse_int)
        wape_ts.append(wape_int)
        smape_ts.append(smape_int)
        mase_ts.append(mase_int)

    dat_eval[i+1]["MAE_TS"]     = np.array(mae_ts)
    dat_eval[i+1]["MAPE_TS"]    = np.array(mape_ts)
    dat_eval[i+1]["MSE_TS"]     = np.array(mse_ts)
    dat_eval[i+1]["RMSE_TS"]    = np.array(rmse_ts)
    dat_eval[i+1]["NRMSE_TS"]   = np.array(nrmse_ts)
    dat_eval[i+1]["WAPE_TS"]    = np.array(wape_ts)
    dat_eval[i+1]["sMAPE_TS"]   = np.array(smape_ts)
    dat_eval[i+1]["MASE_TS"]    = np.array(mase_ts)
