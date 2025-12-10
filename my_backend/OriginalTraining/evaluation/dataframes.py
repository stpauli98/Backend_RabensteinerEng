"""
DataFrames - Kreiranje evaluacijskih DataFrame-ova
Ekstrahirano iz training_original.py linije 3402-3469

Sadrži:
- df_eval: DataFrame s ukupnim metrikama po značajki i delta intervalu
- df_eval_ts: DataFrame s metrikama po vremenskim koracima
"""

import pandas as pd

###############################################################################
# DATAFRAMES ##################################################################
###############################################################################

# Napomena: Ovaj kod zahtijeva prethodno definirane varijable:
# n_ft_o, n_max, dat_eval, o_dat_inf

df_eval = {}

for i_feat in range(n_ft_o):

    # Initialisierung
    delt_int, mae_int, mape_int, mse_int, rmse_int, nrmse_int, wape_int, \
        smape_int, mase_int = ([] for _ in range(9))

    # Initialisierung
    mae_ts, mape_ts, mse_ts, rmse_ts, nrmse_ts, wape_ts, \
        smape_ts, mase_ts = ([] for _ in range(8))

    for i in range(n_max):
        delt_int.append(float(dat_eval[i+1]["delt"][i_feat]))
        mae_int.append(float(dat_eval[i+1]["MAE"][i_feat]))
        mape_int.append(float(dat_eval[i+1]["MAPE"][i_feat]))
        mse_int.append(float(dat_eval[i+1]["MSE"][i_feat]))
        rmse_int.append(float(dat_eval[i+1]["RMSE"][i_feat]))
        nrmse_int.append(float(dat_eval[i+1]["NRMSE"][i_feat]))
        wape_int.append(float(dat_eval[i+1]["WAPE"][i_feat]))
        smape_int.append(float(dat_eval[i+1]["sMAPE"][i_feat]))
        mase_int.append(float(dat_eval[i+1]["MASE"][i_feat]))

    df_eval_int = pd.DataFrame({
    "delta [min]":  delt_int,
    "MAE":          mae_int,
    "MAPE":         mape_int,
    "MSE":          mse_int,
    "RMSE":         rmse_int,
    "NRMSE":        nrmse_int,
    "WAPE":         wape_int,
    "sMAPE":        smape_int,
    "MASE":         mase_int
        })

    df_eval[o_dat_inf.index[i_feat]] = df_eval_int


df_eval_ts = {}

for i_feat in range(n_ft_o):

    df_eval_ts[o_dat_inf.index[i_feat]] = {}

    # Initialisierung
    delt_int, mae_int, mape_int, mse_int, rmse_int, nrmse_int, wape_int, \
        smape_int, mase_int = ([] for _ in range(9))

    # Initialisierung
    mae_ts, mape_ts, mse_ts, rmse_ts, nrmse_ts, wape_ts, \
        smape_ts, mase_ts = ([] for _ in range(8))

    df_eval_ts[o_dat_inf.index[i_feat]] = {}

    for i in range(n_max):

        df_eval_ts_int = pd.DataFrame({
        'MAE':      dat_eval[i+1]["MAE_TS"][i_feat],
        'MAPE':     dat_eval[i+1]["MAPE_TS"][i_feat],
        'MSE':      dat_eval[i+1]["MSE_TS"][i_feat],
        'RMSE':     dat_eval[i+1]["RMSE_TS"][i_feat],
        'NRMSE':    dat_eval[i+1]["NRMSE_TS"][i_feat],
        'WAPE':     dat_eval[i+1]["WAPE_TS"][i_feat],
        'sMAPE':    dat_eval[i+1]["sMAPE_TS"][i_feat],
        'MASE':     dat_eval[i+1]["MASE_TS"][i_feat]
        })

        df_eval_ts[o_dat_inf.index[i_feat]][float(dat_eval[i+1]["delt"][i_feat])] = df_eval_ts_int
