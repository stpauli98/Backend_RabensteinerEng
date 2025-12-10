"""
Averaging - Mittelwertbildung (usrednjavanje)
Ekstrahirano iz training_original.py linije 3249-3292

Sadrži:
- Pripremu evaluacijskih struktura
- Usrednjavanje po različitim vremenskim intervalima (1-12)
- Spremanje usrednjenih y i fcst vrijednosti
"""

import numpy as np
import math

from config.mts import MTS

###############################################################################
###############################################################################
# EVALUIERUNG #################################################################
###############################################################################
###############################################################################

# Napomena: Ovaj kod zahtijeva prethodno definirane varijable:
# tst_y_orig, tst_fcst, n_tst, o_dat_inf

# Anzahl der Merkmale
num_feat = tst_y_orig.shape[2]

###############################################################################
# MITTELWERTBILDUNG ###########################################################
###############################################################################

dat_eval = {}
n_max = 12

y_all       = np.full((n_max, n_tst, MTS.O_N, num_feat), np.nan)
fcst_all    = np.full((n_max, n_tst, MTS.O_N, num_feat), np.nan)

# Schleife über alle Mittelungen
for n_avg in range(1, n_max+1):

    # Anzahl der Zeitschritte der gemittelten Arrays
    n_ts = math.floor(MTS.O_N/n_avg)

    # Array vorbereiten
    y       = np.zeros((n_tst, n_ts, num_feat))
    fcst    = np.zeros((n_tst, n_ts, num_feat))
    dat_eval_int = {}

    # Schleife über jeden Testdatensatz
    for i in range(n_tst):

        # Schleife über jedes Merkmal
        for j in range(num_feat):

            # Schleife über jeden Zeitschritt
            for k in range(n_ts):
                strt = k * n_avg
                end = min(strt + n_avg, MTS.O_N)
                y[i, k, j] = np.mean(tst_y_orig[i, strt:end, j])
                fcst[i, k, j] = np.mean(tst_fcst[i, strt:end, j])

                y_all[n_avg-1, i, k, j]     = np.mean(tst_y_orig[i, strt:end, j])
                fcst_all[n_avg-1, i, k, j]  = np.mean(tst_fcst[i, strt:end, j])

    dat_eval_int["y"] = y
    dat_eval_int["fcst"] = fcst
    dat_eval_int["delt"] = np.array(o_dat_inf["delt_transf"]*n_avg)
    dat_eval[n_avg] = dat_eval_int
