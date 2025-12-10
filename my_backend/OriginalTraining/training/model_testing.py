"""
Model Testing - Testiranje modela i re-scaling
Ekstrahirano iz training_original.py linije 2261-2332

Sadrži:
- Predikciju za svaki test dataset
- Posebnu logiku za SVR i LIN modele
- Re-scaling predikcija na originalne vrijednosti
"""

import numpy as np
import copy

from config.model import MDL

###############################################################################
# MODELL TESTEN  ##############################################################
###############################################################################

# Napomena: Ovaj kod zahtijeva prethodno definirane varijable:
# mdl, tst_x, n_tst, n_ft_i, n_ft_o, o_scalers, tst_y

# Initialisierung
tst_fcst = list()

# Forecast für jeden Testdatensatz
for i in range(n_tst):

    prog_4 = i/n_tst*100
    print(f"Modell testen: {prog_4:.2f}%")

    # Erstellung der Input-Daten für die Prognose
    inp = tst_x[i, : ,:].reshape((1, tst_x.shape[1], n_ft_i))

    if MDL.MODE == "SVR_dir" or MDL.MODE == "SVR_MIMO" or MDL.MODE == "LIN":

        inp = np.squeeze(inp, axis = 0)
        pred = []
        for i in range(n_ft_o):
            pred.append(mdl[i].predict(inp))

        out = np.array(pred).T

        out = np.expand_dims(out, axis = 0)

    else:

        # Prognose erstellen
        out = mdl.predict(inp,
                          verbose = 0)

    # Man gibt nur die Vektorvorhersage aus
    tst_fcst.append(out[0,:,:])

prog_4 = 100
print(f"Modell testen: {prog_4:.2f}%")

# Prognosen in ein Array umwandeln
tst_fcst = np.array(tst_fcst)

if MDL.MODE == "CNN":
    tst_fcst = np.squeeze(tst_fcst, axis = -1)

tst_fcst_scal = copy.deepcopy(tst_fcst)

###############################################################################
# RE-SCALING ##################################################################
###############################################################################

# Durchlauf der Testdatensätze
for i in range(n_tst):

    prog_5 = i/n_tst*100
    print(f"Re-Scaling: {prog_5:.2f}%")

    # Durchlauf der Merkmale der Ausgabe-Testdatensätze
    for i1 in range(n_ft_o):

        if not o_scalers[i1] is None:

            # Skalierer anwenden
            a = o_scalers[i1].inverse_transform(tst_fcst[i,:,i1].reshape(-1, 1))
            b = o_scalers[i1].inverse_transform(tst_y[i,:,i1].reshape(-1, 1))

            # Überschreiben der Spalte
            tst_fcst[i,:,i1] = a.ravel()
            tst_y[i,:,i1] = a.ravel()

prog_5 = 100
print(f"Re-Scaling: {prog_5:.2f}%")
