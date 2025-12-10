"""
Scaling - Skaliranje podataka i kreiranje scalera
Ekstrahirano iz training_original.py linije 1750-1873

Sadrži:
- Kreiranje 3D numpy array-a iz liste
- Konfiguraciju skaliranja za ulazne i izlazne podatke
- Kreiranje MinMaxScaler objekata za svaku kolonu
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config.time_features import T

###############################################################################
# ZAVRŠETAK KREIRANJE DATASETA ################################################
###############################################################################

prog_1 = 100
print(f"Erstellung der Datensätze: {prog_1:.2f}%")

i_array_3D = np.array(i_arrays)
o_array_3D = np.array(o_arrays)

# Anzahl der Datensätze
n_dat = i_array_3D.shape[0]

i_combined_array = np.vstack(i_arrays)
o_combined_array = np.vstack(o_arrays)

del i_arrays, o_arrays

###############################################################################
# KONFIGURACIJA SKALIRANJA ####################################################
###############################################################################

i_scal_list     = i_dat_inf["scal"].tolist()
i_scal_max_list = i_dat_inf["scal_max"].tolist()
i_scal_min_list = i_dat_inf["scal_min"].tolist()

imp = [T.Y.IMP,
       T.M.IMP,
       T.W.IMP,
       T.D.IMP,
       T.H.IMP]

scal = [T.Y.SCAL,
        T.M.SCAL,
        T.W.SCAL,
        T.D.SCAL,
        T.H.SCAL]

scal_max = [T.Y.SCAL_MAX,
            T.M.SCAL_MAX,
            T.W.SCAL_MAX,
            T.D.SCAL_MAX,
            T.H.SCAL_MAX]

scal_min = [T.Y.SCAL_MIN,
            T.M.SCAL_MIN,
            T.W.SCAL_MIN,
            T.D.SCAL_MIN,
            T.H.SCAL_MIN]

for i in range(len(imp)):
    if imp[i] == True and scal[i] == True:
        i_scal_list.append(True)
        i_scal_list.append(True)
        i_scal_max_list.append(scal_max[i])
        i_scal_max_list.append(scal_max[i])
        i_scal_min_list.append(scal_min[i])
        i_scal_min_list.append(scal_min[i])
    elif imp[i]  == True and scal[i] == False:
        i_scal_list.append(False)
        i_scal_list.append(False)
        i_scal_max_list.append(scal_max[i])
        i_scal_max_list.append(scal_max[i])
        i_scal_min_list.append(scal_min[i])
        i_scal_min_list.append(scal_min[i])

o_scal_list     = o_dat_inf["scal"].tolist()
o_scal_max_list = o_dat_inf["scal_max"].tolist()
o_scal_min_list = o_dat_inf["scal_min"].tolist()

###############################################################################
# KREIRANJE SCALERA ZA ULAZNE PODATKE #########################################
###############################################################################

# Erstellen eines leeres Dictionary, um später für jede Spalte im Datenrahmen
# eine eigene Min-Max-Skalierung speichern zu können
i_scalers = {}

scal_all = sum(i_scal_list)+sum(o_scal_list)
scal_i = 0

for i in range(i_combined_array.shape[1]):  # Schleife über Spalten
    if i_scal_list[i] == True:


        prog_2 = scal_i/scal_all*100
        print(f"Skalierer einstellen: {prog_2:.2f}%")


        scaler = MinMaxScaler(feature_range = (i_scal_min_list[i],
                                               i_scal_max_list[i]))
        scaler.fit_transform(i_combined_array[:, i].reshape(-1, 1))
        i_scalers[i] = scaler

        scal_i += 1

        prog_2 = scal_i/scal_all*100
        print(f"Skalierer einstellen: {prog_2:.2f}%")

    else:
        i_scalers[i] = None

###############################################################################
# KREIRANJE SCALERA ZA IZLAZNE PODATKE ########################################
###############################################################################

# Erstellen eines leeres Dictionary, um später für jede Spalte im Datenrahmen
# eine eigene Min-Max-Skalierung speichern zu können
o_scalers = {}

for i in range(o_combined_array.shape[1]):  # Schleife über Spalten
    if o_scal_list[i] == True:

        prog_2 = scal_i/scal_all*100
        print(f"Skalierer einstellen: {prog_2:.2f}%")

        scaler = MinMaxScaler(feature_range = (o_scal_min_list[i],
                                               o_scal_max_list[i]))
        scaler.fit_transform(o_combined_array[:, i].reshape(-1, 1))
        o_scalers[i] = scaler

        scal_i += 1

        prog_2 = scal_i/scal_all*100
        print(f"Skalierer einstellen: {prog_2:.2f}%")

    else:
        o_scalers[i] = None

###############################################################################
# PROVJERA SKALIRANJA #########################################################
###############################################################################

if any(i_scal_list):
   i_scal_button = True
else:
   i_scal_button = False

if any(o_scal_list):
   o_scal_button = True
else:
   o_scal_button = False
