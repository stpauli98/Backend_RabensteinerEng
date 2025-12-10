"""
Data Split - Razdvajanje na train/val/test i skaliranje
Ekstrahirano iz training_original.py linije 2034-2234

Sadrži:
- Konfiguraciju omjera train/val/test (70%/20%/10%)
- Opcionalnu randomizaciju podataka
- Skaliranje dataseta koristeći prethodno kreirane scalere
- Kreiranje finalnih dataseta
"""

import numpy as np
import copy

###############################################################################
###############################################################################
# MODELL TRAINIEREN, VALIDIEREN UND TESTEN ####################################
###############################################################################
###############################################################################

# Napomena: Ovaj kod zahtijeva prethodno definirane varijable:
# n_dat, i_array_3D, o_array_3D, utc_ref_log, n_ft_i, n_ft_o,
# i_scalers, o_scalers, i_dat_inf, o_dat_inf

###############################################################################
# EINGABEN ####################################################################
###############################################################################

random_dat = False

n_train = round(0.7*n_dat)
n_val   = round(0.2*n_dat)
n_tst = n_dat-n_train-n_val

###############################################################################
# RANDOMISIERUNG DER DATENSÄTZE ###############################################
###############################################################################

if random_dat == True:

    indices = np.random.permutation(n_dat)
    i_array_3D = i_array_3D[indices]
    o_array_3D = o_array_3D[indices]

    utc_ref_log_int = copy.deepcopy(utc_ref_log)
    utc_ref_log = [utc_ref_log_int[i] for i in indices]
    del utc_ref_log_int

# UNSKALIERTE DATENSÄTZE SPEICHERN ############################################

i_array_3D_orig = copy.deepcopy(i_array_3D)
o_array_3D_orig = copy.deepcopy(o_array_3D)

###############################################################################
# SKALIERUNG DER DATENSÄTZE ###################################################
###############################################################################

# Durchlauf der Datensätze
for i in range(n_dat):

    prog_3 = i/n_dat*100
    print(f"Skalierung der Datensätze: {prog_3:.2f}%")

    # Durchlauf der Merkmale der Eingabedaten
    for i1 in range(n_ft_i):

        if not i_scalers[i1] is None:

            # Skalierer anwenden
            std_i = i_scalers[i1].transform(i_array_3D[i,:,i1].reshape(-1, 1))

            # Überschreiben der Spalte
            i_array_3D[i,:,i1] = std_i.ravel()

    # Durchlauf der Merkmale der Ausgabedaten
    for i1 in range(len(o_dat_inf)):

        if not o_scalers[i1] is None:

            # Skalierer anwenden
            std_i = o_scalers[i1].transform(o_array_3D[i,:,i1].reshape(-1, 1))

            # Überschreiben der Spalte
            o_array_3D[i,:,i1] = std_i.ravel()

prog_3 = 100
print(f"Skalierung der Datensätze: {prog_3:.2f}%")

###############################################################################
# FINALE DATENSÄTZE ###########################################################
###############################################################################

# SKALIERTE DATENSÄTZE ########################################################

trn_x = i_array_3D[:n_train]
val_x = i_array_3D[n_train:(n_train+n_val)]
tst_x = i_array_3D[(n_train+n_val):]

trn_y = o_array_3D[:n_train]
val_y = o_array_3D[n_train:(n_train+n_val)]
tst_y = o_array_3D[(n_train+n_val):]

# ORIGINALLE (UNSKALIERTE) DATENSÄTZE #########################################

trn_x_orig = i_array_3D_orig[:n_train]
val_x_orig = i_array_3D_orig[n_train:(n_train+n_val)]
tst_x_orig = i_array_3D_orig[(n_train+n_val):]

trn_y_orig = o_array_3D_orig[:n_train]
val_y_orig = o_array_3D_orig[n_train:(n_train+n_val)]
tst_y_orig = o_array_3D_orig[(n_train+n_val):]
