"""
Test Plots - Vizualizacija rezultata testiranja
Ekstrahirano iz training_original.py linije 2340-3241

Sadrži:
- Pripremu subplot mreže za testne podatke
- Plotanje ulaznih podataka, izlaznih podataka i predikcija
- Konfiguraciju y-osi (gemeinsame/separate Achsen)
- Legende i naslove
"""

import datetime
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config.mts import MTS
from config.time_features import T

###############################################################################
###############################################################################
# AUSWERTUNG DER TESTDATENSÄTZE ###############################################
###############################################################################
###############################################################################

# Napomena: Ovaj kod zahtijeva prethodno definirane varijable:
# i_list, o_list, i_dat_inf, o_dat_inf, tst_x, tst_y, tst_fcst,
# tst_x_orig, tst_y_orig, tst_fcst_orig, n_tst, utc_ref_log,
# i_combined_array, palette

###############################################################################
# VORBEREITUNG ################################################################
###############################################################################

df_plot_in = pd.DataFrame({'plot': False}, index = i_list)
df_plot_out = pd.DataFrame({'plot': False}, index = o_list)
df_plot_fcst = pd.DataFrame({'plot': False}, index = o_list)

###############################################################################
# EINGABEN ####################################################################
###############################################################################

# Anzahl an Subplots [-]
num_sbpl = 17

# x-Achse
x_sbpl = "UTC"

# y-Achse - Darstellungsform
y_sbpl_fmt = "original"

# y-Achse - Konfiguration
y_sbpl_set = "separate Achsen"

# Anzeige
df_plot_in.loc["Netzlast [kW]", 'plot']                         = True
df_plot_in.loc["Aussentemperatur Krumpendorf [GradC]", 'plot']  = True
df_plot_out.loc["Netzlast [kW]", 'plot']                        = True
df_plot_fcst.loc["Netzlast [kW]", 'plot']                       = True

###############################################################################
# VORBEREITUNG DER SUBPLOTS ###################################################
###############################################################################

if y_sbpl_set == "separate Achsen":

    # Anzahl separater y-Achsen in einem Subplot
    n_ax = (df_plot_in['plot'].sum()+
            (df_plot_out["plot"]|df_plot_fcst["plot"]).sum())

    # Anzahl separater Y-Achsen auf der linken Seite eines Subplots
    n_ax_l = math.floor(n_ax/2)
    if n_ax_l == 0: n_ax_l = 1

    # Anzahl separater Y-Achsen auf der rechten Seite eines Subplots
    n_ax_r = n_ax-n_ax_l

# Anzahl an befüllten Subplots [-]
num_sbpl = min(num_sbpl, n_tst)

# Anzahl der Subplots in horizontaler Richtung [-]
num_sbpl_x = math.ceil(math.sqrt(num_sbpl))

# Anzahl der Subplot in vertikaler Richtung [-]
num_sbpl_y = math.ceil(num_sbpl/num_sbpl_x)

fig, axs = plt.subplots(num_sbpl_y,
                       num_sbpl_x,
                       figsize              = (20, 13),
                       layout               = 'constrained')

# Anzahl an zu entfernenden Subplots in der letzen Zeile
sbpl_del = num_sbpl_x*num_sbpl_y-num_sbpl
for i in range(sbpl_del):
    axs[num_sbpl_y-1, num_sbpl_x-1-i].axis('off')

# Liste an Zufallszahlen für die Auswahl der Testdatensätze
tst_random = random.sample(range(n_tst), num_sbpl)

# Referenz-UTC der Testdatensätze
utc_ref_log_tst = utc_ref_log[-n_tst:]

###############################################################################
# DATEN VORBEREITEN ###########################################################
###############################################################################

# Dictionary erstellen
tst_inf = {
    random_num: {
        "utc_ref": utc_ref_log_tst[random_num]
    }
    for random_num in tst_random
}

# Durchlauf der Subplots
for key_1 in tst_inf.keys():

    # Referenzzeit des aktuellen Subplots
    utc_ref = tst_inf[key_1]["utc_ref"]

    # DURCHLAUF DER EINGABEDATEN ##############################################

    for i in range(len(i_dat_inf)):

        if i_dat_inf.iloc[i]["spec"] == "Historische Daten":

            # ZEITGRENZEN DER TRANSFERIERUNG ##################################

            utc_th_strt = utc_ref+datetime.timedelta(hours = i_dat_inf.iloc[i]["th_strt"])
            utc_th_end  = utc_ref+datetime.timedelta(hours = i_dat_inf.iloc[i]["th_end"])

            # ZEITSTEMPEL DER TRANSFERIERUNG ##############################
            try:
                utc_th = pd.date_range(start  = utc_th_strt,
                                       end    = utc_th_end,
                                       freq   = f'{i_dat_inf.iloc[i]["delt_transf"]}min'
                                       ).to_list()
            except:

                # Berechne timedelta
                delt = pd.to_timedelta(i_dat_inf.iloc[i]["delt_transf"],
                                       unit = "min")

                # Erzeuge Zeitreihe manuell
                utc_th = []
                utc = utc_th_strt
                for i1 in range(MTS.I_N):
                    utc_th.append(utc)
                    utc += delt

            if y_sbpl_fmt == "original":
                value = tst_x_orig[key_1,:,i]
            elif y_sbpl_fmt == "skaliert":
                value = tst_x[key_1,:,i]

            # DataFrame erstellen
            df = pd.DataFrame({
                "UTC":      utc_th,
                "ts":       list(range(-MTS.I_N+1, 1)),
                "value":    value
            })

            tst_inf[key_1]["IN: "+i_dat_inf.index[i]] = df

        elif i_dat_inf.iloc[i, "spec"] == "Historische Prognosen":
            print("MUSS NOCH PROGRAMMIERT WERDEN!")

        elif i_dat_inf.iloc[i, "spec"] == "Aktueller Wert":
            print("MUSS NOCH PROGRAMMIERT WERDEN!")

    # DURCHLAUF DER AUSGABEDATEN ##############################################

    for i in range(len(o_dat_inf)):

        if o_dat_inf.iloc[i]["spec"] == "Historische Daten":

            # ZEITGRENZEN DER TRANSFERIERUNG ##############################

            utc_th_strt = utc_ref+datetime.timedelta(hours = o_dat_inf.iloc[i]["th_strt"])
            utc_th_end  = utc_ref+datetime.timedelta(hours = o_dat_inf.iloc[i]["th_end"])

            # ZEITSTEMPEL DER TRANSFERIERUNG ##############################
            try:
                utc_th = pd.date_range(start  = utc_th_strt,
                                       end    = utc_th_end,
                                       freq   = f'{o_dat_inf.iloc[i]["delt_transf"]}min'
                                       ).to_list()
            except:

                # Berechne timedelta
                delt = pd.to_timedelta(o_dat_inf.iloc[i]["delt_transf"],
                                       unit = "min")

                # Erzeuge Zeitreihe manuell
                utc_th = []
                utc = utc_th_strt
                for i1 in range(MTS.O_N):
                    utc_th.append(utc)
                    utc += delt

            if y_sbpl_fmt == "original":
                value = tst_y_orig[key_1,:,i]
            elif y_sbpl_fmt == "skaliert":
                value = tst_y[key_1,:,i]

            # DataFrame erstellen
            df = pd.DataFrame({
                "UTC":      utc_th,
                "ts":       list(range(0, MTS.O_N)),
                "value":    value
            })

            tst_inf[key_1]["OUT: "+o_dat_inf.index[i]] = df

        elif o_dat_inf.iloc[i, "spec"] == "Historische Prognosen":
            print("MUSS NOCH PROGRAMMIERT WERDEN!")

# Nastavak koda za plotanje - vidjeti kompletan kod u training_original.py
# Linije 2527-3241 sadrže:
# - Obradu vremenskih informacija za plotanje
# - Prognoze (FCST) za plotanje
# - Plotanje s odvojenim y osima
# - Legendu i naslov

###############################################################################
# PLOTANJE SUBPLOTOVA #########################################################
###############################################################################

# Napomena: Ovaj dio koda je skraćen zbog veličine
# Za kompletan kod pogledati training_original.py linije 2850-3241

# LEGENDE  ####################################################################

# Gemeinsame Legende
# fig.legend(lines, labels, loc="upper right", ncol=5, fontsize=8)

# TITEL #######################################################################

plt.suptitle("Auswertung der Testdatensätze",
             fontsize   = 20,
             fontweight = 'bold')

plt.show()
