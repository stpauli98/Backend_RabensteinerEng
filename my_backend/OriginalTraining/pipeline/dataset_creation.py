"""
Dataset Creation - Kreiranje dataseta iz ulaznih i izlaznih podataka
Ekstrahirano iz training_original.py linije 1049-1368

Sadrži glavnu petlju za kreiranje dataseta:
- Prolazi kroz sve vremenske korake
- Transferira ulazne podatke linearnom interpolacijom
- Transferira izlazne podatke linearnom interpolacijom
- Podržava različite tipove podataka (Historische Daten, Prognosen, Aktueller Wert)
"""

import datetime
import math
import pandas as pd
import numpy as np

from utils.time_utils import utc_idx_pre, utc_idx_post
from config.mts import MTS

###############################################################################
###############################################################################
# DATENSÄTZE ERSTELLEN ########################################################
###############################################################################
###############################################################################

# Napomena: Ovaj kod zahtijeva prethodno definirane varijable:
# i_dat, i_dat_inf, o_dat, o_dat_inf (iz data/ modula)

# Startzeit für die Erstellung der Datensätze
utc_strt = i_dat_inf["utc_min"].min()

# Endzeit für die Erstellung der Datensätze
utc_end = i_dat_inf["utc_max"].min()

# Berechnung der Referenzzeit
utc_ref = utc_strt.replace(minute       = 0,
                           second       = 0,
                           microsecond  = 0)\
    -datetime.timedelta(hours = 1)\
    +datetime.timedelta(minutes = MTS.OFST)

while utc_ref < utc_strt:
    utc_ref += datetime.timedelta(minutes = MTS.DELT)

# Initialisierung
error = False
i_arrays = []
o_arrays = []
utc_ref_log = []
utc_strt = utc_ref


# Durchlauf der Zeitschritte
while True:

    # Endzeit wurde erreicht → Schleife abbrechen
    if utc_ref > utc_end:
        break

    prog_1 = (utc_ref-utc_strt)/(utc_end-utc_strt)*100
    print(f"Erstellung der Datensätze: {prog_1:.2f}%")

    df_int_i = pd.DataFrame()
    df_int_o = pd.DataFrame()

    ###########################################################################
    ###########################################################################
    ###########################################################################
    # DURCHLAUF DER EINGABEDATEN ##############################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    for i, (key, df) in enumerate(i_dat.items()):

        #######################################################################
        #######################################################################
        # HISTORISCHE DATEN ###################################################
        #######################################################################
        #######################################################################

        if i_dat_inf.loc[key, "spec"] == "Historische Daten":

            # ZEITGRENZEN DER TRANSFERIERUNG ##################################

            utc_th_strt = utc_ref+datetime.timedelta(hours = i_dat_inf.loc[key, "th_strt"])
            utc_th_end = utc_ref+datetime.timedelta(hours = i_dat_inf.loc[key, "th_end"])

            ###################################################################
            # MITTELWERTBILDUNG ###############################################
            ###################################################################

            if i_dat_inf.loc[key, "avg"] == True:

                # Erster Index
                idx1 = utc_idx_post(i_dat[key], utc_th_strt)

                # Zweiter Index
                idx2 = utc_idx_pre(i_dat[key], utc_th_end)

                # Mittelwert berechnen
                val = (i_dat[key].iloc[idx1:idx2, 1]).mean()

                # Keine Mittelwertbildung möglich
                if math.isnan(float(val)):
                    error = True
                    break
                else:
                    df_int_i[key] = [val]*MTS.I_N

            ###################################################################
            # KEINE MITTELWERTBILDUNG #########################################
            ###################################################################

            else:

                # Initialisierung
                val_list = []

                # ZEITSTEMPEL DER TRANSFERIERUNG ##############################
                try:
                    utc_th = pd.date_range(start  = utc_th_strt,
                                             end  = utc_th_end,
                                             freq = f'{i_dat_inf.loc[key, "delt_transf"]}min'
                                             ).to_list()
                except:

                    # Berechne timedelta
                    delt = pd.to_timedelta(i_dat_inf.loc[key, "delt_transf"], unit = "min")

                    # Erzeuge Zeitreihe manuell
                    utc_th = []
                    utc = utc_th_strt
                    for i1 in range(MTS.I_N):
                        utc_th.append(utc)
                        utc += delt

                # TRANSFERIERUNG DURCH LINEARE INTERPOLATION ##################
                if i_dat_inf.loc[key, "meth"] == "Lineare Interpolation":

                    # Schleife über den Zeitstempel der Transferierung
                    for i1 in range(len(utc_th)):

                        # Erster Index
                        idx1 = utc_idx_pre(i_dat[key], utc_th[i1])

                        # Zweiter Index
                        idx2 = utc_idx_post(i_dat[key], utc_th[i1])

                        # Kontrolle der Zeitgrenzen
                        if idx1 is None or idx2 is None:
                            error = True
                            break

                        if idx1 == idx2:
                             val = i_dat[key].iloc[idx1,1]
                        else:
                            utc1 = i_dat[key].iloc[idx1,0]
                            utc2 = i_dat[key].iloc[idx2,0]

                            val1 = i_dat[key].iloc[idx1,1]
                            val2 = i_dat[key].iloc[idx2,1]

                            val = (utc_th[i1]-utc1)/(utc2-utc1)*(val2-val1)+val1

                        # Kontrolle, ob der Wert eine Zahl ist
                        if math.isnan(float(val)):
                            error = True
                            break

                        else:
                            val_list.append(val)

                    if error == False:

                        df_int_i[key] = val_list

                    else:
                        break

                # TRANSFERIERUNG DURCH MITTELWERTBILDUNG ######################
                elif i_dat_inf.loc[key, "meth"] == "Mittelwertbildung":
                    print("MUSS NOCH PROGRAMMIERT WERDEN!")

                # TRANSFERIERUNG DURCH NÄCHSTER WERT ##########################
                elif i_dat_inf.loc[key, "meth"] == "Nächster Wert":
                    print("MUSS NOCH PROGRAMMIERT WERDEN!")

        #######################################################################
        #######################################################################
        # HISTORISCHE PROGNOSEN ###############################################
        #######################################################################
        #######################################################################
        elif i_dat_inf.loc[key, "spec"] == "Historische Prognosen":
            print("MUSS NOCH PROGRAMMIERT WERDEN!")

        #######################################################################
        #######################################################################
        # AKTUELLER WERT ######################################################
        #######################################################################
        #######################################################################
        elif i_dat_inf.loc[key, "spec"] == "Aktueller Wert":
            print("MUSS NOCH PROGRAMMIERT WERDEN!")

    ###########################################################################
    ###########################################################################
    ###########################################################################
    # DURCHLAUF DER AUSGABEDATEN ##############################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    if error == False:

        for i, (key, df) in enumerate(o_dat.items()):

            ###################################################################
            ###################################################################
            # HISTORISCHE DATEN ###############################################
            ###################################################################
            ###################################################################

            if o_dat_inf.loc[key, "spec"] == "Historische Daten":

                # ZEITGRENZEN DER TRANSFERIERUNG ##############################

                utc_th_strt = utc_ref+datetime.timedelta(hours = o_dat_inf.loc[key, "th_strt"])
                utc_th_end = utc_ref+datetime.timedelta(hours = o_dat_inf.loc[key, "th_end"])

                ###############################################################
                # MITTELWERTBILDUNG ###########################################
                ###############################################################

                if o_dat_inf.loc[key, "avg"] == True:

                    # Erster Index
                    idx1 = utc_idx_post(o_dat[key], utc_th_strt)

                    # Zweiter Index
                    idx2 = utc_idx_pre(o_dat[key], utc_th_end)

                    # Mittelwert berechnen
                    val = (o_dat[key].iloc[idx1:idx2, 1]).mean()

                    # Keine Mittelwertbildung möglich
                    if math.isnan(float(val)):
                        error = True
                        break
                    else:
                        df_int_o[key] = [val]*MTS.O_N

                ###############################################################
                # KEINE MITTELWERTBILDUNG #####################################
                ###############################################################

                else:

                    # Initialisierung
                    val_list = []

                    # ZEITSTEMPEL DER TRANSFERIERUNG ##########################
                    try:
                        utc_th = pd.date_range(start    = utc_th_strt,
                                                 end    = utc_th_end,
                                                 freq   = f'{o_dat_inf.loc[key, "delt_transf"]}min'
                                                 ).to_list()
                    except:

                        # Berechne timedelta
                        delt = pd.to_timedelta(o_dat_inf.loc[key, "delt_transf"], unit = "min")

                        # Erzeuge Zeitreihe manuell
                        utc_th = []
                        utc = utc_th_strt
                        for i1 in range(MTS.O_N):
                            utc_th.append(utc)
                            utc += delt

                    # TRANSFERIERUNG DURCH LINEARE INTERPOLATION ##############
                    if o_dat_inf.loc[key, "meth"] == "Lineare Interpolation":

                        # Schleife über den Zeitstempel der Transferierung
                        for i1 in range(len(utc_th)):

                            # Erster Index
                            idx1 = utc_idx_pre(o_dat[key], utc_th[i1])

                            # Zweiter Index
                            idx2 = utc_idx_post(o_dat[key], utc_th[i1])

                            # Kontrolle der Zeitgrenzen
                            if idx1 is None or idx2 is None:
                                error = True
                                break

                            if idx1 == idx2:
                                 val = o_dat[key].iloc[idx1,1]
                            else:
                                utc1 = o_dat[key].iloc[idx1,0]
                                utc2 = o_dat[key].iloc[idx2,0]

                                val1 = o_dat[key].iloc[idx1,1]
                                val2 = o_dat[key].iloc[idx2,1]

                                val = (utc_th[i1]-utc1)/(utc2-utc1)*(val2-val1)+val1

                            # Kontrolle, ob der Wert eine Zahl ist
                            if math.isnan(float(val)):
                                error = True
                                break

                            else:
                                val_list.append(val)

                        if error == False:

                            df_int_o[key] = val_list

                        else:
                            break

                    # TRANSFERIERUNG DURCH MITTELWERTBILDUNG ######################
                    elif o_dat_inf.loc[key, "meth"] == "Mittelwertbildung":
                        print("MUSS NOCH PROGRAMMIERT WERDEN!")

                    # TRANSFERIERUNG DURCH NÄCHSTER WERT ##########################
                    elif o_dat_inf.loc[key, "meth"] == "Nächster Wert":
                        print("MUSS NOCH PROGRAMMIERT WERDEN!")

            #######################################################################
            #######################################################################
            # HISTORISCHE PROGNOSEN ###############################################
            #######################################################################
            #######################################################################
            elif i_dat_inf.loc[key, "spec"] == "Historische Prognosen":
                print("MUSS NOCH PROGRAMMIERT WERDEN!")

    # Nastavak obrade vremenskih značajki - vidi time_features_processing.py
