"""
Data Loader - Funkcija za učitavanje podataka
Ekstrahirano iz training_original.py linije 36-109

Funkcija load() učitava i analizira dataframe,
izračunava statistike i dodaje ih u info dataframe.
"""

import pandas as pd
import math

###############################################################################
# FUNKTION ZUR AUSGABE DER INFORMATIONEN ######################################
###############################################################################

def load (dat, inf):

    # Zuletzt geladener Dataframe
    df_name, df = next(reversed(dat.items()))

    # UTC in datetime umwandeln
    df["UTC"] = pd.to_datetime(df["UTC"],
                               format = "%Y-%m-%d %H:%M:%S")

    # Startzeit
    utc_min = df["UTC"].iloc[0]

    # Endzeit
    utc_max = df["UTC"].iloc[-1]

    # Anzahl der Datenpunkte
    n_all = len(df)

    # Zeitschrittweite [min]
    delt = (df["UTC"].iloc[-1]-df["UTC"].iloc[0]).total_seconds()/(60*(n_all-1))

    # Konstanter Offset
    if round(60/delt) == 60/delt:

        ofst = (df["UTC"].iloc[0]-
                (df["UTC"].iloc[0]).replace(minute      = 0,
                                            second      = 0,
                                            microsecond = 0)).total_seconds()/60
        while ofst-delt >= 0:
           ofst -= delt

    # Variabler Offset
    else:

        ofst = "var"

    # Anzahl der numerischen Datenpunkte
    n_num = n_all
    for i in range(n_all):
        try:
            float(df.iloc[i, 1])
            if math.isnan(float(df.iloc[i, 1])):
               n_num -= 1
        except:
            n_num -= 1

    # Anteil an numerischen Datenpunkten [%]
    rate_num = round(n_num/n_all*100, 2)

    # Maximalwert [#]
    val_max = df.iloc[:, 1].max()

    # Minimalwert [#]
    val_min = df.iloc[:, 1].min()

    # Dataframe aktualisieren
    dat[df_name] = df

    # Information einfügen
    inf.loc[df_name] = {
        "utc_min":  utc_min,
        "utc_max":  utc_max,
        "delt":     delt,
        "ofst":     ofst,
        "n_all":    n_all,
        "n_num":    n_num,
        "rate_num": rate_num,
        "val_min":  val_min,
        "val_max":  val_max,
        "scal":     False,
        "avg":      False}

    return dat, inf
