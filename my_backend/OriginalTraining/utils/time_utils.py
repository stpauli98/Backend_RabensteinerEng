"""
Time Utils - Vremenske utility funkcije
Ekstrahirano iz training_original.py linije 111-167

Sadrži funkcije za:
- transf: Izračun vremenskog koraka i offseta transferiranih podataka
- utc_idx_pre: Pronalaženje prethodnog indeksa za UTC vrijeme
- utc_idx_post: Pronalaženje sljedećeg indeksa za UTC vrijeme
"""

import math

###############################################################################
# FUNKTION ZUR BERECHNUNG DER ZEITSCHRITTWEITE UND DES OFFSETS DER ############
# TRANSFERIERTEN DATEN ########################################################
###############################################################################

def transf (inf, N, OFST):

    for i in range(len(inf)):

        key = inf.index[i]

        inf.loc[key, "delt_transf"] = \
            (inf.loc[key, "th_end"]-\
             inf.loc[key, "th_strt"])*60/(N-1)

        # OFFSET KANN BERECHNET WERDEN
        if round(60/inf.loc[key, "delt_transf"]) == \
            60/inf.loc[key, "delt_transf"]:

            # Offset [min]
            ofst_transf = OFST-(inf.loc[key, "th_strt"]-
                                math.floor(inf.loc[key, "th_strt"]))*60+60

            while ofst_transf-inf.loc[key, "delt_transf"] >= 0:
               ofst_transf -= inf.loc[key, "delt_transf"]


            inf.loc[key, "ofst_transf"] = ofst_transf

        # OFFSET KANN NICHT BERECHNET WERDEN
        else:
            inf.loc[key, "ofst_transf"] = "var"

    return inf

###############################################################################
# FUNKTION ZUR ERMITTLUNG DES VORHERIGEN INDEX ################################
###############################################################################

def utc_idx_pre(dat, utc):

    # Index des ersten Elements, das kleinergleich "utc" ist
    idx = dat["UTC"].searchsorted(utc, side = 'right')

    # Ausgabe des Wertes
    if idx > 0:
        return dat.index[idx-1]

    # Kein passender Eintrag
    return None

###############################################################################
# FUNKTION ZUR ERMITTLUNG DES NACHFOLGENDEN INDEX #############################
###############################################################################

def utc_idx_post(dat, utc):

    # Index des ersten Elements, das größergleich "utc" ist
    idx = dat["UTC"].searchsorted(utc, side = 'left')

    # Ausgabe des Wertes
    if idx < len(dat):
        return dat.index[idx]

    # Kein passender Eintrag
    return None
