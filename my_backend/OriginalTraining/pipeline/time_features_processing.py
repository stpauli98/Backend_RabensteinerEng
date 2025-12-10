"""
Time Features Processing - Obrada vremenskih značajki
Ekstrahirano iz training_original.py linije 1369-1748

Sadrži obradu cikličkih vremenskih komponenti:
- Y: Godišnja sinus/kosinus komponenta
- M: Mjesečna sinus/kosinus komponenta
- W: Tjedna sinus/kosinus komponenta
- D: Dnevna sinus/kosinus komponenta
- H: Praznici

Svaka komponenta podržava:
- Zeithorizont: Vremenski horizont s više točaka
- Aktuelle Zeit: Trenutno vrijeme (jedna vrijednost)
- Bezug auf UTC/Lokale Zeit: Referenca na UTC ili lokalno vrijeme
"""

import datetime
import calendar
import numpy as np
import pandas as pd
import pytz

from config.mts import MTS
from config.time_features import T
from config.holidays import HOL

###############################################################################
###########################################################################
###########################################################################
# ZEITINFORMATION #########################################################
###########################################################################
###########################################################################
###########################################################################

# Napomena: Ovaj kod se izvršava unutar while petlje iz dataset_creation.py
# Pretpostavlja da su definirane varijable: error, df_int_i, utc_ref

if error == False:

    #######################################################################
    # JAHRESZEITLICHE SINUS-/COSINUS-KOMPONENTE ###########################
    #######################################################################
    if T.Y.IMP:

        # ZEITHORIZONT ####################################################
        if T.Y.SPEC == "Zeithorizont":

            # ZEITGRENZEN
            utc_th_strt = utc_ref+datetime.timedelta(hours = T.Y.TH_STRT)
            utc_th_end  = utc_ref+datetime.timedelta(hours = T.Y.TH_END)

            # ZEITSTEMPEL
            try:
                utc_th = pd.date_range(start  = utc_th_strt,
                                       end    = utc_th_end,
                                       freq   = f'{T.Y.DELT}min'
                                       ).to_list()
            except:

                # Berechne timedelta
                delt = pd.to_timedelta(T.Y.DELT, unit = "min")

                # Erzeuge Zeitreihe manuell
                utc_th = []
                utc = utc_th_strt
                for i1 in range(MTS.I_N):
                    utc_th.append(utc)
                    utc += delt

            # BEZUG AUF UTC
            if T.Y.LT == False:

                # Sekundenzeitstempel erzeugen
                sec = pd.Series(utc_th).map(pd.Timestamp.timestamp)

                df_int_i["y_sin"] = np.sin(sec/31557600*2*np.pi) # 31557600 = 60×60×24×365.25
                df_int_i["y_cos"] = np.cos(sec/31557600*2*np.pi) # 31557600 = 60×60×24×365.25

            # BEZUG AUF LOKALE ZEIT
            else:

                utc_th = [pytz.utc.localize(dt) for dt in utc_th]
                lt_th = [dt.astimezone(pytz.timezone(T.TZ)) for dt in utc_th]

                # Sekundenzeitstempel erzeugen
                sec = np.array([(dt.timetuple().tm_yday-1)*86400+
                                dt.hour*3600+
                                dt.minute*60+
                                dt.second for dt in lt_th])

                # Jahre als NumPy-Array
                y = np.array([x.year for x in lt_th])

                # Vektorisierte Schaltjahresprüfung
                is_leap = np.vectorize(calendar.isleap)(y)

                # Anzahl der Sekunden des Jahres
                sec_y = np.where(is_leap, 31622400, 31536000)

                df_int_i["y_sin"] = np.sin(sec/sec_y*2*np.pi)
                df_int_i["y_cos"] = np.cos(sec/sec_y*2*np.pi)

        # AKTUELLE ZEIT ###################################################
        elif T.Y.SPEC == "Aktuelle Zeit":

            # BEZUG AUF UTC
            if T.Y.LT == False:
                sec = utc_ref.timestamp()
                df_int_i["y_sin"] = np.sin(sec/31557600*2*np.pi) # 31557600 = 60×60×24×365.25
                df_int_i["y_cos"] = np.cos(sec/31557600*2*np.pi) # 31557600 = 60×60×24×365.25

            # BEZUG AUF LOKALE ZEIT
            else:
                lt = pytz.utc.localize(utc_ref).astimezone(pytz.timezone(T.TZ))
                sec = (lt.timetuple().tm_yday-1)*86400+lt.hour*3600+lt.minute*60+lt.second

                # Anzahl der Sekunden des Jahres
                if calendar.isleap(lt.year):
                    sec_y = 31622400 # 31622400 = 60×60×24×366
                else:
                    sec_y = 31536000 # 31536000 = 60×60×24×365

                df_int_i["y_sin"] = np.sin(sec/sec_y*2*np.pi)
                df_int_i["y_cos"] = np.cos(sec/sec_y*2*np.pi)

    #######################################################################
    # MONATLICHE SINUS-/COSINUS-KOMPONENTE ################################
    #######################################################################
    if T.M.IMP:

        # ZEITHORIZONT ####################################################
        if T.M.SPEC == "Zeithorizont":

            # ZEITGRENZEN
            utc_th_strt = utc_ref+datetime.timedelta(hours = T.M.TH_STRT)
            utc_th_end  = utc_ref+datetime.timedelta(hours = T.M.TH_END)

            # ZEITSTEMPEL
            try:
                utc_th = pd.date_range(start  = utc_th_strt,
                                       end    = utc_th_end,
                                       freq   = f'{T.M.DELT}min'
                                       ).to_list()
            except:

                # Berechne timedelta
                delt = pd.to_timedelta(T.M.DELT, unit = "min")

                # Erzeuge Zeitreihe manuell
                utc_th = []
                utc = utc_th_strt
                for i1 in range(MTS.I_N):
                    utc_th.append(utc)
                    utc += delt

            # BEZUG AUF UTC
            if T.M.LT == False:

                # Sekundenzeitstempel erzeugen
                sec = pd.Series(utc_th).map(pd.Timestamp.timestamp)

                df_int_i["m_sin"] = np.sin(sec/2629800*2*np.pi) # 2629800 = 60×60×24×365.25/12
                df_int_i["m_cos"] = np.cos(sec/2629800*2*np.pi) # 2629800 = 60×60×24×365.25/12

            # BEZUG AUF LOKALE ZEIT
            else:

                utc_th = [pytz.utc.localize(dt) for dt in utc_th]
                lt_th = [dt.astimezone(pytz.timezone(T.TZ)) for dt in utc_th]

                # Sekundenzeitstempel erzeugen
                sec = np.array([(dt.day-1)*86400+
                                dt.hour*3600+
                                dt.minute*60+
                                dt.second for dt in lt_th])


                # Extrahiere Jahre und Monate als NumPy-Arrays
                years = np.array([x.year for x in lt_th])
                months = np.array([x.month for x in lt_th])

                # Anzahl der Sekunden des Monats
                sec_m = 86400*np.array([calendar.monthrange(y, m)[1] for y, m in zip(years, months)])

                df_int_i["y_sin"] = np.sin(sec/sec_m*2*np.pi)
                df_int_i["y_cos"] = np.cos(sec/sec_m*2*np.pi)

        # AKTUELLE ZEIT ###################################################
        elif T.M.SPEC == "Aktuelle Zeit":

            # BEZUG AUF UTC
            if T.M.LT == False:
                sec = utc_ref.timestamp()
                df_int_i["w_sin"] = np.sin(sec/2629800*2*np.pi) # 2629800 = 60×60×24×365.25/12
                df_int_i["w_cos"] = np.cos(sec/2629800*2*np.pi) # 2629800 = 60×60×24×365.25/12

            # BEZUG AUF LOKALE ZEIT
            else:

                lt = pytz.utc.localize(utc_ref).astimezone(pytz.timezone(T.TZ))
                sec = (lt.day-1)*86400+lt.hour*3600+lt.minute*60+lt.second

                # Anzahl der Sekunden des Monats
                sec_m = calendar.monthrange(lt.year, lt.month)[1]*86400

                df_int_i["m_sin"] = np.sin(sec/sec_m*2*np.pi)
                df_int_i["m_cos"] = np.cos(sec/sec_m*2*np.pi)

    #######################################################################
    # WÖCHENTLICHE SINUS-/COSINUS-KOMPONENTE ##############################
    #######################################################################
    if T.W.IMP:

        # ZEITHORIZONT ####################################################
        if T.W.SPEC == "Zeithorizont":

            # ZEITGRENZEN
            utc_th_strt = utc_ref+datetime.timedelta(hours = T.W.TH_STRT)
            utc_th_end  = utc_ref+datetime.timedelta(hours = T.W.TH_END)

            # ZEITSTEMPEL
            try:
                utc_th = pd.date_range(start  = utc_th_strt,
                                       end    = utc_th_end,
                                       freq   = f'{T.W.DELT}min'
                                       ).to_list()
            except:

                # Berechne timedelta
                delt = pd.to_timedelta(T.W.DELT, unit = "min")

                # Erzeuge Zeitreihe manuell
                utc_th = []
                utc = utc_th_strt
                for i1 in range(MTS.I_N):
                    utc_th.append(utc)
                    utc += delt

            # BEZUG AUF UTC
            if T.W.LT == False:

                # Sekundenzeitstempel erzeugen
                sec = np.array([dt.timestamp() for dt in utc_th])

                df_int_i["w_sin"] = np.sin(sec/604800*2*np.pi) # 604800 = 60×60×24×7
                df_int_i["w_cos"] = np.cos(sec/604800*2*np.pi) # 604800 = 60×60×24×7

            # BEZUG AUF LOKALE ZEIT
            else:

                utc_th = [pytz.utc.localize(dt) for dt in utc_th]
                lt_th = [dt.astimezone(pytz.timezone(T.TZ)) for dt in utc_th]

                # Sekundenzeitstempel erzeugen
                sec = np.array([dt.weekday()*86400+
                                dt.hour*3600+
                                dt.minute*60+
                                dt.second for dt in lt_th])

                df_int_i["w_sin"] = np.sin(sec/604800*2*np.pi) # 604800 = 60×60×24×7
                df_int_i["w_cos"] = np.cos(sec/604800*2*np.pi) # 604800 = 60×60×24×7

        # AKTUELLE ZEIT ###################################################
        elif T.W.SPEC == "Aktuelle Zeit":

            # BEZUG AUF UTC
            if T.W.LT == False:
                sec = utc_ref.timestamp()
                df_int_i["w_sin"] = np.sin(sec/604800*2*np.pi) # 604800 = 60×60×24×7
                df_int_i["w_cos"] = np.cos(sec/604800*2*np.pi) # 604800 = 60×60×24×7

            # BEZUG AUF LOKALE ZEIT
            else:
                lt = pytz.utc.localize(utc_ref).astimezone(pytz.timezone(T.TZ))
                sec = lt.weekday()*86400+lt.hour*3600+lt.minute*60+lt.second
                df_int_i["d_sin"] = np.sin(sec/604800*2*np.pi) # 604800 = 60×60×24×7
                df_int_i["d_cos"] = np.cos(sec/604800*2*np.pi) # 604800 = 60×60×24×7

    #######################################################################
    # TÄGLICHE SINUS-/COSINUS-KOMPONENTE ##################################
    #######################################################################
    if T.D.IMP:

        # ZEITHORIZONT ####################################################
        if T.D.SPEC == "Zeithorizont":

            # ZEITGRENZEN
            utc_th_strt = utc_ref+datetime.timedelta(hours = T.D.TH_STRT)
            utc_th_end  = utc_ref+datetime.timedelta(hours = T.D.TH_END)

            # ZEITSTEMPEL
            try:
                utc_th = pd.date_range(start  = utc_th_strt,
                                       end    = utc_th_end,
                                       freq   = f'{T.D.DELT}min'
                                       ).to_list()
            except:

                # Berechne timedelta
                delt = pd.to_timedelta(T.D.DELT, unit = "min")

                # Erzeuge Zeitreihe manuell
                utc_th = []
                utc = utc_th_strt
                for i1 in range(MTS.I_N):
                    utc_th.append(utc)
                    utc += delt

            # BEZUG AUF UTC
            if T.D.LT == False:

                # Sekundenzeitstempel erzeugen
                sec = np.array([dt.timestamp() for dt in utc_th])

                df_int_i["d_sin"] = np.sin(sec/86400*2*np.pi) # 86400 = 60×60×24
                df_int_i["d_cos"] = np.cos(sec/86400*2*np.pi) # 86400 = 60×60×24

            # BEZUG AUF LOKALE ZEIT
            else:

                utc_th = [pytz.utc.localize(dt) for dt in utc_th]
                lt_th = [dt.astimezone(pytz.timezone(T.TZ)) for dt in utc_th]

                # Sekundenzeitstempel erzeugen
                sec = np.array([dt.hour*3600+
                                dt.minute*60+
                                dt.second for dt in lt_th])

                df_int_i["d_sin"] = np.sin(sec/86400*2*np.pi) # 86400 = 60×60×24
                df_int_i["d_cos"] = np.cos(sec/86400*2*np.pi) # 86400 = 60×60×24

        # AKTUELLE ZEIT ###################################################
        elif T.D.SPEC == "Aktuelle Zeit":

            # BEZUG AUF UTC
            if T.D.LT == False:
                sec = utc_ref.timestamp()
                df_int_i["d_sin"] = np.sin(sec/86400*2*np.pi) # 86400 = 60×60×24
                df_int_i["d_cos"] = np.cos(sec/86400*2*np.pi) # 86400 = 60×60×24

            # BEZUG AUF LOKALE ZEIT
            else:
                lt = pytz.utc.localize(utc_ref).astimezone(pytz.timezone(T.TZ))
                sec = lt.hour*3600+lt.minute*60+lt.second
                df_int_i["d_sin"] = np.sin(sec/86400*2*np.pi) # 86400 = 60×60×24
                df_int_i["d_cos"] = np.cos(sec/86400*2*np.pi) # 86400 = 60×60×24

    #######################################################################
    # FEIERTAGE ###########################################################
    #######################################################################
    if T.H.IMP:

        # Set mit Datumsobjekten der Feiertage (welche kein Sonntag sind)
        hol_d = set(d.date() for d in HOL[T.H.CNTRY])

        # ZEITHORIZONT ####################################################
        if T.H.SPEC == "Zeithorizont":

            # ZEITGRENZEN
            utc_th_strt = utc_ref+datetime.timedelta(hours = T.H.TH_STRT)
            utc_th_end  = utc_ref+datetime.timedelta(hours = T.H.TH_END)

            # ZEITSTEMPEL
            try:
                utc_th = pd.date_range(start  = utc_th_strt,
                                       end    = utc_th_end,
                                       freq   = f'{T.H.DELT}min'
                                       ).to_list()
            except:

                # Berechne timedelta
                delt = pd.to_timedelta(T.H.DELT, unit = "min")

                # Erzeuge Zeitreihe manuell
                utc_th = []
                utc = utc_th_strt
                for i1 in range(MTS.I_N):
                    utc_th.append(utc)
                    utc += delt

            # BEZUG AUF UTC
            if T.H.LT == False:

                # Vergleich: Wenn Feiertag und kein Sonntag → 1, sonst 0
                df_int_i["h"] = np.array([1 if dt.date() in hol_d else 0 for dt in utc_th])

            # BEZUG AUF LOKALE ZEIT
            else:

                utc_th = [pytz.utc.localize(dt) for dt in utc_th]
                lt_th = [dt.astimezone(pytz.timezone(T.TZ)) for dt in utc_th]

                # Vergleich: Wenn Feiertag und kein Sonntag → 1, sonst 0
                df_int_i["h"] = np.array([1 if dt.date() in hol_d else 0 for dt in lt_th])

        # AKTUELLE ZEIT ###################################################
        elif T.H.SPEC == "Aktuelle Zeit":

            # BEZUG AUF UTC
            if T.H.LT == False:

                df_int_i["h"] = np.array(1 if utc_ref.date() in hol_d else 0)

            # BEZUG AUF LOKALE ZEIT
            else:

                lt = pytz.utc.localize(utc_ref).astimezone(pytz.timezone(T.TZ))
                df_int_i["h"] = np.array(1 if lt.date() in hol_d else 0)

    i_arrays.append(df_int_i.values)
    o_arrays.append(df_int_o.values)

    utc_ref_log.append(utc_ref)
else:
    error = False

utc_ref = utc_ref+datetime.timedelta(minutes = MTS.DELT)

# Završetak glavne petlje - nastavak u scaling.py
