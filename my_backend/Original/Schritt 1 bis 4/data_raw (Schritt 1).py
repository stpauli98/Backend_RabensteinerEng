###############################################################################
# MODULE ######################################################################
###############################################################################

import pandas as pd
from datetime import datetime
import pytz

###############################################################################
# HILFSFUNKTIONEN #############################################################
###############################################################################

def load_raw_data(LOC, DELIM):
    """
    Lädt und kombiniert CSV-Dateien von den angegebenen Pfaden.
    
        LOC.....Speicherort der csv-Daten
        DELIM...Trennzeichen in den csv-Daten
    
    """
    #df = pd.concat([pd.read_csv(file, delimiter = DELIM) 
    #                for file in reversed(LOC)], ignore_index = True)
    df = pd.concat([pd.read_csv(file, delimiter = DELIM) 
                    for file in LOC], ignore_index = True)
    
    return df

def convert_to_utc(TIME, TIME_ZONE, TIME_FMT_1, TIME_FMT_2):
    """
    Ändert die Zeitzone einer Zeitangabe in UTC.
        TIME........Lokale Zeitangabe [String]
        TIME_ZONE...Zeitzone der lokalen Zeit
        TIME_FMT_1..Zeitformat der lokalen Zeitangabe
        TIME_FMT_2..Zeitformat der verarbeiteten Daten
    
    """
    
    local_time = (datetime.strptime(TIME, TIME_FMT_1)).replace(tzinfo = None)
    
    # Lokale Zeit mit Zeitzone versehen
    localized_time = TIME_ZONE.localize(local_time)
    
    # In UTC umwandeln
    utc_time = localized_time.astimezone(pytz.utc) 
    
    return utc_time.strftime(TIME_FMT_2)

###############################################################################
# EINGABEN ####################################################################
###############################################################################

# INFORMATIONEN ZU DEN ROHDATEN ###############################################
class RAW:
    
    """
    NETZLAST VON KRUMPENDORF [MW]
    """

    # Speicherort
    # LOC = ["historical/raw/FWN(t01).csv",    # MEZ/MESZ 2022-01-01 00:00:00 - 2022-07-13 11:30:00 (delta_time = 3 min)
    #        "historical/raw/FWN(t02).csv",    # MEZ/MESZ 2022-07-13 11:33:00 - 2023-09-13 15:24:00 (delta_time = 3 min)
    #        "historical/raw/FWN(t03).csv"]    # MEZ/MESZ 2023-09-13 15:27:00 - 2024-11-08 11:36:00 (delta_time = 3 min)
    
    LOC = ["historical/raw/FWN(t01).csv",   # MEZ/MESZ 2022-01-01 00:00:00 - 2022-07-13 11:30:00 (delta_time = 3 min)
           "historical/raw/FWN(t02).csv"]   # MEZ/MESZ 2022-07-13 11:33:00 - 2023-09-13 15:24:00 (delta_time = 3 min)

    
    #LOC = ["historical/raw/FWN(t01).csv"]
    
    # Trennzeichen
    DELIM = ";"
    
    # Zeitformat
    TIME_FMT = "%Y-%m-%d %H:%M:%S"
    
    # Zeitzone
    TIME_ZONE = pytz.timezone("Europe/Vienna")
    
    # Spaltenbezeichnung mit Zeitinformation
    TIME_LBL = "Datum"
    
    # Spaltenbezeichnung mit Messwert
    #VALUE_LBL = "Wärmezähler.FW Netz.Leistung_aktuell"
    VALUE_LBL = "Wärmezähler.FW Netz.Rücklauftemperatur"
    
    """
    AUSSENTEMPERATUR VON PÖRTSCHACH [°C]
    """
    
    # LOC = ["historical/raw/t_outside_GeoSphere.csv"]    # Messdaten (Lufttemperatur 2m) von 
    #                                                     # https://data.hub.geosphere.at/dataset/klima-v2-1h für Messstation in
    #                                                     # Pörtschach (ID 20220) UTC wird als Zeitstempel verwendet
    #                                                     # UTC 2022-01-01 00:00:00 - 2024-11-08 11:00:00 (delta_time = 60 min)
    
    # # Trennzeichen
    # DELIM = ","
    
    # # Zeitformat
    # TIME_FMT = "%Y-%m-%dT%H:%M%z"
    
    # # Zeitzone
    # TIME_ZONE = pytz.timezone("UTC")
    
    # # Spaltenbezeichnung mit Zeitinformation
    # TIME_LBL = "time"
    
    # # Spaltenbezeichnung mit Messwert
    # VALUE_LBL = "tl"
       
    
# INFORMATION ZU DEN VERARBEITETEN DATEN ######################################
class FILE:
    
    # Name der csv-Datei
    #NAME = "load_grid_kW_Krumpendorf"
    #NAME = "t_out_grad_C_Krumpendorf"
    NAME = "t_RL"
    
    # Zeitformat
    TIME_FMT = "%Y-%m-%d %H:%M:%S"

###############################################################################
# HAUPTPROGRAMM ###############################################################
###############################################################################

# Laden der Rohdaten
df = load_raw_data(RAW.LOC, RAW.DELIM)

# Anwendung der Funktion auf die Zeitspalte
df[RAW.TIME_LBL] = df[RAW.TIME_LBL].apply(lambda x: convert_to_utc(x, 
                                                                   RAW.TIME_ZONE, 
                                                                   RAW.TIME_FMT, 
                                                                   FILE.TIME_FMT))

# Umbenennen der Spalte mit der Zeitinformation
df.rename(columns = {RAW.TIME_LBL: "UTC"}, 
          inplace = True)

df = df[["UTC", RAW.VALUE_LBL]]
"""
df.rename(columns = {RAW.VALUE_LBL: "load_grid [MW]"}, 
          inplace = True)

df["load_grid [kW]"] = df["load_grid [MW]"] * 1000

df = df.drop(columns = ["load_grid [MW]"])
"""
df.rename(columns = {RAW.VALUE_LBL: "t_VL [°C]"}, 
          inplace = True)

# Daten speichern
df.to_csv("historical/data_1/"+FILE.NAME+".csv", 
              index = False, 
              sep = ";")