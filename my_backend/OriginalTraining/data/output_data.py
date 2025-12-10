"""
Output Data - Inicijalizacija i učitavanje izlaznih podataka
Ekstrahirano iz training_original.py linije 960-1048

Sadrži:
- Inicijalizaciju o_dat rječnika i o_dat_inf DataFrame-a
- Učitavanje CSV datoteka
- Konfiguraciju parametara za svaki izlazni podatak
"""

import pandas as pd
from utils.data_loader import load
from utils.time_utils import transf
from config.mts import MTS

###############################################################################
###############################################################################
# AUSGABEDATEN ################################################################
###############################################################################
###############################################################################

# Initialisierung
o_dat = {}
o_dat_inf = pd.DataFrame(columns = [
    "utc_min",
    "utc_max",
    "delt",
    "ofst",
    "n_all",
    "n_num",
    "rate_num",
    "val_min",
    "val_max",
    "spec",
    "th_strt",
    "th_end",
    "meth",
    "avg",
    "delt_transf",
    "ofst_transf",
    "scal",
    "scal_max",
    "scal_min"
    ])

# LADEN DER ERSTEN DATEI ######################################################

# Netzlast vom HW Krumpendorf
path = "data/historical/grid load/data_4/load_grid_kW_Krumpendorf.csv"
path = "data/historical/solarthermics/data_4/Wert 2.csv"
name = "Netzlast [kW]"

# Laden der Datei
o_dat[name] = pd.read_csv(path, delimiter = ";")
del name, path

# Laden der Information und UTC-String in Datetime umwandeln
o_dat, o_dat_inf = load(o_dat, o_dat_inf)

# LADEN DER ZWEITEN DATEI ######################################################

# Netzlast vom HW Krumpendorf
path = "data/historical/grid load/data_4/load_grid_kW_Krumpendorf.csv"
path = "data/historical/solarthermics/data_4/Wert 3.csv"
name = "a"

# Laden der Datei
o_dat[name] = pd.read_csv(path, delimiter = ";")
del name, path

# Laden der Information und UTC-String in Datetime umwandeln
o_dat, o_dat_inf = load(o_dat, o_dat_inf)

# Folgende zwei Zeilen müssen eingefügt werde, um eine Warnung zu vermeiden
o_dat_inf["spec"] = o_dat_inf["spec"].astype("object")
o_dat_inf["meth"] = o_dat_inf["meth"].astype("object")

# AUSFÜLLEN ###################################################################

o_dat_inf.loc["Netzlast [kW]", "spec"] = "Historische Daten"

o_dat_inf.loc["Netzlast [kW]", "th_strt"] = 0

o_dat_inf.loc["Netzlast [kW]", "th_end"] = 1

o_dat_inf.loc["Netzlast [kW]", "meth"] = "Lineare Interpolation"

o_dat_inf.loc["Netzlast [kW]", "scal"] = True
o_dat_inf.loc["Netzlast [kW]", "scal_max"] = 1
o_dat_inf.loc["Netzlast [kW]", "scal_min"] = 0

o_dat_inf.loc["a", "spec"] = "Historische Daten"

o_dat_inf.loc["a", "th_strt"] = 0

o_dat_inf.loc["a", "th_end"] = 1.5

o_dat_inf.loc["a", "meth"] = "Lineare Interpolation"

o_dat_inf.loc["a", "scal"] = True
o_dat_inf.loc["a", "scal_max"] = 1
o_dat_inf.loc["a", "scal_min"] = 0


# ZEITSCHRITTWEITE UND OFFSET DER TRANSFERIERTEN DATEN ########################

o_dat_inf = transf(o_dat_inf, MTS.O_N, MTS.OFST)
