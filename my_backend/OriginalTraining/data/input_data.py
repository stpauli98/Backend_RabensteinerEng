"""
Input Data - Inicijalizacija i učitavanje ulaznih podataka
Ekstrahirano iz training_original.py linije 696-791

Sadrži:
- Inicijalizaciju i_dat rječnika i i_dat_inf DataFrame-a
- Učitavanje CSV datoteka
- Konfiguraciju parametara za svaki ulazni podatak
"""

import pandas as pd
from utils.data_loader import load
from utils.time_utils import transf
from config.mts import MTS

###############################################################################
###############################################################################
# EINGABEDATEN ################################################################
###############################################################################
###############################################################################

# Initialisierung
i_dat = {}
i_dat_inf = pd.DataFrame(columns = [
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
path = "data/historical/solarthermics/data_4/Wert 1.csv"
name = "Netzlast [kW]"

# Laden der Datei
i_dat[name] = pd.read_csv(path, delimiter = ";")
del name, path

# Laden der Information und UTC-String in Datetime umwandeln
i_dat, i_dat_inf = load(i_dat, i_dat_inf)

# LADEN DER ZWEITEN DATEI #####################################################

# Aussentemperatur von Krumpendorf
path = "data/historical/grid load/data_4/t_out_grad_C_Krumpendorf.csv"
path = "data/historical/solarthermics/data_4/Wert 2.csv"
name = "Aussentemperatur Krumpendorf [GradC]"

# Laden der Datei
i_dat[name] = pd.read_csv(path, delimiter = ";")
del name, path

# Laden der Information und UTC-String in Datetime umwandeln
i_dat, i_dat_inf = load(i_dat, i_dat_inf)

# LÖSCHEN EINES EINTRAGS ######################################################

"""

del_name = "Aussentemperatur Krumpendorf [GradC]"" # Zeile welche gelöscht werden soll

del i_dat[del_name]
i_dat_inf = i_dat_inf.drop(del_name)
"""

# Folgende zwei Zeilen müssen eingefügt werde, um eine Warnung zu vermeiden
i_dat_inf["spec"] = i_dat_inf["spec"].astype("object")
i_dat_inf["meth"] = i_dat_inf["meth"].astype("object")

# AUSFÜLLEN ###################################################################

i_dat_inf.loc["Netzlast [kW]", "spec"]                          = "Historische Daten"
i_dat_inf.loc["Aussentemperatur Krumpendorf [GradC]", "spec"]   = "Historische Daten"

i_dat_inf.loc["Netzlast [kW]", "th_strt"]                           = -1
i_dat_inf.loc["Aussentemperatur Krumpendorf [GradC]", "th_strt"]    = 0

i_dat_inf.loc["Netzlast [kW]", "th_end"]                        = 0
i_dat_inf.loc["Aussentemperatur Krumpendorf [GradC]", "th_end"] = 1

i_dat_inf.loc["Netzlast [kW]", "meth"]                          = "Lineare Interpolation"
i_dat_inf.loc["Aussentemperatur Krumpendorf [GradC]", "meth"]   = "Lineare Interpolation"

i_dat_inf.loc["Netzlast [kW]", "scal"]     = True
i_dat_inf.loc["Netzlast [kW]", "scal_max"] = 1
i_dat_inf.loc["Netzlast [kW]", "scal_min"] = 0

i_dat_inf.loc["Aussentemperatur Krumpendorf [GradC]", "scal"]     = True
i_dat_inf.loc["Aussentemperatur Krumpendorf [GradC]", "scal_max"] = 1
i_dat_inf.loc["Aussentemperatur Krumpendorf [GradC]", "scal_min"] = 0

# ZEITSCHRITTWEITE UND OFFSET DER TRANSFERIERTEN DATEN ########################

i_dat_inf = transf(i_dat_inf, MTS.I_N, MTS.OFST)
