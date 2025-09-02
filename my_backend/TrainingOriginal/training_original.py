###############################################################################
###############################################################################
# MODULE ######################################################################
###############################################################################
###############################################################################

import datetime
import math
import pandas as pd
import sys
import random
import numpy as np
import pytz
import copy
import calendar
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

###############################################################################
###############################################################################
# FUNKTIONEN ##################################################################
###############################################################################
###############################################################################

# FUNKTION ZUR AUSGABE DER INFORMATIONEN ######################################
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

# FUNKTION ZUR BERECHNUNG DER ZEITSCHRITTWEITE UND DES OFFSETS DER ############
# TRANSFERIERTEN DATEN ########################################################
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

# FUNKTION ZUR ERMITTLUNG DES VORHERIGEN INDEX ################################
def utc_idx_pre(dat, utc):
        
    # Index des ersten Elements, das kleinergleich "utc" ist
    idx = dat["UTC"].searchsorted(utc, side = 'right')

    # Ausgabe des Wertes
    if idx > 0:
        return dat.index[idx-1]

    # Kein passender Eintrag
    return None    

# FUNKTION ZUR ERMITTLUNG DES NACHFOLGENDEN INDEX #############################
def utc_idx_post(dat, utc):

    # Index des ersten Elements, das größergleich "utc" ist
    idx = dat["UTC"].searchsorted(utc, side = 'left')

    # Ausgabe des Wertes
    if idx < len(dat):
        return dat.index[idx]

    # Kein passender Eintrag
    return None

# FUNKTION ZUM TRAINIEREN UND VALIDIEREN EINES DENSE-LAYER-MODELLS ############
def train_dense(train_x, train_y, val_x, val_y, MDL):    
    """
    Funktion trainiert und validiert ein Neuronales Netz anhand der 
    eingegebenen Trainingsdaten (train_x, train_y) und Validierungsdaten 
    (val_x, val_y).
    
    train_x...Trainingsdaten (Eingabedaten)
    train_y...Trainingsdaten (Ausgabedaten)
    val_x.....Validierungsdaten (Eingabedaten)
    val_y.....Validierungsdaten (Ausgabedaten)
    MDL.......Informationen zum Modell
    """
       
    # MODELLDEFINITION ########################################################
    
    # Modellinitialisierung (Sequentielles Modell mit linear 
    # hintereinandergeordneten Schichten)
    model = tf.keras.Sequential()
    
    # Input-Schicht → Mehrdimensionale Daten werden in einen 1D-Vektor 
    # umgewandelt
    model.add(tf.keras.layers.Flatten())
    
    # Dense-Layer hinzufügen
    for _ in range(MDL.LAY):
        model.add(tf.keras.layers.Dense(MDL.N,                  # Anzahl an Neuronen
                                        activation = MDL.ACTF)) # Aktivierungsfunktion
    
    # Output-Schicht
    model.add(tf.keras.layers.Dense(train_y.shape[1]*train_y.shape[2], 
                                  kernel_initializer = tf.initializers.zeros))
    model.add(tf.keras.layers.Reshape([train_y.shape[1], train_y.shape[2]]))
    
    """
    Folgender Callback sorgt dafür, dass das Training vorzeitig gestoppt wird, 
    wenn sich die Leistung auf den Validierungsdaten (val_loss) nach einer 
    bestimmten Anzahl von Epochen nicht verbessert. Dies hilft, Overfitting zu 
    vermeiden und das Training effizienter zu gestalten.
    """
    
    earlystopping = tf.keras.callbacks.\
        EarlyStopping(monitor  = "val_loss", 
        mode                   = "min", 
        patience               = 2, 
        restore_best_weights   = True)

    # Konfiguration des Modells für das Training    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss = tf.keras.losses.MeanSquaredError(),
        metrics = [tf.keras.metrics.RootMeanSquaredError()])
    
    # TRAINIEREN ##############################################################
    
    print("Modell wird trainiert.")
    
    model.fit(
        x               = train_x,
        y               = train_y,
        epochs          = MDL.EP,
        verbose         = 1,
        callbacks       = [earlystopping],
        validation_data = (val_x, val_y)
        )
         
    print("Modell wurde trainiert.")
            
    return model

def train_cnn(train_x, train_y, val_x, val_y, MDL):    
    """
    Funktion trainiert und validiert ein CNN anhand der 
    eingegebenen Trainingsdaten (train_x, train_y) und Validierungsdaten 
    (val_x, val_y).
    
    train_x...Trainingsdaten (Eingabedaten)
    train_y...Trainingsdaten (Ausgabedaten)
    val_x.....Validierungsdaten (Eingabedaten)
    val_y.....Validierungsdaten (Ausgabedaten)
    MDL.......Informationen zum Modell
    """
    
    # MODELLDEFINITION ########################################################
    
    # Modellinitialisierung (Sequentielles Modell mit linear 
    # hintereinandergeordneten Schichten)    
    model = tf.keras.Sequential()
       
    # Konverterierung in ein 4D-Array (Conv2D-Layer erwartet Eingabedaten mit vier Dimensionen)
    # - batch_size: Anzahl der Trainingsbeispiele
    # - Höhe und Breite: räumliche Dimensionen deiner Eingabedaten
    # - Kanäle: Anzahl der Kanäle pro Pixel (z. B. 3 für RGB-Bilder, 1 für Graustufen)
    train_x = train_x.reshape(trn_x.shape[0], trn_x.shape[1], trn_x.shape[2], 1)
    
    
    for i in range(MDL.LAY):
        if i == 0:
            # Input-Layer mit Angabe der Input-Shape
            model.add(tf.keras.layers.Conv2D(filters        = MDL.N, 
                                             kernel_size    = MDL.K,
                                             padding        = 'same',
                                             activation     = MDL.ACTF,
                                             input_shape    = train_x.shape[1:]))
        else:
            model.add(tf.keras.layers.Conv2D(filters        = MDL.N, 
                                             kernel_size    = MDL.K,
                                             padding        = 'same',
                                             activation     = MDL.ACTF))
    
    # Output-Layer: Convolution mit 1 Filter (oder Anzahl Kanäle von train_y)
    # und linearer Aktivierung, damit die Ausgabe dieselbe Form wie train_y hat.
    # Falls train_y z.B. (Batch, H, W, C), dann Filteranzahl = C.
    
    output_channels = train_y.shape[-1] if len(train_y.shape) == 4 else 1
    
    model.add(tf.keras.layers.Conv2D(filters            = output_channels,
                                     kernel_size        = 1,
                                     padding            = 'same',
                                     activation         = 'linear',
                                     kernel_initializer = tf.initializers.zeros))
    
    # Callback EarlyStopping
    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor                 = "val_loss",
        mode                    = "min",
        patience                = 2,
        restore_best_weights    = True)
    
    model.compile(
        optimizer   = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss        = tf.keras.losses.MeanSquaredError(),
        metrics     = [tf.keras.metrics.RootMeanSquaredError()]
    )
    
    # TRAINIEREN ##############################################################
    
    print("Modell wird trainiert.")
    MDL.EP = 20
    model.fit(
        x               = train_x,
        y               = train_y,
        epochs          = MDL.EP,
        verbose         = 1,
        callbacks       = [earlystopping],
        validation_data = (val_x, val_y)
    )
    
    print("Modell wurde trainiert.")
    
    return model

def train_lstm(train_x, train_y, val_x, val_y, MDL):
    """
    Funktion trainiert und validiert ein LSTM-Modell anhand der 
    eingegebenen Trainingsdaten (train_x, train_y) und Validierungsdaten 
    (val_x, val_y).
    
    train_x...Trainingsdaten (Eingabedaten)
    train_y...Trainingsdaten (Ausgabedaten)
    val_x.....Validierungsdaten (Eingabedaten)
    val_y.....Validierungsdaten (Ausgabedaten)
    MDL.......Informationen zum Modell
    """
    
    # MODELLDEFINITION ########################################################
    
    # Modellinitialisierung (Sequentielles Modell mit linear 
    # hintereinandergeordneten Schichten)
    model = tf.keras.Sequential()
        
    for i in range(MDL.LAY):
        
        # Alle LSTM-Schichten mit return_sequences = True, um Sequenzen zu erhalten
                
        if i == 0:
            model.add(tf.keras.layers.LSTM(units            = MDL.N,
                                           activation       = MDL.ACTF,
                                           return_sequences = True,
                                           input_shape      = train_x.shape[1:]))
        else:
            model.add(tf.keras.layers.LSTM(units            = MDL.N,
                                           activation       = MDL.ACTF,
                                           return_sequences = True))
    
    # Dense Layer für jedes TimeStep
    output_units = train_y.shape[-1]
    model.add(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(output_units,
                              kernel_initializer = tf.initializers.zeros)
    ))
    
    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor                 = "val_loss",
        mode                    = "min",
        patience                = 2,
        restore_best_weights    = True)
    
    model.compile(
        optimizer   = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss        = tf.keras.losses.MeanSquaredError(),
        metrics     = [tf.keras.metrics.RootMeanSquaredError()]
    )
    
    # TRAINIEREN ##############################################################
    
    print("Modell wird trainiert.")
    
    model.fit(
        x               = train_x,
        y               = train_y,
        epochs          = MDL.EP,
        verbose         = 1,
        callbacks       = [earlystopping],
        validation_data = (val_x, val_y)
    )
    print("Modell wurde trainiert.")
    
    return model

def train_ar_lstm(train_x, train_y, val_x, val_y, MDL):
    """
    Funktion trainiert und validiert ein autoregressives LSTM-Modell anhand der 
    eingegebenen Trainingsdaten (train_x, train_y) und Validierungsdaten 
    (val_x, val_y).
    
    train_x...Trainingsdaten (Eingabedaten)
    train_y...Trainingsdaten (Ausgabedaten)
    val_x.....Validierungsdaten (Eingabedaten)
    val_y.....Validierungsdaten (Ausgabedaten)
    MDL.......Informationen zum Modell
    """
    
    # MODELLDEFINITION ########################################################
    
    # Modellinitialisierung (Sequentielles Modell mit linear 
    # hintereinandergeordneten Schichten)
    model = tf.keras.Sequential()
    
    for i in range(MDL.LAY):
        
        # Alle LSTM-Schichten mit return_sequences = True, um Sequenzen zu erhalten
        
        if i == 0:
            model.add(tf.keras.layers.LSTM(units            = MDL.N,
                                           activation       = MDL.ACTF,
                                           return_sequences = True,
                                           input_shape      = train_x.shape[1:]))
        else:
            model.add(tf.keras.layers.LSTM(units            = MDL.N,
                                           activation       = MDL.ACTF,
                                           return_sequences = True))
    
    # TimeDistributed Dense Layer für Vorhersage pro Zeitschritt
    output_units = train_y.shape[-1] if len(train_y.shape) > 2 else 1
    model.add(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(output_units,
                              kernel_initializer = tf.initializers.zeros)
    ))
    
    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor                 = "val_loss",
        mode                    = "min",
        patience                = 2,
        restore_best_weights    = True)
    
    model.compile(
        optimizer   = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss        = tf.keras.losses.MeanSquaredError(),
        metrics     = [tf.keras.metrics.RootMeanSquaredError()]
    )
    
    # TRAINIEREN ##############################################################
    
    print("Modell wird trainiert.")
    
    model.fit(
        x               = train_x,
        y               = train_y,
        epochs          = MDL.EP,
        verbose         = 1,
        callbacks       = [earlystopping],
        validation_data = (val_x, val_y)
    )
    
    print("Modell wurde trainiert.")
    
    return model

def train_svr_dir(train_x, train_y, MDL):
    """
    Funktion trainiert ein SVR-Modell anhand der 
    eingegebenen Trainingsdaten (train_x, train_y).
    
    train_x...Trainingsdaten (Eingabedaten) [n_samples, n_timesteps, n_features_in]
    train_y...Trainingsdaten (Ausgabedaten) [n_samples, n_timesteps, n_features_out]
    MDL.......Informationen zum Modell
    """
    
    # MODELLDEFINITION ########################################################
    
    n_samples, n_timesteps, n_features = train_x.shape
    X = train_x.reshape(n_samples * n_timesteps, n_features)
    
    y = []
    for i in range(n_features):
        y.append(train_y[:, :, i].reshape(-1))

    # TRAINIEREN ##############################################################

    print("Modell wird trainiert.")
    
    model = []
    for i in range(n_features):
        model.append(make_pipeline(StandardScaler(), 
                                   SVR(kernel  = MDL.KERNEL,
                                       C       = MDL.C, 
                                       epsilon = MDL.EPSILON)))
        model[-1].fit(X, y[i])

    print("Modell wurde trainiert.")  
    
    return model

def train_svr_mimo(train_x, train_y, MDL):
    """
    Funktion trainiert ein SVR-MIMO-Modell anhand der
    eingegebenen Trainingsdaten (train_x, train_y).
    
    train_x...Trainingsdaten (Eingabedaten) [n_samples, n_timesteps, n_features_in]
    train_y...Trainingsdaten (Ausgabedaten) [n_samples, n_timesteps, n_features_out]
    MDL.......Informationen zum Modell
    """

    # MODELLDEFINITION ########################################################

    n_samples, n_timesteps, n_features_in = train_x.shape
    _, _, n_features_out = train_y.shape

    # Eingabedaten 2D umformen: (n_samples * n_timesteps, n_features_in)
    X = train_x.reshape(n_samples * n_timesteps, n_features_in)
    
    # TRAINIEREN ##############################################################

    print("Modell wird trainiert.")

    model = []
    for i in range(n_features_out):
        # Für jedes Ausgabefeature das passende Ziel erstellen
        y_i = train_y[:, :, i].reshape(-1)

        # Pipeline mit StandardScaler + SVR
        svr = make_pipeline(StandardScaler(),
                            SVR(kernel=MDL.KERNEL,
                                C=MDL.C,
                                epsilon=MDL.EPSILON))
        svr.fit(X, y_i)
        model.append(svr)

    print("Modell wurde trainiert.")
    return model
 
def train_linear_model(trn_x, trn_y):
    
    # MODELLDEFINITION ########################################################
    
    # Daten umformen
    n_samples, n_timesteps, n_features_in = trn_x.shape
    _, _, n_features_out = trn_y.shape
    
    X = trn_x.reshape(n_samples * n_timesteps, n_features_in)   # (390, 2)
    
    # TRAINIEREN ##############################################################
    
    print("Modell wird trainiert.")
    models = []
    for i in range(n_features_out):
        y_i = trn_y[:, :, i].reshape(-1)
        model = LinearRegression()
        model.fit(X, y_i)
        models.append(model)
    print("Modell wurde trainiert.")
    return models


# FUNKTION ZUR BERECHNUNG DES GEWICHTETEN ABSOLUTEN PROZENTUALEN FEHLERS ######
def wape(y_true, y_pred):
    
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))
    
    if denominator == 0:
        return np.nan

    return (numerator/denominator)*100

# FUNKTION ZUR BERECHNUNG DES SYMMETRISCHEN MITTLEREN ABSOLUTEN PROZENTUALEN ##
# FEHLERS #####################################################################
def smape(y_true, y_pred):

    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)    

    n = len(y_true)
    smape_values = []

    for yt, yp in zip(y_true, y_pred):
        denominator = (abs(yt)+abs(yp))/2
        if denominator == 0:
            smape_values.append(0)
        else:
            smape_values.append(abs(yp-yt)/denominator)

    return sum(smape_values)/n*100

# FUNKTION ZUR BERECHNUNG DES MITTLEREN ABSOLUTEN FEHLERS #####################
def mase(y_true, y_pred, m = 1):

    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)      
    
    n = len(y_true)
    
    # Vorhersagefehler (MAE der Prognose)
    mae_forecast = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n

    # MAE des Naive-m-Modells (Baseline)
    if n <= m:
        raise ValueError("Zu wenig Daten für gewählte Saisonalität m.")
        
    naive_errors = [abs(y_true[t] - y_true[t - m]) for t in range(m, n)]
    mae_naive = sum(naive_errors) / len(naive_errors)

    if mae_naive == 0:
        raise ZeroDivisionError("Naive MAE ist 0 – MASE nicht definiert.")

    return mae_forecast/mae_naive


###############################################################################
###############################################################################
# ALLGEMEINE DATEN ############################################################
###############################################################################
###############################################################################

# INFORMATIONEN ZUR MULTIVARIATEN ZEITREIHENANALYSE (MULTIVARIATE TIME ########
# SERIES ANALYSIS (MTS)) ###################################################### 
class MTS:
    
    # Anzahl der Zeitschritte der Eingabedaten
    I_N  = 13
    
    # Anzahl der Zeitschritte der Ausgabedaten
    O_N  = 13
    
    # Zeitschrittweite für die Bildung der finalen Datensätze [min]
    DELT = 3
    
    # Offset für die Bildung der finalen Datensätze [min]
    OFST = 0

# INFORMATIONEN ZU DEN FEIERTAGEN (DIE KEINE SONNTAGE SIND) ###################

HOL = {
        "Österreich": [
            "2022-01-01",   # Neujahrstag (SA)
            "2022-01-06",   # Heilige Drei Könige (DO)
            "2022-04-18",   # Ostermontag (MO)
            "2022-05-26",   # Christi Himmelfahrt (DO)
            "2022-06-06",   # Pfingsmontag (MO)
            "2022-06-16",   # Fronleichnam (DO)
            "2022-08-15",   # Mariä Himmelfahrt (MO)
            "2022-10-26",   # Nationalfeiertag (MI)
            "2022-11-01",   # Allerheiligen (DI)
            "2022-12-08",   # Mariä Empfängnis (DO)
            "2022-12-26",   # Stefanitag (MO)
            "2023-01-06",   # Heilige Drei Könige (FR)
            "2023-04-10",   # Ostermontag (MO)
            "2023-05-01",   # Tag der Arbeit (MO)
            "2023-05-18",   # Christi Himmelfahrt (DO)
            "2023-05-29",   # Pfingsmontag (MO)
            "2023-06-08",   # Fronleichnam (DO)
            "2023-08-15",   # Mariä Himmelfahrt (DI)
            "2023-10-26",   # Nationalfeiertag (DO)
            "2023-11-01",   # Allerheiligen (MI)
            "2023-12-08",   # Mariä Empfängnis (FR)
            "2023-12-25",   # Christtag (MO)
            "2023-12-26",   # Stefanitag (DI)
            "2024-01-01",   # Neujahrstag (MO)
            "2024-01-06",   # Heilige Drei Könige (SA)
            "2024-04-01",   # Ostermontag (MO)
            "2024-05-01",   # Tag der Arbeit (MI)
            "2024-05-09",   # Christi Himmelfahrt (DO)
            "2025-05-20",   # Pfingsmontag (MO)
            "2024-05-30",   # Fronleichnam (DO)
            "2024-08-15",   # Mariä Himmelfahrt (DO)
            "2024-10-26",   # Nationalfeiertag (SA)
            "2024-11-01",   # Allerheiligen (FR)
            "2024-12-25",   # Christtag (MI)
            "2024-12-26",   # Stefanitag (DO)
            "2025-01-01",   # Neujahrstag (MI)
            "2025-01-06",   # Heilige Drei Könige (MO)
            "2025-04-21",   # Ostermontag (MO)
            "2025-05-01",   # Tag der Arbeit (DO)
            "2025-05-29",   # Christi Himmelfahrt (DO)
            "2025-06-09",   # Pfingsmontag (MO)
            "2025-06-19",   # Fronleichnam (DO)
            "2025-08-15",   # Mariä Himmelfahrt (FR)
            "2025-11-01",   # Allerheiligen (SA)
            "2025-12-08",   # Mariä Empfängnis (MO)
            "2025-12-25",   # Christtag (DO)
            "2025-12-26"    # Stefanitag (FR)
        ],
        "Deutschland":  [],
        "Schweiz":      []
        }

HOL = {
    land: [datetime.datetime.strptime(datum, "%Y-%m-%d") for datum in daten]
    for land, daten in HOL.items()
}

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

###############################################################################
###############################################################################
# ZEITINFORMATION #############################################################
###############################################################################
###############################################################################

class T:
    
    # Jahreszeitliche Sinus-/Cosinus-Komponente
    class Y:
        
        # Anwendung
        IMP = False
        
        # Bezug auf lokale Zeit (Winter-/Sommerzeit)
        LT = False
        
        # Datenform
        SPEC = "Zeithorizont"
        
        # Zeithorizont Start [h]
        TH_STRT = -24
        
        # Zeithorizont Ende [h]
        TH_END = 0
        
        # Skalierung
        SCAL = True
        
        # Skalierung max [-]
        SCAL_MAX = 1
        
        # Skalierung min [-]
        SCAL_MIN = 0
        
        # Zeitschrittweite [min]
        DELT = (TH_END-TH_STRT)*60/(MTS.I_N-1)
        
    # Monatliche Sinus-/Cosinus-Komponente
    class M:
        
        # Anwendung
        IMP = False
        
        # Bezug auf lokale Zeit (Winter-/Sommerzeit)
        LT = False
        
        # Datenform
        SPEC = "Zeithorizont"
        
        # Zeithorizont Start [h]
        TH_STRT = -1
        
        # Zeithorizont Ende [h]
        TH_END = 0
        
        # Skalierung
        SCAL = True
        
        # Skalierung max [-]
        SCAL_MAX = 1
        
        # Skalierung min [-]
        SCAL_MIN = 0
        
        # Zeitschrittweite [min]
        DELT = (TH_END-TH_STRT)*60/(MTS.I_N-1)
        
    # Wöchentliche Sinus-/Cosinus-Komponente
    class W:

        # Anwendung
        IMP = False
        
        # Bezug auf lokale Zeit (Winter-/Sommerzeit)
        LT = False
        
        # Datenform
        SPEC = "Aktuelle Zeit"
        
        # Zeithorizont Start [h]
        TH_STRT = -24
        
        # Zeithorizont Ende [h]
        TH_END = 0
        
        # Skalierung
        SCAL = True
        
        # Skalierung max [-]
        SCAL_MAX = 1
        
        # Skalierung min [-]
        SCAL_MIN = 0
        
        # Zeitschrittweite [min]
        DELT = (TH_END-TH_STRT)*60/(MTS.I_N-1)
    
    # Tägliche Sinus-/Cosinus-Komponente
    class D:

        # Anwendung
        IMP = False
        
        # Bezug auf lokale Zeit (Winter-/Sommerzeit)
        LT = True
        
        # Datenform
        SPEC = "Zeithorizont"
        
        # Zeithorizont Start [h]
        TH_STRT = -24
        
        # Zeithorizont Ende [h]
        TH_END = 0
        
        # Skalierung
        SCAL = True
        
        # Skalierung max [-]
        SCAL_MAX = 1
        
        # Skalierung min [-]
        SCAL_MIN = 0
        
        # Zeitschrittweite [min]
        DELT = (TH_END-TH_STRT)*60/(MTS.I_N-1)
    
    # Berücksichtigung von Feiertagen
    class H:

        # Anwendung
        IMP = False
        
        # Bezug auf lokale Zeit (Winter-/Sommerzeit)
        LT = False
        
        # Datenform
        SPEC = "Aktuelle Zeit"
        
        # Zeithorizont Start [h]
        TH_STRT = -100
        
        # Zeithorizont Ende [h]
        TH_END = 0
        
        # Skalierung
        SCAL = True
        
        # Skalierung max [-]
        SCAL_MAX = 1
        
        # Skalierung min [-]
        SCAL_MIN = 0
        
        # Land
        CNTRY   = "Österreich"
        
        # Zeitschrittweite [min]
        DELT = (TH_END-TH_STRT)*60/(MTS.I_N-1)
    
    # Zeitzone
    TZ      = "Europe/Vienna"
        
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
o_dat_inf["spec"] = i_dat_inf["spec"].astype("object")
o_dat_inf["meth"] = i_dat_inf["meth"].astype("object")

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
 
###############################################################################
###############################################################################
# DATENSÄTZE ERSTELLEN ########################################################
###############################################################################
###############################################################################

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
        
    ###########################################################################
    ###########################################################################
    ###########################################################################
    # ZEITINFORMATION #########################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
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

prog_1 = 100
print(f"Erstellung der Datensätze: {prog_1:.2f}%")

i_array_3D = np.array(i_arrays)
o_array_3D = np.array(o_arrays)

# Anzahl der Datensätze
n_dat = i_array_3D.shape[0]

i_combined_array = np.vstack(i_arrays)
o_combined_array = np.vstack(o_arrays)

del i_arrays, o_arrays

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


if any(i_scal_list):
   i_scal_button = True
else:
   i_scal_button = False
   
if any(o_scal_list):
   o_scal_button = True
else:
   o_scal_button = False

###############################################################################
###############################################################################
# VIOLINENPLOT ################################################################
###############################################################################
###############################################################################

# Farbpalette
palette = sns.color_palette("tab20", i_combined_array.shape[1]+o_combined_array.shape[1])

color_plot = []

###############################################################################
# EINGABEDATEN ################################################################
###############################################################################

# LISTE MIT DEN EINEZELNEN PLOTNAMEN ##########################################

i_list = i_dat_inf.index.tolist()

if T.Y.IMP == True:
    i_list.append("Y_sin")
    i_list.append("Y_cos")
if T.M.IMP == True:
    i_list.append("M_sin")
    i_list.append("M_cos")
if T.W.IMP == True:
    i_list.append("W_sin")
    i_list.append("W_cos")
if T.D.IMP == True:
    i_list.append("D_sin")
    i_list.append("D_cos")
if T.H.IMP == True:
    i_list.append("Holiday")

# DICTIONARY FÜR VIOLINENPLOT #################################################
df = pd.DataFrame(i_combined_array)
data = {}
for i, name in enumerate(i_list):
    values = df.iloc[:,i]
    data[name] = values

# Anzahl der Merkmale der Eingabedaten
n_ft_i = i_combined_array.shape[1]

fig, axes = plt.subplots(1,                         # Eine Zeile an Subplots
                         n_ft_i,                    # Anzahl an Subplots nebeneinander
                         figsize = (2*n_ft_i, 6))   # Größe des gesamten Plots

if len(data) <= 1:
    
    for i, (name, values) in enumerate(data.items()):
            
        sns.violinplot(y            = values, 
                       ax           = axes, 
                       color        = palette[i], 
                       inner        = "quartile", 
                       linewidth    = 1.5)
         
        # Titel über jedem Subplot
        axes.set_title(name)
        
        # Entfernen der Achsenbeschriftungen
        axes.set_xlabel("")
        axes.set_ylabel("")
    
else:
    
    # Violinplot in jeden Subplot
    for i, (name, values) in enumerate(data.items()):
            
        sns.violinplot(y            = values, 
                       ax           = axes[i], 
                       color        = palette[i], 
                       inner        = "quartile", 
                       linewidth    = 1.5)
         
        # Titel über jedem Subplot
        axes[i].set_title(name)
        
        # Entfernen der Achsenbeschriftungen
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

plt.suptitle("Datenverteilung \nder Eingabedaten",
             fontsize   = 15,
             fontweight = "bold")

plt.tight_layout()
plt.show()

###############################################################################
# AUSGABEDATEN ################################################################
###############################################################################

# LISTE MIT DEN EINEZELNEN PLOTNAMEN ##########################################

o_list = o_dat_inf.index.tolist()

# DICTIONARY FÜR VIOLINENPLOT #################################################
df = pd.DataFrame(o_combined_array)
data = {}
for i, name in enumerate(o_list):
    values = df.iloc[:,i]
    data[name] = values

# Anzahl der Merkmale der Ausgabedaten
n_ft_o = o_combined_array.shape[1]

fig, axes = plt.subplots(1,                         # Eine Zeile an Subplots
                         n_ft_o,                    # Anzahl an Subplots nebeneinander
                         figsize = (2*n_ft_o, 6))   # Größe des gesamten Plots

if len(data) <= 1:
    
    for i, (name, values) in enumerate(data.items()):
            
        sns.violinplot(y            = values, 
                       ax           = axes, 
                       color        = palette[i+n_ft_i], 
                       inner        = "quartile", 
                       linewidth    = 1.5)
        
        # Titel über jedem Subplot
        axes.set_title(name)
        
        # Entfernen der Achsenbeschriftungen
        axes.set_xlabel("")
        axes.set_ylabel("")
        
else:
    
    # Violinplot in jeden Subplot
    for i, (name, values) in enumerate(data.items()):
            
        sns.violinplot(y            = values, 
                       ax           = axes[i], 
                       color        = palette[i+n_ft_i], 
                       inner        = "quartile", 
                       linewidth    = 1.5)
        
        # Titel über jedem Subplot
        axes[i].set_title(name)
        
        # Entfernen der Achsenbeschriftungen
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

plt.suptitle("Datenverteilung \nder Ausgabedaten",
             fontsize   = 15,
             fontweight = "bold")

plt.tight_layout()
plt.show()

###############################################################################
###############################################################################
# MODELL TRAINIEREN, VALIDIEREN UND TESTEN ####################################
###############################################################################
###############################################################################

###############################################################################
# EINGABEN ####################################################################
###############################################################################

random_dat = False

n_train = round(0.7*n_dat)
n_val   = round(0.2*n_dat)
n_tst = n_dat-n_train-n_val

# INFORMATIONEN ZUM MODELL ####################################################

class MDL:
    
    # Ausgewähltes Verfahren
    #MODE = "Dense"
    #MODE = "CNN"
    #MODE = "LSTM"
    #MODE = "AR LSTM"
    #MODE = "SVR_dir"
    #MODE = "SVR_MIMO"
    MODE = "LIN"
    
    if MODE == "Dense":
    
        # Anzahl an Layer [-]
        LAY = 3
            
        # Anzahl der Neuronen pro Layer [-]
        N = 512
        
        # Max. Anzahl der Trainingsdurchläufe [-]
        EP = 20
        
        # Aktivierungsfunktion
        ACTF = "ReLU"
    
    elif MODE == "CNN":
        
        # Anzahl an Layer [-]
        LAY = 3
            
        # Anzahl der Filter pro Layer [-]
        N = 512
        
        # Kernelgröße [-]
        K = 3
        
        # Max. Anzahl der Trainingsdurchläufe [-]
        EP = 20
        
        # Aktivierungsfunktion
        ACTF = "ReLU"
        
        
    elif MODE == "LSTM":
        
        # Anzahl an Layer [-]
        LAY = 3
            
        # Anzahl der Neuronen pro Layer [-]
        N = 512
        
        # Max. Anzahl der Trainingsdurchläufe [-]
        EP = 20
        
        # Aktivierungsfunktion
        ACTF = "ReLU"
        
    elif MODE == "AR LSTM":
        
        # Anzahl an Layer [-]
        LAY = 3
            
        # Anzahl der Neuronen pro Layer [-]
        N = 512
        
        # Max. Anzahl der Trainingsdurchläufe [-]
        EP = 20
        
        # Aktivierungsfunktion
        ACTF = "ReLU"
        
    elif MODE == "SVR_dir":
        
        # Art der Modellierung von Nichtlinearitäten
        KERNEL = "poly"
        
        # Regulationsparameter (Trade-off Bias vs. Varianz) [-]
        C = 1
        
        # Toleranz für Abweichungen [-]
        EPSILON = 0.1
        
    elif MODE == "SVR_MIMO":
        
        # Art der Modellierung von Nichtlinearitäten
        KERNEL = "poly"
        
        # Regulationsparameter (Trade-off Bias vs. Varianz) [-]
        C = 1
        
        # Toleranz für Abweichungen [-]
        EPSILON = 0.1
        
    elif MODE == "LIN":
        a = 5
        
# Für die Umrechnung von Frontend auf Backend

if MDL.MODE == "Dense" or MDL.MODE == "CNN" or MDL.MODE == "LSTM" \
    or MDL.MODE == "AR LSMT":

    mapping = {
            'ReLU':     'relu',
            'Sigmoid':  'sigmoid',
            'Tanh':     'tanh',
            'Linear':   'linear',
            'Softmax':  'softmax',
            'Keine':    None,
        }
    
    MDL.ACTF = mapping.get(MDL.ACTF, None)
    
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

###############################################################################
# MODELL TRAINIEREN UND VALIDIEREN ############################################
###############################################################################

if MDL.MODE == "Dense":
    mdl = train_dense(trn_x, trn_y, val_x, val_y, MDL)
    
elif MDL.MODE == "CNN":
    mdl = train_cnn(trn_x, trn_y, val_x, val_y, MDL)
    
elif MDL.MODE == "LSTM":
    mdl = train_lstm(trn_x, trn_y, val_x, val_y, MDL)
    
elif MDL.MODE == "AR LSTM":
    mdl = train_ar_lstm(trn_x, trn_y, val_x, val_y, MDL)
    
elif MDL.MODE == "SVR_dir":
    mdl = train_svr_dir(trn_x, trn_y, MDL)      
    
elif MDL.MODE == "SVR_MIMO":
    mdl = train_svr_mimo(trn_x, trn_y, MDL)
    
elif MDL.MODE == "LIN":
    mdl = train_linear_model(trn_x, trn_y)

###############################################################################
# MODELL TESTEN  ##############################################################
###############################################################################

# Initialisierung
tst_fcst = list()

# Forecast für jeden Testdatensatz
for i in range(n_tst):
    
    prog_4 = i/n_tst*100
    print(f"Modell testen: {prog_4:.2f}%")
    
    # Erstellung der Input-Daten für die Prognose
    inp = tst_x[i, : ,:].reshape((1, tst_x.shape[1], n_ft_i))
    
    if MDL.MODE == "SVR_dir" or MDL.MODE == "SVR_MIMO" or MDL.MODE == "LIN":
        
        inp = np.squeeze(inp, axis = 0)
        pred = []
        for i in range(n_ft_o):
            pred.append(mdl[i].predict(inp))
        
        out = np.array(pred).T               
        
        out = np.expand_dims(out, axis = 0)
                
    else:
        
        # Prognose erstellen
        out = mdl.predict(inp, 
                          verbose = 0)
        
    # Man gibt nur die Vektorvorhersage aus
    tst_fcst.append(out[0,:,:])

prog_4 = 100
print(f"Modell testen: {prog_4:.2f}%")

# Prognosen in ein Array umwandeln
tst_fcst = np.array(tst_fcst)

if MDL.MODE == "CNN":
    tst_fcst = np.squeeze(tst_fcst, axis = -1)

tst_fcst_scal = copy.deepcopy(tst_fcst)

###############################################################################
# RE-SCALING ##################################################################
###############################################################################

# Durchlauf der Testdatensätze
for i in range(n_tst):

    prog_5 = i/n_tst*100
    print(f"Re-Scaling: {prog_5:.2f}%")    

    # Durchlauf der Merkmale der Ausgabe-Testdatensätze
    for i1 in range(n_ft_o):
        
        if not o_scalers[i1] is None: 
            
            # Skalierer anwenden
            a = o_scalers[i1].inverse_transform(tst_fcst[i,:,i1].reshape(-1, 1))
            b = o_scalers[i1].inverse_transform(tst_y[i,:,i1].reshape(-1, 1))
            
            # Überschreiben der Spalte
            tst_fcst[i,:,i1] = a.ravel()
            tst_y[i,:,i1] = a.ravel()
            
prog_5 = 100
print(f"Re-Scaling: {prog_5:.2f}%")

###############################################################################
###############################################################################
# PLOT ########################################################################
###############################################################################
###############################################################################

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
      
    # DURCHLAUF DER ZEITINFORMATIONEN #########################################
    
    xx = len(i_dat_inf)
    
    # JAHRESZEITLICHE SINUS-/COSINUS-KOMPONENTE
    
    if T.Y.IMP:
                    
        # ZEITHORIZONT
        if T.Y.SPEC == "Zeithorizont":
        
            utc_th_strt = utc_ref+datetime.timedelta(hours = T.Y.TH_STRT)
            utc_th_end  = utc_ref+datetime.timedelta(hours = T.Y.TH_END)

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
                    
        # AKTUELLE ZEIT
        elif T.Y.SPEC == "Aktuelle Zeit":
            utc_th = []
            for i1 in range(MTS.I_N):
                utc_th.append(utc_ref)
                
        # SINUS
        if y_sbpl_fmt == "original":
            value = tst_x_orig[key_1,:,xx]
        elif y_sbpl_fmt == "skaliert":
            value = tst_x[key_1,:,xx]
            
        # DataFrame erstellen
        df = pd.DataFrame({                
            "UTC":      utc_th,
            "ts":       list(range(-MTS.I_N+1, 1)),
            "value":    value
        })
    
        tst_inf[key_1]["TIME: Y_sin"] = df
        
        # COSINUS
        if y_sbpl_fmt == "original":
            value = tst_x_orig[key_1,:,xx+1]
        elif y_sbpl_fmt == "skaliert":
            value = tst_x[key_1,:,xx+1]
            
        # DataFrame erstellen
        df = pd.DataFrame({
            "UTC":      utc_th,
            "ts":       list(range(-MTS.I_N+1, 1)),
            "value":    value
        })
    
        tst_inf[key_1]["TIME: Y_cos"] = df
 
        xx = xx+2
    
    # MONATLICHE SINUS-/COSINUS-KOMPONENTE
            
    if T.M.IMP:
                    
        # ZEITHORIZONT
        if T.M.SPEC == "Zeithorizont":
        
            utc_th_strt = utc_ref+datetime.timedelta(hours = T.M.TH_STRT)
            utc_th_end  = utc_ref+datetime.timedelta(hours = T.M.TH_END)

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
                    
        # AKTUELLE ZEIT
        elif T.M.SPEC == "Aktuelle Zeit":
            utc_th = []
            for i1 in range(MTS.I_N):
                utc_th.append(utc_ref)
                
        # SINUS
        if y_sbpl_fmt == "original":
            value = tst_x_orig[key_1,:,xx]
        elif y_sbpl_fmt == "skaliert":
            value = tst_x[key_1,:,xx]
            
        # DataFrame erstellen
        df = pd.DataFrame({
            "UTC":      utc_th,
            "ts":       list(range(-MTS.I_N+1, 1)),
            "value":    value
        })
    
        tst_inf[key_1]["TIME: M_sin"] = df
        
        # COSINUS
        if y_sbpl_fmt == "original":
            value = tst_x_orig[key_1,:,xx+1]
        elif y_sbpl_fmt == "skaliert":
            value = tst_x[key_1,:,xx+1]
            
        # DataFrame erstellen
        df = pd.DataFrame({
            "UTC":      utc_th,
            "ts":       list(range(-MTS.I_N+1, 1)),
            "value":    value
        })
    
        tst_inf[key_1]["TIME: M_cos"] = df
 
        xx = xx+2
        
    # WÖCHENTLICHE SINUS-/COSINUS-KOMPONENTE
            
    if T.W.IMP:
                    
        # ZEITHORIZONT
        if T.W.SPEC == "Zeithorizont":
        
            utc_th_strt = utc_ref+datetime.timedelta(hours = T.W.TH_STRT)
            utc_th_end  = utc_ref+datetime.timedelta(hours = T.W.TH_END)

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
                    
        # AKTUELLE ZEIT
        elif T.W.SPEC == "Aktuelle Zeit":
            utc_th = []
            for i1 in range(MTS.I_N):
                utc_th.append(utc_ref)
                
        # SINUS
        if y_sbpl_fmt == "original":
            value = tst_x_orig[key_1,:,xx]
        elif y_sbpl_fmt == "skaliert":
            value = tst_x[key_1,:,xx]
            
        # DataFrame erstellen
        df = pd.DataFrame({
            "UTC":      utc_th,
            "ts":       list(range(-MTS.I_N+1, 1)),
            "value":    value
        })
    
        tst_inf[key_1]["TIME: W_sin"] = df
        
        # COSINUS
        if y_sbpl_fmt == "original":
            value = tst_x_orig[key_1,:,xx+1]
        elif y_sbpl_fmt == "skaliert":
            value = tst_x[key_1,:,xx+1]
            
        # DataFrame erstellen
        df = pd.DataFrame({
            "UTC":      utc_th,
            "ts":       list(range(-MTS.I_N+1, 1)),
            "value":    value
        })
    
        tst_inf[key_1]["TIME: W_cos"] = df
 
        xx = xx+2
        
    # TÄGLICHE SINUS-/COSINUS-KOMPONENTE
            
    if T.D.IMP:
                    
        # ZEITHORIZONT
        if T.D.SPEC == "Zeithorizont":
        
            utc_th_strt = utc_ref+datetime.timedelta(hours = T.D.TH_STRT)
            utc_th_end  = utc_ref+datetime.timedelta(hours = T.D.TH_END)

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
                    
        # AKTUELLE ZEIT
        elif T.D.SPEC == "Aktuelle Zeit":
            utc_th = []
            for i1 in range(MTS.I_N):
                utc_th.append(utc_ref)
                
        # SINUS
        if y_sbpl_fmt == "original":
            value = tst_x_orig[key_1,:,xx]
        elif y_sbpl_fmt == "skaliert":
            value = tst_x[key_1,:,xx]
            
        # DataFrame erstellen
        df = pd.DataFrame({
            "UTC":      utc_th,
            "ts":       list(range(-MTS.I_N+1, 1)),
            "value":    value
        })
    
        tst_inf[key_1]["TIME: D_sin"] = df
        
        # COSINUS
        if y_sbpl_fmt == "original":
            value = tst_x_orig[key_1,:,xx+1]
        elif y_sbpl_fmt == "skaliert":
            value = tst_x[key_1,:,xx+1]
            
        # DataFrame erstellen
        df = pd.DataFrame({
            "UTC":      utc_th,
            "ts":       list(range(-MTS.I_N+1, 1)),
            "value":    value
        })
    
        tst_inf[key_1]["TIME: D_cos"] = df
 
        xx = xx+2
        
    # FEIERTAGE
            
    if T.H.IMP:
                    
        # ZEITHORIZONT
        if T.H.SPEC == "Zeithorizont":
        
            utc_th_strt = utc_ref+datetime.timedelta(hours = T.H.TH_STRT)
            utc_th_end  = utc_ref+datetime.timedelta(hours = T.H.TH_END)

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
                    
        # AKTUELLE ZEIT
        elif T.H.SPEC == "Aktuelle Zeit":
            utc_th = []
            for i1 in range(MTS.I_N):
                utc_th.append(utc_ref)
                
        # Information
        if y_sbpl_fmt == "original":
            value = tst_x_orig[key_1,:,xx]
        elif y_sbpl_fmt == "skaliert":
            value = tst_x[key_1,:,xx]
            
        # DataFrame erstellen
        df = pd.DataFrame({
            "UTC":      utc_th,
            "ts":       list(range(-MTS.I_N+1, 1)),
            "value":    value
        })
    
        tst_inf[key_1]["TIME: Holiday"] = df

        xx = xx+1
            
    # DURCHLAUF DER PROGNOSEDATEN #############################################
    
    for i in range(len(o_dat_inf)):
                    
        # ZEITGRENZEN DER TRANSFERIERUNG ##################################
            
        utc_th_strt = utc_ref+datetime.timedelta(hours = o_dat_inf.iloc[i]["th_strt"])
        utc_th_end  = utc_ref+datetime.timedelta(hours = o_dat_inf.iloc[i]["th_end"])
            
        # ZEITSTEMPEL DER TRANSFERIERUNG ##################################
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
            value = tst_fcst[key_1,:,i]
        elif y_sbpl_fmt == "skaliert":
            value = tst_fcst_scal[key_1,:,i]
            
        # DataFrame erstellen
        df = pd.DataFrame({
            "UTC":      utc_th,
            "ts":       list(range(0, MTS.O_N)),
            "value":    value
        })
        
        tst_inf[key_1]["FCST: "+o_dat_inf.index[i]] = df
                    
###############################################################################
# PLOT FÜLLEN #################################################################
###############################################################################    

# Schleife über alle Subplots
for i_sbpl in range (num_sbpl):
    
    # Zeile des aktuellen Subplots
    i_y_sbpl = math.floor(i_sbpl/num_sbpl_x)
    
    # Spalte des aktuellen Subplots
    i_x_sbpl = i_sbpl-i_y_sbpl*num_sbpl_x
    
    # Gesuchter Key des übergeordneten Dictionaries
    key_1 = list(tst_inf.keys())[i_sbpl]
    
    # Hauptachsen des Subplots definieren
    ax_sbpl_orig = axs[i_y_sbpl, i_x_sbpl]
    ax_sbpl = [ax_sbpl_orig]

    # Zähler
    i_line = 0
    i_ax_l = 0
    i_ax_r = 0

    # PLOTTEN DER AUSGEWÄHLTEN EINGABEDATEN UND ZEITINFORMATION ###############
    for i_feat in range(len(df_plot_in)):
        
        # Eingabedaten werden angezeigt
        if df_plot_in.iloc[i_feat, 0]:
            
            # Aktuelle Farbe
            color_plt = palette[i_feat]
            
            # Gesuchter Key des untergeordneten Dictionaries
            if i_feat < len(i_dat_inf):
                key_2 = "IN: "+df_plot_in.index[i_feat]
            else:
                key_2 = "TIME: "+df_plot_in.index[i_feat]
            
            df = tst_inf[key_1][key_2]

            if x_sbpl == "UTC":
                x_value = tst_inf[key_1][key_2]["UTC"]
            else:
                x_value = tst_inf[key_1][key_2]["ts"]
            
            y_value = tst_inf[key_1][key_2]["value"]
                                    
            if y_sbpl_set == "gemeinsame Achse":
            
                # Plotten
                ax_sbpl_orig.plot(x_value, 
                                  y_value,
                                  label      = key_2 if i_sbpl == 0 else None,
                                  color      = color_plt,
                                  marker     = 'o',
                                  linewidth  = 1,
                                  markersize = 2)    
             
            elif y_sbpl_set == "separate Achsen":
                
                # Weitere Linie im Subplot → Neue y-Achse erzeugen
                if i_line > 0: ax_sbpl.append(ax_sbpl_orig.twinx())
                    
                # Plotten
                ax_sbpl[-1].plot(x_value, 
                                 y_value,
                                 label      = key_2 if i_sbpl == 0 else None,
                                 color      = color_plt,
                                 marker     = 'o',
                                 linewidth  = 1,
                                 markersize = 2)    
                
                # Separate y-Achse kommt auf die linke Seite
                if i_line < n_ax_l:
                    
                    pos = 'left'
                    
                    # Zähler inkrementieren
                    i_ax = i_ax_l
                    i_ax_l += 1
                    
                # Separate y-Achse kommt auf die rechte Seite
                else: 
                    
                    pos = 'right'
                    
                    # Zähler inkrementieren
                    i_ax = i_ax_r
                    i_ax_r += 1
                
                # Verschieben der y-Achsenlinie nach außen
                ax_sbpl[-1].spines[pos].set_position(('outward', i_ax*30))

                # Ausblenden der oberen und gegenüberliegenden Achsenlinie
                for spine in ['top', 'left' if pos == 'right' else 'right']:
                    ax_sbpl[-1].spines[spine].set_visible(False)

                # Position der y-Achsenticks
                ax_sbpl[-1].yaxis.set_ticks_position(pos)
                    
                # Farbe der Achsenlinie der y-Achse
                ax_sbpl[-1].spines[pos].set_color(color_plt)

                # Konfiguration der y-Achsenticks und ihrer Beschriftung
                ax_sbpl[-1].tick_params(axis        = 'y',
                                        direction   = 'inout',
                                        colors      = color_plt,
                                        labelcolor  = color_plt,
                                        labelsize   = 8)
                    
                # Konfiguration der x-Achsenticks und ihrer Beschriftung
                ax_sbpl[0].tick_params(axis        = "x",
                                       labelsize   = 8)
                    
                # Zähler inkrementieren
                i_line += 1
                                        
    # PLOTTEN DER AUSGEWÄHLTEN AUSGABEDATEN ###################################
    for i_feat in range(len(df_plot_out)):
        
        # Eingabedaten werden angezeigt
        if df_plot_out.iloc[i_feat, 0]:
            
            # Aktuelle Farbe
            color_plt = palette[i_combined_array.shape[1]+i_feat]
            
            # Gesuchter Key des untergeordneten Dictionaries
            key_2 = "OUT: "+df_plot_out.index[i_feat]
            
            df = tst_inf[key_1][key_2]

            if x_sbpl == "UTC":
                x_value = tst_inf[key_1][key_2]["UTC"]
            else:
                x_value = tst_inf[key_1][key_2]["ts"]
            
            y_value = tst_inf[key_1][key_2]["value"]
             
            if y_sbpl_set == "gemeinsame Achse":
            
                # Plotten
                ax_sbpl_orig.plot(x_value, 
                                  y_value,
                                  label      = key_2 if i_sbpl == 0 else None,
                                  color      = color_plt,
                                  marker     = 'o',
                                  linewidth  = 1,
                                  markersize = 2)
            
            elif y_sbpl_set == "separate Achsen":
                
                # Weitere Linie im Subplot → Neue y-Achse erzeugen
                if i_line > 0: ax_sbpl.append(ax_sbpl_orig.twinx())
                    
                # Plotten
                ax_sbpl[-1].plot(x_value, 
                                 y_value,
                                 label      = key_2 if i_sbpl == 0 else None,
                                 color      = color_plt,
                                 marker     = 'o',
                                 linewidth  = 1,
                                 markersize = 2)    
                
                # Separate y-Achse kommt auf die linke Seite
                if i_line < n_ax_l:
                    
                    pos = 'left'
                    
                    # Zähler inkrementieren
                    i_ax = i_ax_l
                    i_ax_l += 1
                    
                # Separate y-Achse kommt auf die rechte Seite
                else: 
                    
                    pos = 'right'
                    
                    # Zähler inkrementieren
                    i_ax = i_ax_r
                    i_ax_r += 1
                
                # Verschieben der y-Achsenlinie nach außen
                ax_sbpl[-1].spines[pos].set_position(('outward', i_ax*30))

                # Ausblenden der oberen und gegenüberliegenden Achsenlinie
                for spine in ['top', 'left' if pos == 'right' else 'right']:
                    ax_sbpl[-1].spines[spine].set_visible(False)

                    # Position der y-Achsenticks
                    ax_sbpl[-1].yaxis.set_ticks_position(pos)
                    
                    # Farbe der Achsenlinie der y-Achse
                    ax_sbpl[-1].spines[pos].set_color(color_plt)

                    # Konfiguration der y-Achsenticks und ihrer Beschriftung
                    ax_sbpl[-1].tick_params(axis        = 'y',
                                            direction   = 'inout',
                                            colors      = color_plt,
                                            labelcolor  = color_plt,
                                            labelsize   = 8)
                    
                    # Konfiguration der x-Achsenticks und ihrer Beschriftung
                    ax_sbpl[0].tick_params(axis        = "x",
                                           labelsize   = 8)
                    
                    # Zähler inkrementieren
                    i_line += 1
                                        
    # PLOTTEN DER AUSGEWÄHLTEN PROGNOSEDATEN ##################################
        
    for i_feat in range(len(df_plot_out)):
        
        # Prognosedaten werden angezeigt
        if df_plot_fcst.iloc[i_feat, 0]:
            
            # Aktuelle Farbe
            color_plt = palette[i_combined_array.shape[1]+i_feat]
            
            # Gesuchter Key des untergeordneten Dictionaries
            key_2 = "FCST: "+df_plot_out.index[i_feat]
            
            df = tst_inf[key_1][key_2]

            if x_sbpl == "UTC":
                x_value = tst_inf[key_1][key_2]["UTC"]
            else:
                x_value = tst_inf[key_1][key_2]["ts"]
            
            y_value = tst_inf[key_1][key_2]["value"]
            
            if y_sbpl_set == "gemeinsame Achse":
            
                # Plotten
                ax_sbpl_orig.plot(x_value, 
                                  y_value,
                                  label      = key_2 if i_sbpl == 0 else None,
                                  color      = color_plt,
                                  marker     = 'x',
                                  linestyle  = '--',
                                  linewidth  = 1,
                                  markersize = 4)
                                    
            elif y_sbpl_set == "separate Achsen":
                
                # Weitere Linie im Subplot → Neue y-Achse erzeugen
                if df_plot_out.iloc[i_feat, 0] == False:
                    ax_sbpl.append(ax_sbpl_orig.twinx())
                    i_pos = len(ax_sbpl)-1
                else:
                    i_pos = df_plot_out.iloc[:(i_feat+1)]['plot'].sum()-1+df_plot_in.iloc[:]["plot"].sum()
                    
                    
                # Plotten
                ax_sbpl[i_pos].plot(x_value, 
                                    y_value,
                                    label      = key_2 if i_sbpl == 0 else None,
                                    color      = color_plt,
                                    marker     = 'x',
                                    linestyle  = '--',
                                    linewidth  = 1,
                                    markersize = 4)    
                
                if df_plot_out.iloc[i_feat, 0] == False:
                    
                    # Separate y-Achse kommt auf die linke Seite
                    if i_line < n_ax_l:
                        
                        pos = 'left'
                        
                        # Zähler inkrementieren
                        i_ax = i_ax_l
                        i_ax_l += 1
                        
                    # Separate y-Achse kommt auf die rechte Seite
                    else: 
                        
                        pos = 'right'
                        
                        # Zähler inkrementieren
                        i_ax = i_ax_r
                        i_ax_r += 1
                    
                    # Verschieben der y-Achsenlinie nach außen
                    ax_sbpl[-1].spines[pos].set_position(('outward', i_ax*30))

                    # Ausblenden der oberen und gegenüberliegenden Achsenlinie
                    for spine in ['top', 'left' if pos == 'right' else 'right']:
                        ax_sbpl[-1].spines[spine].set_visible(False)

                        # Position der y-Achsenticks
                        ax_sbpl[-1].yaxis.set_ticks_position(pos)
                        
                        # Farbe der Achsenlinie der y-Achse
                        ax_sbpl[-1].spines[pos].set_color(color_plt)

                        # Konfiguration der y-Achsenticks und ihrer Beschriftung
                        ax_sbpl[-1].tick_params(axis        = 'y',
                                                direction   = 'inout',
                                                colors      = color_plt,
                                                labelcolor  = color_plt,
                                                labelsize   = 8)
                        
                        # Konfiguration der x-Achsenticks und ihrer Beschriftung
                        ax_sbpl[0].tick_params(axis        = "x",
                                               labelsize   = 8)
                        
                        # Zähler inkrementieren
                        i_line += 1
                  
    # PLOTTEN EINER SENKRECHTEN LINIE #########################################
    if x_sbpl == "UTC":
        ax_sbpl_orig.axvline(x          = tst_inf[key_1]["utc_ref"], 
                             color      = "black", 
                             linestyle  = '--',
                             label = None)
    else:
        ax_sbpl_orig.axvline(x          = 0, 
                             color      = "black", 
                             linestyle  = '--',
                             label = None)
             
    ax_sbpl_orig.set_title("UTC: "+tst_inf[key_1]["utc_ref"].strftime("%Y-%m-%d %H:%M:%S"))
    

    # Nur für den ersten Subplot werden die Linien und Achsen zusammengefasst
    if i_sbpl == 0:
        # Alle Linien-Handles und Labels sammeln
        lines = []
        labels = []
        
        for ax in fig.axes:
            for line in ax.get_lines():
                lines.append(line)
                labels.append(line.get_label())
        
        try:
            b = labels.index("_child1")
            del lines[b], labels[b]
        except:
            pass
    
# LEGENDE  ####################################################################

# Gemeinsame Legende
fig.legend(lines, 
           labels, 
           loc      = "upper right", 
           ncol     = 5,
           fontsize = 8)

# TITEL #######################################################################

plt.suptitle("Auswertung der Testdatensätze",
             fontsize   = 20, 
             fontweight = 'bold')

plt.show()

###############################################################################
###############################################################################
# EVALUIERUNG #################################################################
###############################################################################
###############################################################################

# Anzahl der Merkmale
num_feat = tst_y_orig.shape[2]

###############################################################################
# MITTELWERTBILDUNG ###########################################################
###############################################################################

dat_eval = {}
n_max = 12

y_all       = np.full((n_max, n_tst, MTS.O_N, num_feat), np.nan)
fcst_all    = np.full((n_max, n_tst, MTS.O_N, num_feat), np.nan)

# Schleife über alle Mittelungen
for n_avg in range(1, n_max+1):

    # Anzahl der Zeitschritte der gemittelten Arrays
    n_ts = math.floor(MTS.O_N/n_avg)
    
    # Array vorbereiten
    y       = np.zeros((n_tst, n_ts, num_feat))
    fcst    = np.zeros((n_tst, n_ts, num_feat))
    dat_eval_int = {}
    
    # Schleife über jeden Testdatensatz
    for i in range(n_tst):
        
        # Schleife über jedes Merkmal
        for j in range(num_feat):
            
            # Schleife über jeden Zeitschritt
            for k in range(n_ts):
                strt = k * n_avg
                end = min(strt + n_avg, MTS.O_N)
                y[i, k, j] = np.mean(tst_y_orig[i, strt:end, j])
                fcst[i, k, j] = np.mean(tst_fcst[i, strt:end, j])
                
                y_all[n_avg-1, i, k, j]     = np.mean(tst_y_orig[i, strt:end, j])
                fcst_all[n_avg-1, i, k, j]  = np.mean(tst_fcst[i, strt:end, j])
    
    dat_eval_int["y"] = y
    dat_eval_int["fcst"] = fcst
    dat_eval_int["delt"] = np.array(o_dat_inf["delt_transf"]*n_avg)
    dat_eval[n_avg] = dat_eval_int

###############################################################################
# FEHLERBERECHNUNG ############################################################
###############################################################################

# GESAMT ######################################################################

# Schleife über alle Mittelungen
for i in range(n_max):

    # Initialisierung
    mae_int, mape_int, mse_int, rmse_int, nrmse_int, wape_int, \
        smape_int, mase_int = ([] for _ in range(8))  
        
    # Durchlauf aller Merkmale
    for i_feat in range(num_feat):
        
        v_true = y_all[i,:,:,i_feat]
        v_fcst = fcst_all[i,:,:,i_feat]
        
        mask = ~np.isnan(v_true) & ~np.isnan(v_fcst)
        mask_1 = ~np.isnan(v_true)
        
        try:
            mae_int.append(mae(v_true[mask], v_fcst[mask]))
            mape_int.append(100*mape(v_true[mask], v_fcst[mask]))
            mse_int.append(mse(v_true[mask], v_fcst[mask]))
            rmse_int.append(rmse(v_true[mask], v_fcst[mask]))
            nrmse_int.append(rmse(v_true[mask], v_fcst[mask])/np.mean(v_true[mask_1]))
            wape_int.append(wape(v_true[mask], v_fcst[mask]))
            smape_int.append(smape(v_true[mask], v_fcst[mask]))
            mase_int.append(mase(v_true[mask], v_fcst[mask]))
        except:

            pass
    
    # Mittlerer absoluter Fehler (Mean Absolute Error, MAE)
    dat_eval[i+1]["MAE"]    = np.array(mae_int)
    
    # Mittlerer absoluter prozentualer Fehler (Mean Absolute Percentage Error, MAPE)
    dat_eval[i+1]["MAPE"]   = np.array(mape_int)
    
    # Mittlerer quatratischer Fehler (Mean Squared Error, MSE)
    dat_eval[i+1]["MSE"]    = np.array(mse_int)
    
    # Wurzel des mittleren quatratischen Fehlers (Root Mean Squared Error, RMSE)
    dat_eval[i+1]["RMSE"]    = np.array(rmse_int)
    
    # Wurzel des mittleren quatratischen Fehlers, normalisiert (Normalized Root Mean Squared Error, NRMSE)
    dat_eval[i+1]["NRMSE"]    = np.array(nrmse_int)
    
    # Gewichteter absoluter prozentualer Fehler (Weighted Average Percentage Error, WAPE)
    dat_eval[i+1]["WAPE"]    = np.array(wape_int)
    
    # Symmetrischer mittlerer absoluter prozentualer Fehler (Symmetric Mean Absolute Percentage Error, sMAPE)
    dat_eval[i+1]["sMAPE"]    = np.array(smape_int)
    
    # Skalierter mittlerer absoluter Fehler (Mean Absolute Scaled Error, MASE)
    dat_eval[i+1]["MASE"]    = np.array(mase_int)

# ZEITSCHRITTE ################################################################

# Schleife über alle Mittelungen
for i in range(n_max):

    # Initialisierung
    mae_ts, mape_ts, mse_ts, rmse_ts, nrmse_ts, wape_ts, \
        smape_ts, mase_ts = ([] for _ in range(8))
    
    # Durchlauf aller Merkmale
    for i_feat in range(num_feat):
        
        # Initialisierung
        mae_int, mape_int, mse_int, rmse_int, nrmse_int, wape_int, \
            smape_int, mase_int = ([] for _ in range(8))
        
        # Durchlauf aller Zeitschritte
        for i_ts in range(dat_eval[i+1]["y"].shape[1]):
    
            v_true = y_all[i,:,i_ts,i_feat]
            v_fcst = fcst_all[i,:,i_ts,i_feat]
                        
            mae_int.append(mae(v_true, v_fcst))
            mape_int.append(100*mape(v_true, v_fcst))
            mse_int.append(mse(v_true, v_fcst))
            rmse_int.append(rmse(v_true, v_fcst))
            nrmse_int.append(rmse(v_true, v_fcst)/np.mean(v_true))
            wape_int.append(wape(v_true, v_fcst))
            smape_int.append(smape(v_true, v_fcst))
            mase_int.append(mase(v_true, v_fcst))
                 
        mae_ts.append(mae_int)
        mape_ts.append(mape_int)
        mse_ts.append(mse_int)
        rmse_ts.append(rmse_int)
        nrmse_ts.append(nrmse_int)
        wape_ts.append(wape_int)
        smape_ts.append(smape_int)
        mase_ts.append(mase_int)
        
    dat_eval[i+1]["MAE_TS"]     = np.array(mae_ts)
    dat_eval[i+1]["MAPE_TS"]    = np.array(mape_ts)
    dat_eval[i+1]["MSE_TS"]     = np.array(mse_ts)
    dat_eval[i+1]["RMSE_TS"]    = np.array(rmse_ts)
    dat_eval[i+1]["NRMSE_TS"]   = np.array(nrmse_ts)
    dat_eval[i+1]["WAPE_TS"]    = np.array(wape_ts)
    dat_eval[i+1]["sMAPE_TS"]   = np.array(smape_ts)
    dat_eval[i+1]["MASE_TS"]    = np.array(mase_ts)

df_eval = {}

for i_feat in range(n_ft_o):

    # Initialisierung
    delt_int, mae_int, mape_int, mse_int, rmse_int, nrmse_int, wape_int, \
        smape_int, mase_int = ([] for _ in range(9))
        
    # Initialisierung
    mae_ts, mape_ts, mse_ts, rmse_ts, nrmse_ts, wape_ts, \
        smape_ts, mase_ts = ([] for _ in range(8))
    
    for i in range(n_max):
        delt_int.append(float(dat_eval[i+1]["delt"][i_feat]))
        mae_int.append(float(dat_eval[i+1]["MAE"][i_feat]))
        mape_int.append(float(dat_eval[i+1]["MAPE"][i_feat]))
        mse_int.append(float(dat_eval[i+1]["MSE"][i_feat]))
        rmse_int.append(float(dat_eval[i+1]["RMSE"][i_feat]))
        nrmse_int.append(float(dat_eval[i+1]["NRMSE"][i_feat]))
        wape_int.append(float(dat_eval[i+1]["WAPE"][i_feat]))
        smape_int.append(float(dat_eval[i+1]["sMAPE"][i_feat]))
        mase_int.append(float(dat_eval[i+1]["MASE"][i_feat]))
                
    df_eval_int = pd.DataFrame({
    "delta [min]":  delt_int,
    "MAE":          mae_int,
    "MAPE":         mape_int,
    "MSE":          mse_int,
    "RMSE":         rmse_int,
    "NRMSE":        nrmse_int,
    "WAPE":         wape_int,
    "sMAPE":        smape_int,
    "MASE":         mase_int
        })
    
    df_eval[o_dat_inf.index[i_feat]] = df_eval_int


df_eval_ts = {}

for i_feat in range(n_ft_o):

    df_eval_ts[o_dat_inf.index[i_feat]] = {}

    # Initialisierung
    delt_int, mae_int, mape_int, mse_int, rmse_int, nrmse_int, wape_int, \
        smape_int, mase_int = ([] for _ in range(9))
        
    # Initialisierung
    mae_ts, mape_ts, mse_ts, rmse_ts, nrmse_ts, wape_ts, \
        smape_ts, mase_ts = ([] for _ in range(8))
    
    df_eval_ts[o_dat_inf.index[i_feat]] = {}
    
    for i in range(n_max):   
    
        df_eval_ts_int = pd.DataFrame({
        'MAE':      dat_eval[i+1]["MAE_TS"][i_feat],
        'MAPE':     dat_eval[i+1]["MAPE_TS"][i_feat],
        'MSE':      dat_eval[i+1]["MSE_TS"][i_feat],
        'RMSE':     dat_eval[i+1]["RMSE_TS"][i_feat],
        'NRMSE':    dat_eval[i+1]["NRMSE_TS"][i_feat],
        'WAPE':     dat_eval[i+1]["WAPE_TS"][i_feat],
        'sMAPE':    dat_eval[i+1]["sMAPE_TS"][i_feat],
        'MASE':     dat_eval[i+1]["MASE_TS"][i_feat]
        })

        df_eval_ts[o_dat_inf.index[i_feat]][float(dat_eval[i+1]["delt"][i_feat])] = df_eval_ts_int