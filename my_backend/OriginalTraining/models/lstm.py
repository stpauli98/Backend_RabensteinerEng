"""
LSTM Models - Long Short-Term Memory mreže
Ekstrahirano iz training_original.py linije 321-456

Sadrži funkcije:
- train_lstm: Standardni LSTM model
- train_ar_lstm: Autoregressive LSTM model
"""

import tensorflow as tf

###############################################################################
# FUNKTION ZUM TRAINIEREN UND VALIDIEREN EINES LSTM-MODELLS ###################
###############################################################################

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

###############################################################################
# FUNKTION ZUM TRAINIEREN UND VALIDIEREN EINES AUTOREGRESSIVEN LSTM-MODELLS ###
###############################################################################

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
