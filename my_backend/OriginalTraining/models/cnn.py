"""
CNN Model - Konvolucijska neuronska mreža
Ekstrahirano iz training_original.py linije 239-319

Funkcija train_cnn() trenira i validira CNN model
koristeći TensorFlow/Keras Conv2D slojeve.
"""

import tensorflow as tf

###############################################################################
# FUNKTION ZUM TRAINIEREN UND VALIDIEREN EINES CNN-MODELLS ####################
###############################################################################

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
    # - Kanäle: Anzahl der Kanäle pro Pixel (z. B. 3 für RGB-Bilder, 1 für Graustufen)
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
