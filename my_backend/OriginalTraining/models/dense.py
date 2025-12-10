"""
Dense Model - Fully Connected neuronska mreža
Ekstrahirano iz training_original.py linije 169-237

Funkcija train_dense() trenira i validira Dense-Layer model
koristeći TensorFlow/Keras.
"""

import tensorflow as tf

###############################################################################
# FUNKTION ZUM TRAINIEREN UND VALIDIEREN EINES DENSE-LAYER-MODELLS ############
###############################################################################

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
