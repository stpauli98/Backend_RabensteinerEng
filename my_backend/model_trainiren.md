# Model Training Functions - Extracted from training_backend_test_2.py

This document contains all 7 model training functions extracted from the reference implementation.

## 1. Dense Neural Network (train_dense)

```python
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
```

## 2. Convolutional Neural Network (train_cnn)

```python
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
```

## 3. LSTM Neural Network (train_lstm)

```python
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
```

## 4. Autoregressive LSTM (train_ar_lstm)

```python
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
```

## 5. Support Vector Regression Direct (train_svr_dir)

```python
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
```

## 6. Support Vector Regression MIMO (train_svr_mimo)

```python
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
```

## 7. Linear Regression (train_linear_model)

```python
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
```

## Required Imports

```python
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
```

## MDL Class Parameters

The MDL class should contain the following parameters based on model type:

### For Neural Networks (Dense, CNN, LSTM, AR LSTM):
- `MDL.LAY`: Number of layers
- `MDL.N`: Number of neurons/filters per layer
- `MDL.EP`: Number of epochs
- `MDL.ACTF`: Activation function
- `MDL.K`: Kernel size (CNN only)

### For SVR Models (SVR_dir, SVR_MIMO):
- `MDL.KERNEL`: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
- `MDL.C`: Regularization parameter
- `MDL.EPSILON`: Epsilon value

### For Linear Model:
No specific MDL parameters required (uses default LinearRegression)

## Notes

1. **CNN Note**: The reference implementation has a bug on line 262 where it uses `trn_x` instead of `train_x`. This should be corrected in production code.

2. **CNN Epochs**: The CNN function forces epochs to 20 (line 307) regardless of MDL.EP value.

3. **SVR_dir**: This model loops through features to create separate models, which might be incorrect - it should loop through output features, not input features.

4. **Data Shape**: All models expect 3D input data with shape (samples, timesteps, features).

5. **Early Stopping**: Neural network models use early stopping with patience=2 to prevent overfitting.

6. **StandardScaler**: SVR models use StandardScaler in a pipeline to normalize data before training.