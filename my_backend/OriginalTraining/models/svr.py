"""
SVR Models - Support Vector Regression
Ekstrahirano iz training_original.py linije 458-529

Sadrži funkcije:
- train_svr_dir: SVR direktna implementacija
- train_svr_mimo: SVR Multi-Input Multi-Output
"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

###############################################################################
# FUNKTION ZUM TRAINIEREN EINES SVR-MODELLS (DIREKTNA) ########################
###############################################################################

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

###############################################################################
# FUNKTION ZUM TRAINIEREN EINES SVR-MIMO-MODELLS ##############################
###############################################################################

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
