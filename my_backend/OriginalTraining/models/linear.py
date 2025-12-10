"""
Linear Model - Linearna regresija
Ekstrahirano iz training_original.py linije 531-551

Funkcija train_linear_model() trenira linearni model
koristeći sklearn LinearRegression.
"""

from sklearn.linear_model import LinearRegression

###############################################################################
# FUNKTION ZUM TRAINIEREN EINES LINEAREN MODELLS ##############################
###############################################################################

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
