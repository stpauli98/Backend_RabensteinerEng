"""
Model Training - Treniranje modela
Ekstrahirano iz training_original.py linije 2236-2260

Sadrži:
- Pozivanje odgovarajuće train funkcije ovisno o MDL.MODE
- Podržava: Dense, CNN, LSTM, AR LSTM, SVR_dir, SVR_MIMO, LIN
"""

from models import (
    train_dense,
    train_cnn,
    train_lstm,
    train_ar_lstm,
    train_svr_dir,
    train_svr_mimo,
    train_linear_model
)
from config.model import MDL

###############################################################################
# MODELL TRAINIEREN UND VALIDIEREN ############################################
###############################################################################

# Napomena: Ovaj kod zahtijeva prethodno definirane varijable:
# trn_x, trn_y, val_x, val_y

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
