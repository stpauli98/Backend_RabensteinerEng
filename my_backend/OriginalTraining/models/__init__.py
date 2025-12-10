"""
Models Module - Funkcije za treniranje modela
Ekstrahirano iz training_original.py linije 169-551

Ovaj modul sadrži funkcije za treniranje različitih modela:
- train_dense: Dense (Fully Connected) neuronska mreža
- train_cnn: Konvolucijska neuronska mreža
- train_lstm: LSTM mreža
- train_ar_lstm: Autoregressive LSTM
- train_svr_dir: Support Vector Regression (direktna)
- train_svr_mimo: Support Vector Regression (Multi-Input Multi-Output)
- train_linear_model: Linearna regresija
"""

from .dense import train_dense
from .cnn import train_cnn
from .lstm import train_lstm, train_ar_lstm
from .svr import train_svr_dir, train_svr_mimo
from .linear import train_linear_model

__all__ = [
    'train_dense',
    'train_cnn',
    'train_lstm',
    'train_ar_lstm',
    'train_svr_dir',
    'train_svr_mimo',
    'train_linear_model'
]
