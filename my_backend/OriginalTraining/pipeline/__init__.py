"""
Pipeline Module - Dataset kreiranje, vremenske značajke i skaliranje
Ekstrahirano iz training_original.py linije 1049-1875

Ovaj modul sadrži:
- dataset_creation.py: Glavna petlja za kreiranje dataseta iz ulaznih/izlaznih podataka
- time_features_processing.py: Obrada vremenskih komponenti (Y, M, W, D, H)
- scaling.py: Skaliranje podataka i kreiranje MinMaxScaler objekata
"""

# Komponente se izvršavaju sekvencijalno u main.py orchestratoru
# jer ovise jedna o drugoj
