"""
Main Orchestrator - Glavni orchestrator za training_original.py
Ovaj fajl prikazuje sekvencijalni tok izvršavanja originalnog koda.

NAPOMENA: Ovo je demonstracijski fajl koji pokazuje kako su komponente
organizirane. Originalni kod se izvršavao kao jedna monolitna skripta
gdje su sve varijable bile dijeljene u globalnom scope-u.

Originalni fajl: training_original.py (~3500 linija)
"""

###############################################################################
# STRUKTURA ORIGINALNOG KODA ##################################################
###############################################################################

"""
1. IMPORTI I POMOĆNE FUNKCIJE (Linije 1-610)
   - utils/data_loader.py: load() funkcija za učitavanje podataka
   - utils/time_utils.py: transf(), utc_idx_pre(), utc_idx_post()
   - utils/metrics.py: wape(), smape(), mase()

2. MODELI - FUNKCIJE ZA TRENIRANJE (Linije 169-551)
   - models/dense.py: train_dense()
   - models/cnn.py: train_cnn()
   - models/lstm.py: train_lstm(), train_ar_lstm()
   - models/svr.py: train_svr_dir(), train_svr_mimo()
   - models/linear.py: train_linear_model()

3. KONFIGURACIJA (Linije 611-960)
   - config/mts.py: MTS klasa - parametri vremenske serije
   - config/holidays.py: HOL rječnik praznika (AT, DE, CH)
   - config/time_features.py: T klasa s Y, M, W, D, H podklasama
   - config/model.py: MDL klasa - parametri modela

4. DEFINICIJA PODATAKA (Linije 696-1048)
   - data/input_data.py: i_dat, i_dat_inf (ulazni podaci)
   - data/output_data.py: o_dat, o_dat_inf (izlazni podaci)

5. KREIRANJE DATASETA (Linije 1049-1875)
   - pipeline/dataset_creation.py: Glavna petlja za kreiranje dataseta
   - pipeline/time_features_processing.py: Obrada Y, M, W, D, H komponenti
   - pipeline/scaling.py: Kreiranje MinMaxScaler-a

6. VIZUALIZACIJA - VIOLIN PLOTOVI (Linije 1876-2027)
   - visualization/violin_plots.py: Distribucija podataka

7. TRENIRANJE I TESTIRANJE (Linije 2028-2340)
   - training/data_split.py: Podjela na train/val/test (70%/20%/10%)
   - training/model_training.py: Pozivanje odgovarajuće train funkcije
   - training/model_testing.py: Testiranje i re-scaling

8. VIZUALIZACIJA - TEST PLOTOVI (Linije 2340-3241)
   - visualization/test_plots.py: Plotanje rezultata testiranja

9. EVALUACIJA (Linije 3242-3469)
   - evaluation/averaging.py: Mittelwertbildung
   - evaluation/error_calculation.py: MAE, MAPE, MSE, RMSE, NRMSE, WAPE, sMAPE, MASE
   - evaluation/dataframes.py: Kreiranje evaluacijskih DataFrame-ova
"""

###############################################################################
# SEKVENCIJALNI TOK IZVRŠAVANJA ###############################################
###############################################################################

"""
KORAK 1: Učitavanje konfiguracije
    from config.mts import MTS
    from config.holidays import HOL
    from config.time_features import T
    from config.model import MDL

KORAK 2: Učitavanje pomoćnih funkcija
    from utils import load, transf, utc_idx_pre, utc_idx_post
    from utils import wape, smape, mase

KORAK 3: Definicija ulaznih i izlaznih podataka
    # Izvršava se: data/input_data.py
    # Rezultat: i_dat, i_dat_inf, i_list, n_ft_i

    # Izvršava se: data/output_data.py
    # Rezultat: o_dat, o_dat_inf, o_list, n_ft_o

KORAK 4: Kreiranje dataseta
    # Izvršava se: pipeline/dataset_creation.py
    # Izvršava se: pipeline/time_features_processing.py
    # Rezultat: i_array_3D, o_array_3D, utc_ref_log, n_dat

KORAK 5: Skaliranje podataka
    # Izvršava se: pipeline/scaling.py
    # Rezultat: i_scalers, o_scalers

KORAK 6: Violin plotovi (opcionalno)
    # Izvršava se: visualization/violin_plots.py
    # Rezultat: Matplotlib figure

KORAK 7: Podjela podataka
    # Izvršava se: training/data_split.py
    # Rezultat: trn_x, trn_y, val_x, val_y, tst_x, tst_y
    #           trn_x_orig, trn_y_orig, val_x_orig, val_y_orig, tst_x_orig, tst_y_orig

KORAK 8: Treniranje modela
    # Izvršava se: training/model_training.py
    # Koristi: models/dense.py, models/cnn.py, models/lstm.py,
    #          models/svr.py, models/linear.py
    # Rezultat: mdl (trenirani model)

KORAK 9: Testiranje modela
    # Izvršava se: training/model_testing.py
    # Rezultat: tst_fcst, tst_fcst_scal

KORAK 10: Test plotovi
    # Izvršava se: visualization/test_plots.py
    # Rezultat: Matplotlib figure

KORAK 11: Evaluacija
    # Izvršava se: evaluation/averaging.py
    # Rezultat: dat_eval, y_all, fcst_all

    # Izvršava se: evaluation/error_calculation.py
    # Rezultat: dat_eval s MAE, MAPE, MSE, RMSE, NRMSE, WAPE, sMAPE, MASE

    # Izvršava se: evaluation/dataframes.py
    # Rezultat: df_eval, df_eval_ts
"""

###############################################################################
# NAPOMENA O IZVRŠAVANJU ######################################################
###############################################################################

"""
Originalni kod je bio dizajniran kao Jupyter notebook ili Python skripta
gdje se sve izvršava u jednom globalnom namespace-u. Varijable definirane
u jednom dijelu koda automatski su dostupne u svim sljedećim dijelovima.

Ova reorganizacija služi samo za:
- Preglednost i razumijevanje strukture koda
- Dokumentaciju pojedinih komponenti
- Lakše snalaženje u velikom fajlu od ~3500 linija

Za stvarno izvršavanje, originalni training_original.py se pokreće direktno.
"""

###############################################################################
# POPIS SVIH MODULA ###########################################################
###############################################################################

"""
OriginalTraining/
├── main.py                          # Ovaj fajl - orchestrator
├── training_original.py             # Originalni monolitni fajl
│
├── config/                          # Konfiguracija
│   ├── __init__.py
│   ├── mts.py                       # MTS klasa (vremenska serija)
│   ├── holidays.py                  # HOL rječnik praznika
│   ├── time_features.py             # T klasa (Y, M, W, D, H)
│   └── model.py                     # MDL klasa (parametri modela)
│
├── utils/                           # Pomoćne funkcije
│   ├── __init__.py
│   ├── data_loader.py               # load() funkcija
│   ├── time_utils.py                # transf(), utc_idx_pre(), utc_idx_post()
│   └── metrics.py                   # wape(), smape(), mase()
│
├── models/                          # Funkcije za treniranje modela
│   ├── __init__.py
│   ├── dense.py                     # train_dense()
│   ├── cnn.py                       # train_cnn()
│   ├── lstm.py                      # train_lstm(), train_ar_lstm()
│   ├── svr.py                       # train_svr_dir(), train_svr_mimo()
│   └── linear.py                    # train_linear_model()
│
├── data/                            # Definicija podataka
│   ├── __init__.py
│   ├── input_data.py                # Ulazni podaci (i_dat, i_dat_inf)
│   └── output_data.py               # Izlazni podaci (o_dat, o_dat_inf)
│
├── pipeline/                        # Obrada podataka
│   ├── __init__.py
│   ├── dataset_creation.py          # Kreiranje dataseta
│   ├── time_features_processing.py  # Obrada vremenskih komponenti
│   └── scaling.py                   # MinMaxScaler kreiranje
│
├── visualization/                   # Vizualizacija
│   ├── __init__.py
│   ├── violin_plots.py              # Violin distribucije
│   └── test_plots.py                # Test rezultati
│
├── training/                        # Treniranje i testiranje
│   ├── __init__.py
│   ├── data_split.py                # Podjela train/val/test
│   ├── model_training.py            # Treniranje modela
│   └── model_testing.py             # Testiranje modela
│
└── evaluation/                      # Evaluacija
    ├── __init__.py
    ├── averaging.py                 # Mittelwertbildung
    ├── error_calculation.py         # Izračun metrika grešaka
    └── dataframes.py                # Evaluacijski DataFrames
"""

if __name__ == "__main__":
    print("Ovo je demonstracijski orchestrator za training_original.py")
    print("Za izvršavanje koristite originalni fajl: python training_original.py")
    print("\nStruktura modula:")
    print("  - config/: Konfiguracija (MTS, HOL, T, MDL)")
    print("  - utils/: Pomoćne funkcije (load, transf, metrics)")
    print("  - models/: Funkcije za treniranje (Dense, CNN, LSTM, SVR, LIN)")
    print("  - data/: Definicija podataka (input, output)")
    print("  - pipeline/: Obrada (dataset creation, time features, scaling)")
    print("  - visualization/: Vizualizacija (violin plots, test plots)")
    print("  - training/: Treniranje i testiranje")
    print("  - evaluation/: Evaluacija rezultata")
