# Analiza training_backend_test_2.py - Potpuna Dokumentacija

## Pregled Projekta

Ovaj Python skript predstavlja kompleksan sistem za multivarijatnu analizu vremenskih serija (Multivariate Time Series Analysis - MTS) koji implementira mašinsko učenje za prognoziranje na osnovu istorijskih podataka. Sistem podržava različite modele uključujući Dense, CNN, LSTM, SVR i linearne modele.

## Struktura Projekta

### 1. IMPORTS I ZÁVISLOSTI (Linije 1-28)

```python
# Osnove biblioteke
datetime, math, pandas, sys, random, numpy, pytz, copy, calendar

# Vizualizacija
matplotlib.pyplot, seaborn

# Machine Learning - Sklearn
mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
root_mean_squared_error, LinearRegression, make_pipeline, StandardScaler
SVR, MinMaxScaler

# Deep Learning - TensorFlow
tensorflow
```

### 2. KLASA KONFIGURACIJE (Linije 619-632)

```python
class MTS:
    I_N = 13        # Broj vremenskih koraka za ulazne podatke
    O_N = 13        # Broj vremenskih koraka za izlazne podatke
    DELT = 3        # Korak vremena u minutima
    OFST = 0        # Offset u minutima
```

### 3. DEFINISANJE PRAZNIKA (Linije 634-692)

Kompletan kalendar praznika za Austriju, trajan 2022-2025.

## Detaljni Tok Izvršavanja Komandi

### FAZA 1: UČITAVANJE I PREPROCESSING PODATAKA

#### 1.1 Učitavanje Ulaznih Podataka (Linije 724-790)

```python
# Definisanje putanja
path_1 = "data/historical/solarthermics/data_4/Wert 1.csv"  # Mrežno opterećenje
path_2 = "data/historical/solarthermics/data_4/Wert 2.csv"  # Temperatura

# Učitavanje CSV fajlova
i_dat["Netzlast [kW]"] = pd.read_csv(path_1, delimiter=";")
i_dat["Aussentemperatur Krumpendorf [GradC]"] = pd.read_csv(path_2, delimiter=";")

# Procesiranje kroz load() funkciju
i_dat, i_dat_inf = load(i_dat, i_dat_inf)
```

**Funkcija load()** (Linije 37-109):
- Konvertuje UTC stringove u datetime objekte
- Izračunava vremenske parametre (startno/krajnje vreme, broj tačaka)
- Određuje korak vremena i offset
- Proverava numeričke vrednosti i izračunava statistike

#### 1.2 Učitavanje Izlaznih Podataka (Linije 986-1047)

```python
# Slično ulaznim podacima
o_dat["Netzlast [kW]"] = pd.read_csv("Wert 2.csv", delimiter=";")
o_dat["a"] = pd.read_csv("Wert 3.csv", delimiter=";")
```

#### 1.3 Konfiguracija Podataka (Linije 766-790)

```python
# Definisanje specifikacija za svaki dataset
i_dat_inf.loc["Netzlast [kW]", "spec"] = "Historische Daten"
i_dat_inf.loc["Netzlast [kW]", "th_strt"] = -1  # Početni horizont
i_dat_inf.loc["Netzlast [kW]", "th_end"] = 0    # Krajnji horizont
i_dat_inf.loc["Netzlast [kW]", "meth"] = "Lineare Interpolation"

# Skaliranje
i_dat_inf.loc["Netzlast [kW]", "scal"] = True
i_dat_inf.loc["Netzlast [kW]", "scal_max"] = 1
i_dat_inf.loc["Netzlast [kW]", "scal_min"] = 0
```

### FAZA 2: KREIRANJE DATASET-A

#### 2.1 Glavni Loop za Kreiranje Dataset-a (Linije 1079-1748)

```python
while True:
    if utc_ref > utc_end:
        break
    
    # Progres
    prog_1 = (utc_ref-utc_strt)/(utc_end-utc_strt)*100
    print(f"Erstellung der Datensätze: {prog_1:.2f}%")
    
    # Procesiranje ulaznih podataka
    for i, (key, df) in enumerate(i_dat.items()):
        # ... procesiranje ...
    
    # Procesiranje izlaznih podataka
    for i, (key, df) in enumerate(o_dat.items()):
        # ... procesiranje ...
    
    # Procesiranje vremenskih informacija
    # ... sin/cos komponente ...
    
    utc_ref += datetime.timedelta(minutes=MTS.DELT)
```

#### 2.2 Linearna Interpolacija (Linije 1163-1204)

```python
# Za svaki vremenski korak
for i1 in range(len(utc_th)):
    idx1 = utc_idx_pre(i_dat[key], utc_th[i1])
    idx2 = utc_idx_post(i_dat[key], utc_th[i1])
    
    if idx1 == idx2:
        val = i_dat[key].iloc[idx1,1]
    else:
        # Linearna interpolacija
        val = (utc_th[i1]-utc1)/(utc2-utc1)*(val2-val1)+val1
```

### FAZA 3: VREMENSKE KOMPONENTE

#### 3.1 Sin/Cos komponente za različite cikluse (Linije 1369-1740)

```python
# Godišnje (Y), Mesečne (M), Nedeljne (W), Dnevne (D)
class T:
    class Y:  # Godišnje ciklusi
        IMP = False
        SPEC = "Zeithorizont"
        TH_STRT = -24
        TH_END = 0
        
    class M:  # Mesečni ciklusi
        # ...slično...
        
    class W:  # Nedeljni ciklusi
        # ...slično...
        
    class D:  # Dnevni ciklusi
        # ...slično...
        
    class H:  # Praznici
        IMP = False
        CNTRY = "Österreich"
```

#### 3.2 Kreiranje Sin/Cos komponenti

```python
# Za godišnje cikluse
if T.Y.IMP:
    sec = pd.Series(utc_th).map(pd.Timestamp.timestamp)
    df_int_i["y_sin"] = np.sin(sec/31557600*2*np.pi)
    df_int_i["y_cos"] = np.cos(sec/31557600*2*np.pi)

# Za dnevne cikluse
if T.D.IMP:
    sec = np.array([dt.hour*3600+dt.minute*60+dt.second for dt in lt_th])
    df_int_i["d_sin"] = np.sin(sec/86400*2*np.pi)
    df_int_i["d_cos"] = np.cos(sec/86400*2*np.pi)
```

### FAZA 4: SKALIRANJE PODATAKA

#### 4.1 Kreiranje Scalers-a (Linije 1814-1872)

```python
# Za ulazne podatke
i_scalers = {}
for i in range(i_combined_array.shape[1]):
    if i_scal_list[i] == True:
        scaler = MinMaxScaler(feature_range=(i_scal_min_list[i], i_scal_max_list[i]))
        scaler.fit_transform(i_combined_array[:, i].reshape(-1, 1))
        i_scalers[i] = scaler
    else:
        i_scalers[i] = None

# Slično za izlazne podatke
o_scalers = {}
```

#### 4.2 Primena Skaliranja (Linije 2181-2210)

```python
for i in range(n_dat):
    for i1 in range(n_ft_i):
        if not i_scalers[i1] is None:
            std_i = i_scalers[i1].transform(i_array_3D[i,:,i1].reshape(-1, 1))
            i_array_3D[i,:,i1] = std_i.ravel()
```

### FAZA 5: PODELA PODATAKA I KREIRANJE MODELA

#### 5.1 Train/Validation/Test Split (Linije 2040-2234)

```python
n_train = round(0.7*n_dat)  # 70% za obuku
n_val = round(0.2*n_dat)    # 20% za validaciju
n_tst = n_dat-n_train-n_val # 10% za testiranje

trn_x = i_array_3D[:n_train]
val_x = i_array_3D[n_train:(n_train+n_val)]
tst_x = i_array_3D[(n_train+n_val):]
```

#### 5.2 Model Konfiguracija (Linije 2046-2157)

```python
class MDL:
    MODE = "LIN"  # Dostupno: "Dense", "CNN", "LSTM", "AR LSTM", "SVR_dir", "SVR_MIMO", "LIN"
    
    # Za Neural Network modele
    if MODE == "Dense":
        LAY = 3      # Broj slojeva
        N = 512      # Broj neurona
        EP = 20      # Broj epoha
        ACTF = "ReLU"
    
    # Za SVR modele
    elif MODE == "SVR_dir":
        KERNEL = "poly"
        C = 1
        EPSILON = 0.1
```

### FAZA 6: TRENING MODELA

#### 6.1 Funkcije za Trening Različitih Modela

**Dense Model** (Linije 170-237):
```python
def train_dense(train_x, train_y, val_x, val_y, MDL):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    
    for _ in range(MDL.LAY):
        model.add(tf.keras.layers.Dense(MDL.N, activation=MDL.ACTF))
    
    model.add(tf.keras.layers.Dense(train_y.shape[1]*train_y.shape[2]))
    model.add(tf.keras.layers.Reshape([train_y.shape[1], train_y.shape[2]]))
    
    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    model.fit(train_x, train_y, epochs=MDL.EP, callbacks=[earlystopping])
```

**LSTM Model** (Linije 321-387):
```python
def train_lstm(train_x, train_y, val_x, val_y, MDL):
    model = tf.keras.Sequential()
    
    for i in range(MDL.LAY):
        if i == 0:
            model.add(tf.keras.layers.LSTM(
                units=MDL.N, return_sequences=True, input_shape=train_x.shape[1:]))
        else:
            model.add(tf.keras.layers.LSTM(
                units=MDL.N, return_sequences=True))
    
    model.add(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(train_y.shape[-1])))
```

**SVR Model** (Linije 458-529):
```python
def train_svr_dir(train_x, train_y, MDL):
    n_samples, n_timesteps, n_features = train_x.shape
    X = train_x.reshape(n_samples * n_timesteps, n_features)
    
    model = []
    for i in range(n_features):
        y_i = train_y[:, :, i].reshape(-1)
        model.append(make_pipeline(
            StandardScaler(), 
            SVR(kernel=MDL.KERNEL, C=MDL.C, epsilon=MDL.EPSILON)))
        model[-1].fit(X, y_i)
```

**Linear Model** (Linije 531-551):
```python
def train_linear_model(trn_x, trn_y):
    n_samples, n_timesteps, n_features_in = trn_x.shape
    _, _, n_features_out = trn_y.shape
    
    X = trn_x.reshape(n_samples * n_timesteps, n_features_in)
    
    models = []
    for i in range(n_features_out):
        y_i = trn_y[:, :, i].reshape(-1)
        model = LinearRegression()
        model.fit(X, y_i)
        models.append(model)
```

#### 6.2 Pokretanje Treninga (Linije 2240-2259)

```python
if MDL.MODE == "Dense":
    mdl = train_dense(trn_x, trn_y, val_x, val_y, MDL)
elif MDL.MODE == "CNN":
    mdl = train_cnn(trn_x, trn_y, val_x, val_y, MDL)
elif MDL.MODE == "LSTM":
    mdl = train_lstm(trn_x, trn_y, val_x, val_y, MDL)
elif MDL.MODE == "LIN":
    mdl = train_linear_model(trn_x, trn_y)
```

### FAZA 7: TESTIRANJE I EVALUACIJA

#### 7.1 Kreiranje Predikcija (Linije 2265-2305)

```python
tst_fcst = list()

for i in range(n_tst):
    inp = tst_x[i, :, :].reshape((1, tst_x.shape[1], n_ft_i))
    
    if MDL.MODE in ["SVR_dir", "SVR_MIMO", "LIN"]:
        inp = np.squeeze(inp, axis=0)
        pred = []
        for j in range(n_ft_o):
            pred.append(mdl[j].predict(inp))
        out = np.array(pred).T
        out = np.expand_dims(out, axis=0)
    else:
        out = mdl.predict(inp, verbose=0)
    
    tst_fcst.append(out[0,:,:])

tst_fcst = np.array(tst_fcst)
```

#### 7.2 Re-scaling (Linije 2312-2332)

```python
for i in range(n_tst):
    for i1 in range(n_ft_o):
        if not o_scalers[i1] is None:
            # Inverzno skaliranje
            a = o_scalers[i1].inverse_transform(tst_fcst[i,:,i1].reshape(-1, 1))
            b = o_scalers[i1].inverse_transform(tst_y[i,:,i1].reshape(-1, 1))
            
            tst_fcst[i,:,i1] = a.ravel()
            tst_y[i,:,i1] = b.ravel()
```

### FAZA 8: VIZUALIZACIJA

#### 8.1 Violin Plots (Linije 1875-2026)

```python
# Kreiranje Violin plots za distribuciju podataka
fig, axes = plt.subplots(1, n_ft_i, figsize=(2*n_ft_i, 6))

for i, (name, values) in enumerate(data.items()):
    sns.violinplot(y=values, ax=axes[i], color=palette[i])
    axes[i].set_title(name)

plt.suptitle("Datenverteilung der Eingabedaten")
plt.show()
```

#### 8.2 Rezultati Testiranja (Linije 2334-3241)

```python
# Kreiranje subplotova za prikaz rezultata
fig, axs = plt.subplots(num_sbpl_y, num_sbpl_x, figsize=(20, 13))

# Plotovanje ulaznih podataka, izlaznih podataka i predikcija
for i_sbpl in range(num_sbpl):
    # ... kompleksno plotovanje sa multiple y-axes ...
    
    # Plotovanje ulaznih podataka
    if df_plot_in.iloc[i_feat, 0]:
        ax_sbpl[-1].plot(x_value, y_value, color=color_plt, marker='o')
    
    # Plotovanje izlaznih podataka  
    if df_plot_out.iloc[i_feat, 0]:
        ax_sbpl[-1].plot(x_value, y_value, color=color_plt, marker='o')
    
    # Plotovanje predikcija
    if df_plot_fcst.iloc[i_feat, 0]:
        ax_sbpl[-1].plot(x_value, y_value, color=color_plt, marker='x', linestyle='--')
```

### FAZA 9: EVALUACIJA PERFORMANSI

#### 9.1 Funkcije za Metriku Evaluacije

**WAPE** (Weighted Absolute Percentage Error) - Linije 554-566:
```python
def wape(y_true, y_pred):
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return np.nan
    return (numerator/denominator)*100
```

**SMAPE** (Symmetric Mean Absolute Percentage Error) - Linije 570-585:
```python
def smape(y_true, y_pred):
    smape_values = []
    for yt, yp in zip(y_true, y_pred):
        denominator = (abs(yt)+abs(yp))/2
        if denominator == 0:
            smape_values.append(0)
        else:
            smape_values.append(abs(yp-yt)/denominator)
    return sum(smape_values)/len(y_true)*100
```

**MASE** (Mean Absolute Scaled Error) - Linije 587-608:
```python
def mase(y_true, y_pred, m=1):
    mae_forecast = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)
    naive_errors = [abs(y_true[t] - y_true[t - m]) for t in range(m, len(y_true))]
    mae_naive = sum(naive_errors) / len(naive_errors)
    if mae_naive == 0:
        raise ZeroDivisionError("Naive MAE je 0 – MASE nije definisan.")
    return mae_forecast/mae_naive
```

#### 9.2 Kreiranje Evaluacionih Tabela (Linije 3295-3468)

```python
# Kreiranje evaluacionih tabela za različite metrije
dat_eval = {}

# Računanje različitih metrika
for i in range(n_max):
    mae_int, mape_int, mse_int, rmse_int, nrmse_int, wape_int, smape_int, mase_int = ([] for _ in range(8))
    
    for i_feat in range(num_feat):
        v_true = y_all[i,:,:,i_feat]
        v_fcst = fcst_all[i,:,:,i_feat]
        
        mask = ~np.isnan(v_true) & ~np.isnan(v_fcst)
        
        mae_int.append(mae(v_true[mask], v_fcst[mask]))
        mape_int.append(100*mape(v_true[mask], v_fcst[mask]))
        # ... ostale metrije ...
    
    dat_eval[i+1]["MAE"] = np.array(mae_int)
    dat_eval[i+1]["MAPE"] = np.array(mape_int)
    # ... ostale metrije ...
```

## Algoritam za Obradu Podataka

### 1. ULAZNI PODACI
- **Format**: CSV fajlovi sa kolonima UTC i numerička vrednost
- **Tipovi**: Historijski podaci (mrežno opterećenje, temperatura)
- **Procesiranje**: Linearna interpolacija za nedostajuće vrednosti

### 2. VREMENSKE KOMPONENTE
- **Sin/Cos komponente** za različite cikluse:
  - Godišnje (31557600 sekundi)
  - Mesečne (2629800 sekundi)
  - Nedeljne (604800 sekundi)
  - Dnevne (86400 sekundi)
- **Praznici**: Binarna vrednost (0/1) za austriške praznike

### 3. ARHITEKTURA MODELA

#### Dense Model:
- Input Layer: Flatten()
- Hidden Layers: Dense(512) × 3, ReLU aktivacija
- Output Layer: Dense → Reshape za multivariate output

#### LSTM Model:
- LSTM slojevi sa return_sequences=True
- TimeDistributed Dense layer za output
- Pogodan za sekvencijalne podatke

#### CNN Model:
- Conv2D slojevi sa padding='same'
- Kernel size i broj filtera se mogu konfigurisati
- Reshape ulaza u 4D format

#### SVR Model:
- StandardScaler preprocessing
- Polynomial kernel
- Treniran zasebno za svaki output feature

### 4. METRIJE EVALUACIJE
- **MAE**: Srednji apsolutni error
- **MAPE**: Srednji apsolutni procentualni error
- **RMSE**: Koren srednjeg kvadratnog error-a
- **WAPE**: Ponderisani apsolutni procentualni error
- **SMAPE**: Simetrični srednji apsolutni procentualni error
- **MASE**: Skaliran srednji apsolutni error

## Struktura Izvršavanja

1. **Inicijalizacija** (Linije 1-700)
2. **Učitavanje podataka** (Linije 700-1050)
3. **Kreiranje dataset-a** (Linije 1050-1750)
4. **Skaliranje** (Linije 1750-1875)
5. **Vizualizacija distribucije** (Linije 1875-2030)
6. **Podela podataka** (Linije 2030-2240)
7. **Trening modela** (Linije 2240-2260)
8. **Testiranje** (Linije 2260-2335)
9. **Vizualizacija rezultata** (Linije 2335-3245)
10. **Evaluacija** (Linije 3245-3468)

## Konfigurabilne Komponente

### Vremenske komponente:
```python
T.Y.IMP = False  # Godišnje komponente
T.M.IMP = False  # Mesečne komponente  
T.W.IMP = False  # Nedeljne komponente
T.D.IMP = False  # Dnevne komponente
T.H.IMP = False  # Praznici
```

### Model parametri:
```python
MDL.MODE = "LIN"    # Tip modela
MDL.LAY = 3         # Broj slojeva
MDL.N = 512         # Broj neurona/filtera
MDL.EP = 20         # Broj epoha
```

### Dataset parametri:
```python
MTS.I_N = 13    # Input timesteps
MTS.O_N = 13    # Output timesteps  
MTS.DELT = 3    # Vremenski korak (minuti)
MTS.OFST = 0    # Offset (minuti)
```

## Optimizacije i Najbolje Prakse

1. **EarlyStopping** callback za sprečavanje overfitting-a
2. **MinMaxScaler** za normalizaciju features
3. **Progres bar** za praćenje izvršavanja
4. **Error handling** za robusnost
5. **Modularni design** za lakše proširivanje
6. **Comprehensivna evaluacija** sa multiple metrikama

## Ključne Funkcije

- `load()`: Učitavanje i analiza podataka
- `transf()`: Računanje vremenskih parametara
- `utc_idx_pre()/utc_idx_post()`: Pronalaženje indeksa za interpolaciju
- `train_*()`: Funkcije za trening različitih modela
- `wape()/smape()/mase()`: Custom metrije evaluacije

Sistem predstavlja kompletan pipeline za multivariate time series forecasting sa podrškom za različite arhitekture modela i comprehensivnu evaluaciju performansi.