# Dictionary Scalers & Model Creation Logic

## Pregled Arhitekture

Sistem generiše **2 scaler fajla** i **1 model fajl**:

| Fajl | Sadržaj | Tip |
|------|---------|-----|
| `i_scaler.save` | i_scalers dictionary | `{index: MinMaxScaler}` |
| `o_scaler.save` | o_scalers dictionary | `{index: MinMaxScaler}` |
| `model.h5` | mdl | Keras Sequential model |

---

## 1. Input Scaler Dictionary (i_scalers)

### Struktura
```python
i_scalers = {
    0: MinMaxScaler(),   # Input fajl 1 - podaci
    1: MinMaxScaler(),   # Input fajl 2 - podaci
    ...
    N: MinMaxScaler(),   # Input fajl N - podaci
    # Time features (ako su aktivni):
    N+1: MinMaxScaler(), # Y_sin (godina)
    N+2: MinMaxScaler(), # Y_cos
    N+3: MinMaxScaler(), # M_sin (mjesec)
    N+4: MinMaxScaler(), # M_cos
    N+5: MinMaxScaler(), # W_sin (sedmica)
    N+6: MinMaxScaler(), # W_cos
    N+7: MinMaxScaler(), # D_sin (dan)
    N+8: MinMaxScaler(), # D_cos
    N+9: MinMaxScaler(), # Holiday
}
```

### Formula za broj unosa
```
len(i_scalers) = num_input_files + (active_time_pairs × 2) + holiday_if_active

Gdje su time_pairs: Year, Month, Week, Day (svaki ima sin + cos)
```

### Kreiranje (training_original.py ~linija 1814)
```python
i_scalers = {}

# Konfiguracija iz i_dat_inf DataFrame
i_scal_list = i_dat_inf["scal"].tolist()        # [True, True, ...]
i_scal_max_list = i_dat_inf["scal_max"].tolist() # [1, 1, ...]
i_scal_min_list = i_dat_inf["scal_min"].tolist() # [0, 0, ...]

# Dodavanje time feature konfiguracije
for i in range(len(imp)):  # imp = [Y, M, W, D, Holiday] aktivnost
    if imp[i] == True and scal[i] == True:
        i_scal_list.append(True)   # sin
        i_scal_list.append(True)   # cos
        i_scal_max_list.append(1)
        i_scal_max_list.append(1)
        i_scal_min_list.append(-1)
        i_scal_min_list.append(-1)

# Kreiranje scalera po koloni
for i in range(i_combined_array.shape[1]):
    if i_scal_list[i] == True:
        scaler = MinMaxScaler(feature_range=(i_scal_min_list[i], i_scal_max_list[i]))
        scaler.fit_transform(i_combined_array[:, i].reshape(-1, 1))
        i_scalers[i] = scaler
    else:
        i_scalers[i] = None
```

---

## 2. Output Scaler Dictionary (o_scalers)

### Struktura
```python
o_scalers = {
    0: MinMaxScaler(),  # Output fajl 1 - podaci
    1: MinMaxScaler(),  # Output fajl 2 - podaci
    ...
    M: MinMaxScaler(),  # Output fajl M - podaci
}
```

### Formula za broj unosa
```
len(o_scalers) = num_output_files
```

**Napomena**: Output NEMA time features - samo sirove kolone podataka.

### Kreiranje (training_original.py ~linija 1840)
```python
o_scalers = {}

o_scal_list = o_dat_inf["scal"].tolist()
o_scal_max_list = o_dat_inf["scal_max"].tolist()
o_scal_min_list = o_dat_inf["scal_min"].tolist()

for i in range(o_combined_array.shape[1]):
    if o_scal_list[i] == True:
        scaler = MinMaxScaler(feature_range=(o_scal_min_list[i], o_scal_max_list[i]))
        scaler.fit_transform(o_combined_array[:, i].reshape(-1, 1))
        o_scalers[i] = scaler
    else:
        o_scalers[i] = None
```

---

## 3. Model (mdl)

### Karakteristike
- **Uvijek 1 model** bez obzira na broj input/output fajlova
- Input shape: `(timesteps, len(i_scalers))`
- Output shape: `(len(o_scalers),)`

### Model tipovi
| Tip | Arhitektura |
|-----|-------------|
| Dense | Flatten → Dense layers → Output |
| CNN | Conv1D layers → Flatten → Dense → Output |
| LSTM | LSTM layers → Dense → Output |
| SVR | Support Vector Regression |
| Linear | Linear Regression |

### Trening sa skaliranim podacima
```python
# Skaliranje inputa
for i in range(X.shape[2]):
    if i_scalers[i] is not None:
        X[:, :, i] = i_scalers[i].transform(X[:, :, i].reshape(-1, 1)).reshape(...)

# Skaliranje outputa
for i in range(y.shape[1]):
    if o_scalers[i] is not None:
        y[:, i] = o_scalers[i].transform(y[:, i].reshape(-1, 1)).flatten()

mdl.fit(X, y, epochs=epochs, batch_size=batch_size)
```

---

## 4. Primjer Scenarija

### Scenario: 3 input fajla + 2 output fajla + svi time features

```python
# i_scalers (14 unosa)
i_scalers = {
    0: MinMaxScaler(0,1),   # Input 1
    1: MinMaxScaler(0,1),   # Input 2
    2: MinMaxScaler(0,1),   # Input 3
    3: MinMaxScaler(-1,1),  # Y_sin
    4: MinMaxScaler(-1,1),  # Y_cos
    5: MinMaxScaler(-1,1),  # M_sin
    6: MinMaxScaler(-1,1),  # M_cos
    7: MinMaxScaler(-1,1),  # W_sin
    8: MinMaxScaler(-1,1),  # W_cos
    9: MinMaxScaler(-1,1),  # D_sin
    10: MinMaxScaler(-1,1), # D_cos
    11: MinMaxScaler(0,1),  # Holiday
}

# o_scalers (2 unosa)
o_scalers = {
    0: MinMaxScaler(0,1),  # Output 1
    1: MinMaxScaler(0,1),  # Output 2
}

# mdl (1 model)
mdl = Sequential(...)  # Input: (timesteps, 14), Output: (2,)
```

### Rezultat: 2 .save fajla + 1 .h5 fajl
- `i_scaler.save` → dictionary sa 14 scalera
- `o_scaler.save` → dictionary sa 2 scalera
- `model.h5` → 1 Keras model

---

## 5. Ključne Relacije

### DataFrame → Lists → Dictionary
```
i_dat_inf["scal"]     → i_scal_list     → i_scalers keys
i_dat_inf["scal_max"] → i_scal_max_list → feature_range[1]
i_dat_inf["scal_min"] → i_scal_min_list → feature_range[0]
```

### Array Dimensions → Scaler Count
```
i_combined_array.shape[1] = len(i_scalers)
o_combined_array.shape[1] = len(o_scalers)
```

### Scaler → Model Shape
```
Model input features = len(i_scalers)
Model output features = len(o_scalers)
```

---

## 6. Inference Flow

Pri predikciji koristi se isti mapping:
```python
# Transform input
for i, scaler in i_scalers.items():
    if scaler is not None:
        new_input[:, :, i] = scaler.transform(...)

# Predict
prediction = mdl.predict(new_input)

# Inverse transform output
for i, scaler in o_scalers.items():
    if scaler is not None:
        prediction[:, i] = scaler.inverse_transform(...)
```

---

## Sažetak Formule

```
┌─────────────────────────────────────────────────────────┐
│ FAJL OUTPUTS:                                           │
│   i_scaler.save = 1 fajl (dictionary)                   │
│   o_scaler.save = 1 fajl (dictionary)                   │
│   model.h5      = 1 fajl (Keras model)                  │
├─────────────────────────────────────────────────────────┤
│ DICTIONARY VELIČINE:                                    │
│   len(i_scalers) = N_input + (time_pairs×2) + holiday   │
│   len(o_scalers) = N_output                             │
├─────────────────────────────────────────────────────────┤
│ MODEL:                                                  │
│   Uvijek 1, adapta shape prema scaler dictionary-ima    │
└─────────────────────────────────────────────────────────┘
```
