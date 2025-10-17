# Training Models - Kompletan Spisak i Specifikacija

**Datum:** 2025-10-17
**Verzija:** 1.0
**Ukupno modela:** 7

---

## 📑 Sadržaj

1. [Pregled Modela](#pregled-modela)
2. [Neural Network Modeli](#neural-network-modeli)
3. [SVR Modeli](#svr-modeli)
4. [Linear Model](#linear-model)
5. [Parametri i Validacija](#parametri-i-validacija)
6. [Backend Format](#backend-format)
7. [Primjeri JSON Payload-a](#primjeri-json-payload-a)

---

## Pregled Modela

### Dostupni Model Tipovi

| # | Model Tip | Kategorija | Opis |
|---|-----------|------------|------|
| 1 | **Dense** | Neural Network | Dense (fully connected) neuronska mreža |
| 2 | **CNN** | Neural Network | Convolutional Neural Network |
| 3 | **LSTM** | Neural Network | Long Short-Term Memory rekurentna mreža |
| 4 | **AR LSTM** | Neural Network | Autoregressive LSTM |
| 5 | **SVR_dir** | Support Vector | Support Vector Regression (Direct) |
| 6 | **SVR_MIMO** | Support Vector | Support Vector Regression (Multi-Input Multi-Output) |
| 7 | **LIN** | Linear | Linearna regresija |

**Izvor:** [ModelConfiguration.tsx:34-42](src/features/training/components/ModelConfiguration.tsx#L34-42)

---

## Neural Network Modeli

### 1. Dense (Fully Connected Neural Network)

**Tip:** `Dense`

#### Obavezni Parametri

| Parametar | Tip | Opis | Raspon | Default |
|-----------|-----|------|--------|---------|
| `LAY` | integer | Broj skrivenih slojeva | 1-10 | - |
| `N` | integer | Broj neurona po sloju | 1-2048 | - |
| `EP` | integer | Broj epoha treniranja | 1-1000 | - |
| `ACTF` | string | Aktivaciona funkcija | `ReLU`, `Sigmoid`, `Tanh`, `Linear`, `Softmax`, `None` | - |

#### Primjer UI Parametara
```typescript
{
  MODE: "Dense",
  LAY: 3,
  N: 128,
  EP: 100,
  ACTF: "ReLU"
}
```

#### Primjer Backend Formata
```json
{
  "MODE": "Dense",
  "LAY": 3,
  "N": 128,
  "EP": 100,
  "ACTF": "ReLU"
}
```

---

### 2. CNN (Convolutional Neural Network)

**Tip:** `CNN`

#### Obavezni Parametri

| Parametar | Tip | Opis | Raspon | Napomena |
|-----------|-----|------|--------|----------|
| `LAY` | integer | Broj konvolucijskih slojeva | 1-10 | - |
| `N` | integer | Broj filtera po sloju | 1-2048 | Step: 32 |
| `K` | integer | Veličina kernela | 1-11 | **Mora biti neparan broj** |
| `EP` | integer | Broj epoha | 1-1000 | - |
| `ACTF` | string | Aktivaciona funkcija | `ReLU`, `Sigmoid`, `Tanh`, `Linear`, `Softmax`, `None` | - |

#### Validacija
- **K** mora biti neparan broj (1, 3, 5, 7, 9, 11)
- Ako je paran, automatski se povećava za 1

#### Primjer UI Parametara
```typescript
{
  MODE: "CNN",
  LAY: 2,
  N: 64,
  K: 3,
  EP: 50,
  ACTF: "ReLU"
}
```

---

### 3. LSTM (Long Short-Term Memory)

**Tip:** `LSTM`

#### Obavezni Parametri

| Parametar | Tip | Opis | Raspon |
|-----------|-----|------|--------|
| `LAY` | integer | Broj LSTM slojeva | 1-10 |
| `N` | integer | Broj LSTM jedinica | 1-2048 |
| `EP` | integer | Broj epoha | 1-1000 |
| `ACTF` | string | Aktivaciona funkcija | `ReLU`, `Sigmoid`, `Tanh`, `Linear`, `Softmax`, `None` |

#### Primjer UI Parametara
```typescript
{
  MODE: "LSTM",
  LAY: 2,
  N: 128,
  EP: 100,
  ACTF: "Tanh"
}
```

---

### 4. AR LSTM (Autoregressive LSTM)

**Tip:** `AR LSTM`

#### Obavezni Parametri

| Parametar | Tip | Opis | Raspon |
|-----------|-----|------|--------|
| `LAY` | integer | Broj AR LSTM slojeva | 1-10 |
| `N` | integer | Broj LSTM jedinica | 1-2048 |
| `EP` | integer | Broj epoha | 1-1000 |
| `ACTF` | string | Aktivaciona funkcija | `ReLU`, `Sigmoid`, `Tanh`, `Linear`, `Softmax`, `None` |

#### Primjer UI Parametara
```typescript
{
  MODE: "AR LSTM",
  LAY: 3,
  N: 256,
  EP: 150,
  ACTF: "ReLU"
}
```

---

## SVR Modeli

### 5. SVR_dir (Support Vector Regression - Direct)

**Tip:** `SVR_dir`

#### Obavezni Parametri

| Parametar | Tip | Opis | Raspon/Opcije |
|-----------|-----|------|---------------|
| `KERNEL` | string | Tip kernela | `linear`, `poly`, `rbf`, `sigmoid` |
| `C` | float | Regularizacioni parametar | 0.001-1000 |
| `EPSILON` | float | Epsilon parametar | 0.001-1 |

#### Primjer UI Parametara
```typescript
{
  MODE: "SVR_dir",
  KERNEL: "rbf",
  C: 1.0,
  EPSILON: 0.1
}
```

#### Primjer Backend Formata
```json
{
  "MODE": "SVR_dir",
  "KERNEL": "rbf",
  "C": 1.0,
  "EPSILON": 0.1
}
```

---

### 6. SVR_MIMO (Support Vector Regression - Multi-Output)

**Tip:** `SVR_MIMO`

#### Obavezni Parametri

| Parametar | Tip | Opis | Raspon/Opcije |
|-----------|-----|------|---------------|
| `KERNEL` | string | Tip kernela | `linear`, `poly`, `rbf`, `sigmoid` |
| `C` | float | Regularizacioni parametar | 0.001-1000 |
| `EPSILON` | float | Epsilon parametar | 0.001-1 |

#### Razlika između SVR_dir i SVR_MIMO
- **SVR_dir**: Direct multi-step forecast (jedan output po predict pozivu)
- **SVR_MIMO**: Multi-Input Multi-Output (predviđa sve output korake odjednom)

#### Primjer UI Parametara
```typescript
{
  MODE: "SVR_MIMO",
  KERNEL: "linear",
  C: 10.0,
  EPSILON: 0.05
}
```

---

## Linear Model

### 7. LIN (Linear Regression)

**Tip:** `LIN`

#### Parametri

**Nema dodatnih parametara** - koristi default linearnu regresiju.

#### Primjer UI Parametara
```typescript
{
  MODE: "LIN"
}
```

#### Primjer Backend Formata
```json
{
  "MODE": "LIN"
}
```

#### UI Prikaz
Prikazuje poruku: "Linear model uses default configuration"

**Izvor:** [ModelConfiguration.tsx:385-394](src/features/training/components/ModelConfiguration.tsx#L385-394)

---

## Parametri i Validacija

### Aktivacione Funkcije

Dostupne za Neural Network modele:

| Funkcija | Opis | Tipična Upotreba |
|----------|------|------------------|
| `ReLU` | Rectified Linear Unit | Default za većinuNN |
| `Sigmoid` | Sigmoid (logistic) | Binary classification |
| `Tanh` | Hyperbolic Tangent | LSTM, rekurentne mreže |
| `Linear` | Linearna aktivacija | Output layer za regresiju |
| `Softmax` | Softmax | Multi-class classification |
| `None` | Bez aktivacije | Specijalni slučajevi |

**Izvor:** [ModelConfiguration.tsx:44-52](src/features/training/components/ModelConfiguration.tsx#L44-52)

---

### Kernel Tipovi (SVR)

Dostupni za SVR modele:

| Kernel | Opis | Kada Koristiti |
|--------|------|----------------|
| `linear` | Linearni kernel | Linearni odnosi |
| `poly` | Polinomijalni kernel | Nelinearni odnosi sa stepenom |
| `rbf` | Radial Basis Function | Default, dobro za većinu slučajeva |
| `sigmoid` | Sigmoid kernel | Neural network aproksimacija |

**Izvor:** [ModelConfiguration.tsx:54-60](src/features/training/components/ModelConfiguration.tsx#L54-60)

---

### Validaciona Pravila

#### Neural Network Modeli (Dense, CNN, LSTM, AR LSTM)

```typescript
{
  LAY: {
    min: 1,
    max: 10,
    required: true,
    type: "integer"
  },
  N: {
    min: 1,
    max: 2048,
    required: true,
    type: "integer",
    step: 32  // preporuka
  },
  EP: {
    min: 1,
    max: 1000,
    required: true,
    type: "integer"
  },
  ACTF: {
    required: true,
    type: "enum",
    values: ["ReLU", "Sigmoid", "Tanh", "Linear", "Softmax", "None"]
  }
}
```

#### CNN Dodatna Validacija

```typescript
{
  K: {
    min: 1,
    max: 11,
    required: true,
    type: "integer",
    constraint: "Must be odd number",
    step: 2
  }
}
```

#### SVR Modeli (SVR_dir, SVR_MIMO)

```typescript
{
  KERNEL: {
    required: true,
    type: "enum",
    values: ["linear", "poly", "rbf", "sigmoid"]
  },
  C: {
    min: 0.001,
    max: 1000,
    required: true,
    type: "float",
    step: 0.1
  },
  EPSILON: {
    min: 0.001,
    max: 1,
    required: true,
    type: "float",
    step: 0.01
  }
}
```

**Izvor:** [modelParameterUtils.ts:94-198](src/utils/modelParameterUtils.ts#L94-198)

---

## Backend Format

### Konverzija UI → Backend

Frontend šalje podatke koji se konvertuju funkcijom `convertUIToBackendFormat()`:

**Izvor:** [modelParameterUtils.ts:32-87](src/utils/modelParameterUtils.ts#L32-87)

#### Process Flow

```
UI Parameters (ModelParameters)
        ↓
convertUIToBackendFormat()
        ↓
Backend Parameters (BackendModelParameters)
        ↓
POST /api/training/train-models/{sessionId}
```

#### Backend Interface

```typescript
interface BackendModelParameters {
  MODE: string;     // Model type
  LAY?: number;     // Number of layers
  N?: number;       // Neurons/filters per layer
  EP?: number;      // Epochs
  ACTF?: string;    // Activation function
  K?: number;       // Kernel size (CNN)
  KERNEL?: string;  // Kernel type (SVR)
  C?: number;       // C parameter (SVR)
  EPSILON?: number; // Epsilon (SVR)
}
```

---

## Primjeri JSON Payload-a

### Complete Training Request Payload

```json
{
  "model_parameters": {
    "MODE": "Dense",
    "LAY": 3,
    "N": 128,
    "EP": 100,
    "ACTF": "ReLU"
  },
  "training_split": {
    "trainPercentage": 70,
    "valPercentage": 20,
    "testPercentage": 10,
    "random_dat": true,
    "n_train": 0,
    "n_val": 0,
    "n_test": 0
  }
}
```

### Dense Model Payload
```json
{
  "model_parameters": {
    "MODE": "Dense",
    "LAY": 2,
    "N": 64,
    "EP": 50,
    "ACTF": "ReLU"
  }
}
```

### CNN Model Payload
```json
{
  "model_parameters": {
    "MODE": "CNN",
    "LAY": 3,
    "N": 128,
    "K": 5,
    "EP": 100,
    "ACTF": "ReLU"
  }
}
```

### LSTM Model Payload
```json
{
  "model_parameters": {
    "MODE": "LSTM",
    "LAY": 2,
    "N": 256,
    "EP": 150,
    "ACTF": "Tanh"
  }
}
```

### AR LSTM Model Payload
```json
{
  "model_parameters": {
    "MODE": "AR LSTM",
    "LAY": 3,
    "N": 128,
    "EP": 200,
    "ACTF": "ReLU"
  }
}
```

### SVR_dir Model Payload
```json
{
  "model_parameters": {
    "MODE": "SVR_dir",
    "KERNEL": "rbf",
    "C": 1.0,
    "EPSILON": 0.1
  }
}
```

### SVR_MIMO Model Payload
```json
{
  "model_parameters": {
    "MODE": "SVR_MIMO",
    "KERNEL": "linear",
    "C": 10.0,
    "EPSILON": 0.05
  }
}
```

### Linear Model Payload
```json
{
  "model_parameters": {
    "MODE": "LIN"
  }
}
```

---

## Model Selection Guide

### Kada Koristiti Koji Model?

#### Dense (Fully Connected)
- ✅ Tabular data sa fiksnim brojem feature-a
- ✅ Brzo treniranje
- ✅ Dobra baseline opcija
- ❌ Ne hvata sekvencijalne/vremenske zavisnosti

#### CNN
- ✅ Grid-like data (slike, 2D vremenski nizovi)
- ✅ Lokalne pattern-e
- ✅ Translation invariance
- ❌ Zahtijeva više računanja

#### LSTM
- ✅ Sekvencijalni data (time series)
- ✅ Dugoročne zavisnosti
- ✅ Variable-length sequences
- ❌ Sporo treniranje
- ❌ Zahtijeva više podataka

#### AR LSTM
- ✅ Autoregressive time series
- ✅ Multi-step forecasting
- ✅ Sekvencijalne zavisnosti
- ❌ Kompleksnije od standardnog LSTM

#### SVR_dir
- ✅ Manji datasets
- ✅ Direct multi-step prediction
- ✅ Ne zahtijeva GPU
- ❌ Sporije sa velikim datasetima

#### SVR_MIMO
- ✅ Multi-output prediction
- ✅ Simultano predviđanje više koraka
- ✅ Efikasnije od SVR_dir za multi-step
- ❌ Memorijski zahtjevnije

#### LIN (Linear)
- ✅ Baseline model
- ✅ Veoma brzo treniranje
- ✅ Interpretabilni rezultati
- ❌ Samo linearni odnosi

---

## Quick Reference

### Model Comparison Table

| Model | Parametri | Brzina | Precision | Use Case |
|-------|-----------|--------|-----------|----------|
| Dense | LAY, N, EP, ACTF | ⚡⚡⚡ | ⭐⭐⭐ | General purpose, tabular |
| CNN | LAY, N, K, EP, ACTF | ⚡⚡ | ⭐⭐⭐⭐ | Spatial patterns, grid data |
| LSTM | LAY, N, EP, ACTF | ⚡ | ⭐⭐⭐⭐ | Time series, sequences |
| AR LSTM | LAY, N, EP, ACTF | ⚡ | ⭐⭐⭐⭐ | Autoregressive forecasting |
| SVR_dir | KERNEL, C, EPSILON | ⚡⚡ | ⭐⭐⭐ | Small datasets, direct |
| SVR_MIMO | KERNEL, C, EPSILON | ⚡⚡ | ⭐⭐⭐ | Multi-output prediction |
| LIN | - | ⚡⚡⚡⚡ | ⭐⭐ | Baseline, linear only |

---

## Validation Error Messages

### Common Validation Errors

```typescript
// Missing MODE
"Model type (MODE) is required"

// Invalid MODE
"Invalid model type: XYZ. Must be one of: Dense, CNN, LSTM, AR LSTM, SVR_dir, SVR_MIMO, LIN"

// Neural Network Errors
"Number of layers is required for neural network models"
"Number of neurons/filters is required"
"Number of epochs is required for neural network models"
"Activation function is required for neural network models"

// CNN Specific
"Kernel size is required for CNN models"
"Kernel size should be an odd number between 1 and 11"

// SVR Errors
"Kernel type is required for SVR models. Must be one of: linear, poly, rbf, sigmoid"
"C parameter is required for SVR models"
"Epsilon is required for SVR models"

// Range Warnings
"Number of layers should be between 1 and 10"
"Number of neurons/filters should be between 1 and 2048"
"Number of epochs should be between 1 and 1000"
"C parameter should be positive"
"Epsilon should be between 0 and 1"
```

---

## Testing Checklist

### Backend Testing - Preporučene Testove za Svaki Model

- [ ] Dense model sa minimalnim parametrima
- [ ] Dense model sa maksimalnim parametrima
- [ ] CNN sa različitim kernel veličinama (3, 5, 7)
- [ ] LSTM sa različitim brojem slojeva
- [ ] AR LSTM sa različitim konfig
- [ ] SVR_dir sa svakim kernel tipom
- [ ] SVR_MIMO sa različitim C i EPSILON vrednostima
- [ ] LIN model (bez dodatnih parametara)
- [ ] Invalid model type handling
- [ ] Missing required parameters

---

**Generisano:** 2025-10-17
**Autor:** Claude Code Analysis
**Status:** ✅ Kompletna specifikacija svih modela
