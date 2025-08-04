# 📊 IZVEŠTAJ O HARDKODOVANIM VREDNOSTIMA U TRAINING SISTEMU

**Datum analize**: 2025-01-31  
**Analizirani sistem**: Frontend (React/TypeScript) + Backend (Flask/Python)

---

## 🔍 REZIME

Pronađeno je **15+ hardkodovanih vrednosti** koje bi trebalo da budu konfigurabine od strane korisnika. Najkritičnije su one vezane za training proces koje direktno utiču na performanse modela.

---

## 📋 LISTA HARDKODOVANIH VREDNOSTI

### 🎨 FRONTEND (Training.tsx)

#### ✅ Vrednosti koje korisnik MOŽE da menja:
1. **Model parametri** (preko ModelConfiguration komponente):
   - MODE (tip modela)
   - LAY (broj slojeva)
   - N (broj neurona/filtera)
   - EP (broj epoha)
   - ACTF (aktivaciona funkcija)
   - K (kernel size za CNN)
   - KERNEL (tip kernela za SVR)
   - C i EPSILON (SVR parametri)

2. **Training data split** (preko TrainingDataSplit komponente):
   - trainPercentage
   - valPercentage  
   - testPercentage
   - random_dat (randomizacija)

#### ❌ Vrednosti koje korisnik NE MOŽE da menja:
```typescript
const POLLING_CONFIG = {
    DATASET_GENERATION_INTERVAL: 5000,      // ⚠️ HARDKODOVANO
    MODEL_TRAINING_INTERVAL: 10000,         // ⚠️ HARDKODOVANO
    ANALYSIS_STATUS_INTERVAL: 10000,        // ⚠️ HARDKODOVANO
    MAX_POLLING_ATTEMPTS: 100,              // ⚠️ HARDKODOVANO
    TIMEOUT_MINUTES: 5                      // ⚠️ HARDKODOVANO
};
```

### 🔧 BACKEND (parameter_converter.py i config.py)

#### ❌ KRITIČNE hardkodovane vrednosti koje korisnik NE MOŽE da menja:

##### 1. **Training parametri za sve Neural Network modele**:
```python
VAL_S: float = 0.2          # ⚠️ Validation split - UVEK 20%
BS: int = 32                # ⚠️ Batch size - UVEK 32
LR: float = 0.001           # ⚠️ Learning rate - UVEK 0.001
OPT: str = "adam"           # ⚠️ Optimizer - UVEK Adam
LOSS: str = "mse"           # ⚠️ Loss function - UVEK MSE
METRICS: List[str] = ["mae"] # ⚠️ Metrics - UVEK samo MAE
```

##### 2. **CNN specifične hardkodovane vrednosti**:
```python
L1_P: int = 2               # ⚠️ Pool size za prvi sloj
L2_P: int = 2               # ⚠️ Pool size za drugi sloj
L3_N: int = 50              # ⚠️ Dense layer neurons - FIKSNO 50
```

##### 3. **LSTM specifične hardkodovane vrednosti**:
```python
L1_D: float = 0.2           # ⚠️ Dropout za prvi LSTM sloj
L2_D: float = 0.2           # ⚠️ Dropout za drugi LSTM sloj
L1_RS: bool = True          # ⚠️ Return sequences za prvi sloj
L2_RS: bool = False         # ⚠️ Return sequences za drugi sloj
L3_N: int = 25              # ⚠️ Dense layer neurons - FIKSNO 25
```

##### 4. **SVR specifične hardkodovane vrednosti**:
```python
GAMMA: str = "scale"        # ⚠️ Kernel coefficient
DEGREE: int = 3             # ⚠️ Polynomial degree
COEF0: float = 0.0          # ⚠️ Independent term
SHRINKING: bool = True      # ⚠️ Shrinking heuristic
TOL: float = 0.001          # ⚠️ Tolerance
CACHE_SIZE: int = 200       # ⚠️ Cache size in MB
MAX_ITER: int = -1          # ⚠️ Maximum iterations
```

---

## 🚨 KRITIČNA ANALIZA

### Najvažnije problemi:

1. **Validation Split (VAL_S = 0.2)**
   - Trenutno: UVEK 20% podataka za validaciju
   - Problem: Korisnik ne može da prilagodi ovaj odnos svojim potrebama
   - Uticaj: Za male datasets, 20% može biti previše za validaciju

2. **Batch Size (BS = 32)**
   - Trenutno: FIKSNO 32 za sve modele
   - Problem: Različiti modeli i datasets zahtevaju različite batch sizes
   - Uticaj: Može negativno uticati na konvergenciju i brzinu treniranja

3. **Learning Rate (LR = 0.001)**
   - Trenutno: FIKSNO 0.001 za sve modele
   - Problem: Kritičan hiperparametar koji zavisi od problema
   - Uticaj: Može dovesti do lošeg treniranja ili divergencije

4. **Optimizer (OPT = "adam")**
   - Trenutno: Samo Adam optimizer
   - Problem: Različiti problemi mogu zahtevati različite optimizatore
   - Uticaj: Ograničava fleksibilnost treniranja

---

## 💡 PREPORUKE ZA IZMENE

### 🔴 PRIORITET 1 (Kritično):

1. **Dodati u ModelConfiguration.tsx**:
   ```typescript
   // Za sve modele
   batchSize: number
   learningRate: number  
   optimizer: 'adam' | 'sgd' | 'rmsprop' | 'adagrad'
   validationSplit: number (0.1 - 0.4)
   ```

2. **Dodati za CNN**:
   ```typescript
   poolSizes: number[]  // Za svaki conv layer
   denseLayers: number[] // Konfigurabini dense slojevi
   ```

3. **Dodati za LSTM**:
   ```typescript
   dropoutRates: number[] // Za svaki LSTM layer
   returnSequences: boolean[] // Za svaki LSTM layer
   ```

### 🟡 PRIORITET 2 (Važno):

1. **Loss function izbor**:
   ```typescript
   lossFunction: 'mse' | 'mae' | 'huber' | 'binary_crossentropy'
   ```

2. **Metrics izbor**:
   ```typescript
   metrics: string[] // ['mae', 'mse', 'rmse', 'mape']
   ```

3. **Early Stopping parametri**:
   ```typescript
   earlyStoppingPatience: number
   earlyStoppingMinDelta: number
   ```

### 🟢 PRIORITET 3 (Nice to have):

1. **Polling konfiguracija**:
   - Omogućiti korisniku da podesi polling intervale
   - Ili bar prikazati progress bar sa procenom vremena

2. **SVR napredni parametri**:
   - gamma, degree, coef0 za različite kernel tipove
   - tolerance i max iterations

---

## 📝 IMPLEMENTACIONE BELEŠKE

### Frontend izmene potrebne u:
1. `ModelConfiguration.tsx` - Dodati nova input polja
2. `TrainingApiService.ts` - Proslediti nove parametre
3. `modelParameterUtils.ts` - Validacija novih parametara

### Backend izmene potrebne u:
1. `parameter_converter.py` - Prihvatiti nove parametre sa frontenda
2. `model_trainer.py` - Koristiti prosleđene parametre umesto default vrednosti

---

## 🎯 ZAKLJUČAK

Trenutni sistem ima značajan broj hardkodovanih vrednosti koje ograničavaju fleksibilnost treniranja. Najkritičnije su:
- Validation split (20%)
- Batch size (32)
- Learning rate (0.001)
- Optimizer (samo Adam)

Ove vrednosti direktno utiču na kvalitet i brzinu treniranja modela. Preporučuje se prioritetna implementacija konfigurisanja ovih parametara kroz UI.