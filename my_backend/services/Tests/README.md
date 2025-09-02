# Tests Folder

Ovaj folder sadrži sve testove za backend training sistem.

## Struktura foldera

```
Tests/
├── README.md               # Ovaj fajl
├── test_simple.py         # Jednostavan test osnovnih funkcionalnosti
├── test_all_models.py     # Test svih 7 modela (Dense, CNN, LSTM, itd)
├── test_backend_standalone.py  # Kompletan test sa CSV podacima
├── data/                  # Test podaci
│   ├── input_test.csv    # Test ulazni podaci
│   └── output_test.csv   # Test izlazni podaci
└── results/              # Rezultati testova
    └── TEST_RESULTS.md   # Dokumentacija rezultata

```

## Pokretanje testova

### 1. Jednostavan test
Testira osnovne import-e i Linear model:
```bash
cd services/Tests
python3 test_simple.py
```

### 2. Test svih modela
Testira svih 7 modela (Dense, CNN, LSTM, AR-LSTM, SVR_dir, SVR_MIMO, Linear):
```bash
cd services/Tests
python3 test_all_models.py
```

### 3. Kompletan test sa CSV podacima
Testira ceo pipeline sa učitavanjem CSV fajlova:
```bash
cd services/Tests
python3 test_backend_standalone.py
```

## Rezultati

Svi testovi trenutno PROLAZE ✅:
- Linear Model ✅
- Dense Neural Network ✅
- CNN (Conv2D) ✅
- LSTM ✅
- AR-LSTM ✅
- SVR_dir ✅
- SVR_MIMO ✅

## Aktivacijske funkcije

Podržane aktivacijske funkcije:
- ReLU (i sa velikim slovima)
- relu, sigmoid, tanh, softmax
- linear, elu, selu
- softplus, softsign
- swish, gelu

## Napomene

- Testovi koriste minimalne podatke za brzo izvršavanje
- Za Dense/CNN/LSTM modele, epochs je postavljen na 1 za brzinu
- Svi testovi rade BEZ potrebe za frontendom
- Config.py pruža default vrednosti kada frontend ne pošalje parametre