# Backend Rabensteiner Engineering

Backend aplikacija za Rabensteiner Engineering projekt. Ova aplikacija koristi Flask, TensorFlow i druge pakete za obradu podataka i pružanje API-ja.

## Tehnologije

- Python 3.11
- Flask
- TensorFlow
- Pandas
- NumPy

## Postavljanje razvoja

1. Klonirajte repozitorij
2. Stvorite virtualno okruženje: `python -m venv venv_tf`
3. Aktivirajte virtualno okruženje: `source venv_tf/bin/activate`
4. Instalirajte ovisnosti: `pip install -r requirements.txt`

## API Endpoints

- `/api/cloud/generate-chart` - Generiranje grafikona na temelju podataka
- `/api/cloud/upload-chunk` - Učitavanje dijelova datoteka
- `/api/adjustmentsOfData/complete` - Završetak procesa učitavanja
- `/api/cloud/prepare-save` - Priprema i spremanje CSV podataka

## Pokretanje aplikacije

```bash
python app.py
```

## Napomena

Ova aplikacija zahtijeva Python 3.11 zbog kompatibilnosti s TensorFlow-om.