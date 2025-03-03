# Koristi laganu verziju Python-a
FROM python:3.9-slim

# Postavi radni direktorijum
WORKDIR /app

# Kopiraj requirements.txt pre nego sto kopiras sve fajlove
COPY my_backend/requirements.txt •

# Kreiraj i aktiviraj virtualno okruzenje u kontejneru
RUN python -m venv venv && . venv/bin/activate && pip install --no-cache-dir -r requirements.txt

# Sada kopiraj sav kod iz backend foldera, ali 'venv/' nece biti kopiran jer je u -dockeri Accept
COPY my_backend/ .

# Postavi defaultnu komandu za pokretanje aplikacije (prilagodi ako je tvoj main fajl drugaciji)
CMD ["venv/bin/python", "app.py"]


