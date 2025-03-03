# Koristi laganu verziju Python-a
FROM python:3.9-slim

# Postavi radni direktorijum
WORKDIR /app

# Kopiraj requirements.txt
COPY requirements.txt .

# Instaliraj zavisnosti
RUN pip install --no-cache-dir -r requirements.txt

# Kopiraj sav kod
COPY . .

# Postavi defaultnu komandu za pokretanje aplikacije (prilagodi ako je tvoj main fajl drugaciji)
CMD ["venv/bin/python", "app.py"]


