# Koristi laganu verziju Python-a
FROM python:3.9-slim

# Postavi radni direktorijum
WORKDIR /app

# Kopiraj requirements.txt
COPY my_backend/requirements.txt .

# Kreiraj virtualno okruženje unutar kontejnera
RUN python -m venv venv

# Aktiviraj venv i instaliraj dependencije
RUN . venv/bin/activate && pip install --no-cache-dir -r requirements.txt

# Kopiraj ostatak koda u kontejner
COPY my_backend/ .

# Definiši komandu za pokretanje aplikacije
CMD [".venv/bin/python", "load_row_data.py"]
