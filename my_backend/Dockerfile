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

# Postavi environment varijable
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Izloži port koji Cloud Run očekuje
EXPOSE 8080

# Pokreni Flask aplikaciju na portu 8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app


