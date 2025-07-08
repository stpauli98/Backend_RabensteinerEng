# Stage 1: Builder - instalira sve Python ovisnosti
FROM python:3.9-slim as builder

# Instaliraj sistemske ovisnosti potrebne za izgradnju (npr. curl, build-essential za neke pakete)
# build-essential je koristan ako neki Python paketi zahtijevaju kompilaciju C/C++ koda
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Kopiraj samo requirements.txt kako bi se iskoristio Docker cache
COPY requirements.txt .

# Instaliraj Python ovisnosti. --no-cache-dir sprječava pip da kešira pakete unutar ove faze,
# što je dobro jer ne želimo da se taj cache kopira u finalnu sliku.
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production - kreira laganu sliku za pokretanje aplikacije
FROM python:3.9-slim

# Instaliraj samo sistemske ovisnosti potrebne za pokretanje aplikacije (ako ih ima)
# U ovom slučaju, curl je možda potreban za HTTP zahtjeve.
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Kopiraj instalirane Python pakete iz "builder" faze
# Ovo je ključno za ubrzanje, jer se paketi ne instaliraju ponovno
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Kopiraj ostatak koda aplikacije
COPY . .

# Izloži port na kojem aplikacija sluša
EXPOSE 8080

# Definiraj naredbu za pokretanje aplikacije
# S obzirom na to da koristite Flask-SocketIO i eventlet, preporučuje se gunicorn s eventlet workerom.
CMD ["gunicorn", "-k", "eventlet", "-w", "1", "app:app"]