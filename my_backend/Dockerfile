FROM python:3.9-slim

WORKDIR /app

# Kopiraj requirements.txt
COPY my_backend/requirements.txt .


# Instaliraj dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Kopiraj sve fajlove iz trenutnog direktorijuma u container
COPY . .

# Expose port 5001
EXPOSE 5001

# Pokreni aplikaciju
CMD ["python", "firstProcessing.py"]
