# Locust Load Testing Suite

Kompletan load testing sistem za Rabensteiner Engineering backend optimizovan za testiranje velikih dataset-a (do 50MB CSV fajlova).

## 📋 Testirani Endpointi

Load testovi pokrivaju **SVE** endpointe iz `my_backend/api/routes/adjustments.py`:

1. ✅ **POST /api/adjustmentsOfData/upload-chunk** - Chunked CSV upload (5MB chunks)
2. ✅ **POST /api/adjustmentsOfData/adjust-data-chunk** - Processing parameters (timestep, offset, methods)
3. ✅ **POST /api/adjustmentsOfData/adjustdata/complete** - Data processing i transformacija
4. ✅ **POST /api/adjustmentsOfData/prepare-save** - Priprema CSV-a za download
5. ✅ **GET /api/adjustmentsOfData/download/<file_id>** - Download obrađenog fajla

---

## 🚀 Instalacija

### 1. Instaliraj Python Dependencies

```bash
cd Backend_RabensteinerEng
pip3 install -r requirements-locust.txt
```

**Dependencies:**
- `locust>=2.15.0` - Load testing framework
- `python-socketio>=5.10.0` - Socket.IO client za real-time tracking
- `requests>=2.31.0` - HTTP client
- `websocket-client>=1.6.0` - WebSocket support

### 2. Provjeri Backend

Osiguraj da je backend pokrenut i dostupan:

```bash
# Test health endpoint
curl http://localhost:8080/health
# Expected: {"status": "ok"}

# Ili pokreni backend Docker container
docker-compose up -d backend
```

---

## ⚡ Dostupni Test Scenariji

### Test #1: Random Load Test (locustfile.py)

Simulira realističko ponašanje korisnika sa nasumičnim task-ovima.

**Karakteristike:**
- Weight-based task selection (upload:3, params:2, processing:1)
- Socket.IO progress tracking
- Performance validation (upload < 5s, processing < 30s)
- Konfigurabilan file size

**Pokretanje:**
```bash
# Headless mode (CLI)
python3 -m locust -f locustfile.py --host=http://localhost:8080 \
       --users 10 --spawn-rate 2 --run-time 5m --headless \
       --data-size small --csv results/random_test

# Web UI mode (interaktivno)
python3 -m locust -f locustfile.py --host=http://localhost:8080
# Otvori: http://localhost:8089
```

**Parametri:**
- `--users` - Broj concurrent korisnika (default: 10)
- `--spawn-rate` - Korisnika/sekundi (default: 2)
- `--run-time` - Trajanje testa (npr: 5m, 10m, 1h)
- `--data-size` - Veličina test podataka: `small` (2K rows), `medium` (10K), `large` (50K)

---

### Test #2: Heavy Load Test - 50MB Files (locust_heavy_50mb.py)

Testira backend performanse sa velikim fajlovima (~2.1 miliona zapisa).

**Karakteristike:**
- Sequential workflow (testira SVE endpointe u pravilnom redosljedu)
- 50MB CSV fajlovi (~2,097,152 rows)
- 5MB chunked upload
- Real-time progress logging
- Extended timeout-i za processing (120s)

**Pokretanje:**
```bash
python3 -m locust -f locust_heavy_50mb.py --host=http://localhost:8080 \
       --users 2 --spawn-rate 1 --run-time 5m --headless \
       --csv results/heavy_50mb --logfile results/heavy.log
```

**VAŽNO:**
- Koristi **maksimum 2-3 korisnika** za heavy test (veliki memory footprint)
- Svaki korisnik generira 50MB fajl u memoriji
- Potrebno ~200MB RAM po korisniku

**Očekivani Rezultati:**
- Upload: ~30-35s za 50MB (10 chunkova po 5MB)
- Processing: ~110s za 2.1M zapisa (mean aggregation)
- Total workflow: ~150-160s po korisniku

---

## 📊 Interpretacija Rezultata

### CSV Reports (results/)

Nakon testa, Locust generira 3 CSV fajla:

1. **`*_stats.csv`** - Detaljne statistike po endpointu
2. **`*_stats_history.csv`** - Vremenska serija metrika
3. **`*_failures.csv`** - Lista grešaka (ako ih ima)

**Ključne Metrike:**
- **Request Count** - Broj zahtjeva
- **Failure Count** - Broj neuspješnih zahtjeva
- **Median Response Time** - Median vremena odgovora (ms)
- **Average Response Time** - Prosječno vrijeme odgovora (ms)
- **95th Percentile** - 95% zahtjeva brži od ovog vremena
- **Requests/s** - Throughput (zahtjeva po sekundi)

### Log Files (results/*.log)

Log fajlovi sadrže detaljne poruke o svakom koraku:

```
[STEP 1] ✅ Upload COMPLETE: heavy_50mb_test.csv (49.80MB in 34.13s)
[STEP 2] ✅ Parameters set for 50MB file
[STEP 3] ✅ Processing COMPLETE: 1,258,291 records in 108.45s (11602 rec/s)
[STEP 4] ✅ File prepared for download: 20251002_105049
[STEP 5] ✅ Download complete: 20979 bytes
```

---

## 🎯 Performance Benchmarks

### Small Files (2K rows, ~50KB)

| Endpoint | Avg Time | 95th Percentile |
|----------|----------|-----------------|
| upload-chunk | 29ms | 38ms |
| adjust-data-chunk | 7ms | 11ms |
| adjustdata/complete | 104ms | 120ms |
| prepare-save | 7ms | 10ms |
| download | 8ms | 13ms |

**Expected Throughput:** 3-5 req/s sa 10 concurrent users

---

### Large Files (2.1M rows, ~50MB)

| Endpoint | Avg Time | 95th Percentile |
|----------|----------|-----------------|
| upload-chunk (5MB) | 2.5s | 25s (final chunk) |
| adjust-data-chunk | 12ms | 16ms |
| adjustdata/complete | 110s | 111s |
| prepare-save | 10ms | 10ms |
| download | 8ms | 10ms |

**Expected Throughput:** 0.2-0.3 req/s sa 2 concurrent users

**Processing Performance:**
- ~11,400 records/second (mean aggregation)
- 1.26M output records iz 2.1M input records (5-min timestep)

---

## 🔧 Backend Optimizacije (Implementovane)

Backend je optimizovan za processing velikih dataset-a:

1. ✅ **Socket.IO throttling** - Progress updates samo na svakih 10% ili 1s interval
2. ✅ **NumPy array umjesto liste** - 15-25% brže memory operacije
3. ✅ **Vectorized operations** - Eliminirani df.iterrows() bottlenecks
4. ✅ **Pre-alocirani arrays** - Brži append operacije

**Rezultati optimizacija:**
- Processing time: **115s → 110s** (~5% brže)
- Failure rate: **8.33% → 0%** (100% uspješno)
- Stabilniji processing za velike fajlove

---

## 📁 Struktura Projekta

```
Backend_RabensteinerEng/
├── locustfile.py              # Random load test (general purpose)
├── locust_heavy_50mb.py       # Heavy load test (50MB files)
├── requirements-locust.txt    # Python dependencies
├── README_LOCUST.md          # Ova dokumentacija
└── results/                   # Test rezultati (CSV, logs)
    └── .gitkeep
```

---

## 🌐 Distributed Testing

Za high-load testove sa više mašina:

### 1. Pokreni Master Node

```bash
locust -f locustfile.py --master --host=http://localhost:8080
```

**Master:** Koordinira testove, agregira rezultate, Web UI dostupan na `:8089`

### 2. Pokreni Worker Nodes (različite mašine)

```bash
# Na worker mašini #1
locust -f locustfile.py --worker --master-host=192.168.1.10

# Na worker mašini #2
locust -f locustfile.py --worker --master-host=192.168.1.10
```

**Workers:** Izvršavaju testove, šalju rezultate master-u

### 3. Prati Statistiku na Master-u

Otvori browser: `http://<master_ip>:8089`

---

## 🐛 Troubleshooting

### Backend nije dostupan

```bash
# Provjeri da li backend radi
curl http://localhost:8080/health

# Pokreni backend Docker container
cd Backend_RabensteinerEng
docker-compose up -d backend

# Provjeri logs
docker-compose logs -f backend
```

### Locust nije instaliran

```bash
pip3 install -r requirements-locust.txt
```

### Test timeout-uje sa velikim fajlovima

```bash
# Povećaj timeout u locust_heavy_50mb.py
# Linija 122: timeout=120 → timeout=180
```

### Out of Memory greška

```bash
# Smanji broj korisnika za heavy test
python3 -m locust -f locust_heavy_50mb.py \
       --users 1 --spawn-rate 1  # Umjesto 2-3 users
```

### Backend treba restart nakon izmjena

```bash
# Restartuj Docker container da učita nove izmjene
docker-compose restart backend
```

---

## 📝 Best Practices

1. **Započni sa malim testom** - Prvo pokreni sa 1-2 korisnika da provjeriš da sve radi
2. **Prati backend logs** - `docker-compose logs -f backend` tokom testa
3. **Postepeno povećavaj load** - Ne skoči odmah na 100+ korisnika
4. **Koristi headless mode** - Za produkcijske testove (manje overhead)
5. **Čuvaj rezultate** - CSV rezultati su važni za analizu trendova
6. **Test u izolaciji** - Pokreni backend lokalno, ne na production serveru
7. **Restart backend** - Nakon code izmjena, uvijek restartuj Docker container

---

## 📞 Podrška

Za pitanja ili probleme:
- Backend issues: Provjerite `my_backend/api/routes/adjustments.py`
- Test issues: Provjerite `locustfile.py` ili `locust_heavy_50mb.py`
- Performance issues: Analizirajte `results/*_stats.csv` fajlove

---

**Last Updated:** 2025-10-02
**Locust Version:** 2.34.0
**Backend Version:** Optimized for 50MB+ files
