# RowData Storage Options

RowData modul podržava 3 različita storage backend-a, tako da **Redis NIJE obavezan**.

## 🗄️ Storage Backend Opcije

### 1. File-Based Storage (Preporučeno za većinu slučajeva)
- **Prednosti**: 
  - Nema dodatnih zavisnosti
  - Podatci perzistentni između restarta
  - Lako backup-ovanje (samo kopiraj folder)
  - Jednostavno debugovanje
- **Nedostaci**: 
  - Malo sporije od Redis-a
  - Ne skalira se na više servera

### 2. Redis Storage
- **Prednosti**:
  - Najbrže performanse
  - Skalira se na više servera
  - Profesionalno rešenje
- **Nedostaci**:
  - Zahteva Redis instalaciju
  - Dodatna kompleksnost

### 3. In-Memory Storage
- **Prednosti**:
  - Najjednostavnije
  - Bez ikakvih zavisnosti
- **Nedostaci**:
  - Podatci se gube pri restartu
  - Samo za development/testiranje

## 🚀 Kako koristiti File-Based Storage

### Opcija 1: Automatska detekcija (najlakše)
Ne morate ništa menjati! Sistem će automatski koristiti file-based storage ako Redis nije dostupan.

### Opcija 2: Eksplicitno forsiraj file storage
Postavite environment varijablu:
```bash
export ROWDATA_STORAGE_BACKEND=file
```

Ili u `.env` fajlu:
```
ROWDATA_STORAGE_BACKEND=file
```

### Opcija 3: Custom putanja za skladištenje
```bash
export ROWDATA_FILE_STORAGE_PATH=/path/to/your/storage
```

Default putanja je `/tmp/row_data_uploads/`

## 📁 File Storage Struktura

```
/tmp/row_data_uploads/
├── metadata/
│   ├── upload_[id].json          # Metadata za svaki upload
│   ├── active_uploads.json       # Lista aktivnih upload-ova
│   └── chunks_[id]_received.json # Lista primljenih chunk-ova
├── chunks/
│   └── [upload_id]/
│       └── chunk_00001.part      # Chunk fajlovi
├── results/
│   └── result_[id].json          # Rezultati procesiranja
└── locks/
    └── lock_[id]                 # Lock fajlovi
```

## 🔄 Migracija između Storage Backend-ova

Ako želite da promenite storage backend:

1. **Iz In-Memory u File-Based**: Samo restartujte sa `ROWDATA_STORAGE_BACKEND=file`
2. **Iz File-Based u Redis**: Instalirajte Redis i restartujte
3. **Iz Redis u File-Based**: Postavite `ROWDATA_STORAGE_BACKEND=file`

⚠️ **Napomena**: Postojeći podatci se neće automatski migrirati između backend-ova.

## 🛠️ Konfiguracija

### Environment varijable
```bash
# Storage backend izbor
ROWDATA_STORAGE_BACKEND=file  # 'auto', 'redis', 'file', 'memory'

# File storage putanja
ROWDATA_FILE_STORAGE_PATH=/var/data/rowdata

# Upload expiry vreme (sekunde)
ROWDATA_UPLOAD_EXPIRY_TIME=1800  # 30 minuta
```

### Python konfiguracija
```python
# my_backend/RowData/config/settings.py
STORAGE_BACKEND = 'file'
FILE_STORAGE_CONFIG = {
    'base_path': '/var/data/rowdata',
    'cleanup_interval': 3600,
    'file_permissions': 0o600,
}
```

## 📊 Provera koji Storage koristite

```bash
curl http://localhost:5000/api/loadRowData/storage-info
```

Response:
```json
{
  "success": true,
  "storage": {
    "backend": "File",
    "persistent": true,
    "distributed": false,
    "description": "File-based persistent storage"
  }
}
```

## 🚨 Troubleshooting

### Problem: "Permission denied" greške
**Rešenje**: Proverite da li aplikacija ima write permisije za storage folder:
```bash
chmod -R 755 /tmp/row_data_uploads
chown -R your-user:your-group /tmp/row_data_uploads
```

### Problem: Disk pun
**Rešenje**: Cleanup endpoint briše stare upload-ove:
```bash
curl -X POST http://localhost:5000/api/loadRowData/cleanup \
  -H "Authorization: Bearer <token>"
```

### Problem: Sporije performanse
**Rešenje**: 
- Koristite SSD disk
- Prebacite storage na brži disk
- Razmotrite prelazak na Redis za high-traffic

## 📈 Performanse

Tipične performanse na prosečnom serveru:

| Operacija | File Storage | Redis | In-Memory |
|-----------|-------------|-------|-----------|
| Store chunk | ~5ms | ~2ms | ~0.5ms |
| Get metadata | ~3ms | ~1ms | ~0.1ms |
| List chunks | ~10ms | ~3ms | ~1ms |

Za većinu aplikacija, file-based storage je sasvim dovoljan!