# Taktika - Status i Plan za rešavanje problema sa učitavanjem podataka

## Problem koji rešavamo
**Glavni problem:** Podaci se uspešno čuvaju u bazu podataka, ali se ne prikazuju u form fields / input fields kada se stranica osvežava ili komponente resetuju.

## Trenutno stanje (POPRAVLJENO)

### ✅ Što je uspešno rešeno:

1. **Backend konektivnost** - REŠENO ✅
   - Problem: Flask-SocketIO (eventlet mode) je interferisao sa httpx bibliotekom (Supabase client)
   - Rešenje: Promenjen `async_mode` sa `'eventlet'` na `'threading'` u `app.py:31`
   - Dodato: `allow_unsafe_werkzeug=True` za development server

2. **Import greška** - REŠENO ✅ 
   - Problem: `request` nije bio importovan u `app.py`
   - Rešenje: Dodato `request` u import statement: `from flask import Flask, jsonify, request`

3. **"Illegal request line" greške** - REŠENO ✅
   - Uzrok: eventlet je mešao HTTP protokol komunikaciju
   - Rešenje: threading mode u SocketIO

4. **Backend API endpoints** - RADE ✅
   - `/health` - OK
   - `/api/training/create-database-session` - OK  
   - `/api/training/get-session-uuid/<session_id>` - OK
   - Svi Supabase operacije rade savršeno

5. **Podaci se čuvaju u bazi** - FUNKCIONŠE ✅
   - time_info tabela: ✅ 
   - zeitschritte tabela: ✅
   - files tabela: ✅ (tested)
   - Session UUID kreiranje: ✅

## Trenutni problem koji ostaje - Session ID Mapping

### ❌ Problem koji još uvek postoji:

**Session ID Mapping problem:**
- Frontend koristi string session ID format: `session_1751529005379_n4hr2ww` 
- Backend kreira UUID session ID: `788faa60-f0c6-4a6f-a2f0-7932246e3b8d`
- Mapping između njih se čuva samo u memoriji (`session_mapping_cache = {}`)
- Kada se backend restartuje, mapping se gubi
- Rezultat: Frontend ne može da učita postojeće podatke jer se kreira novi UUID umesto korišćenja postojećeg

### Test podaci u bazi:
```json
session_uuid: "788faa60-f0c6-4a6f-a2f0-7932246e3b8d"
time_info: {
  "jahr": true,
  "woche": true, 
  "category_data": {
    "jahr": {"datenform": "Zeit Horizont", "skalierung": "ja", ...},
    "woche": {"datenform": "Aktuelle zeit", "skalierung": "nein", ...}
  }
}
zeitschritte: {
  "eingabe": "10min",
  "ausgabe": "1h", 
  "zeitschrittweite": "5min"
}
```

## Fajlovi koji su modifikovani:

1. **`/my_backend/app.py`**
   - Dodato: `request` import
   - Promenjen: `async_mode='threading'` (linija 31)
   - Dodato: `allow_unsafe_werkzeug=True` (linija 138)

2. **`/my_backend/supabase_client.py`**
   - Dodato: Enhanced logging za troubleshooting
   - Dodano: `load_existing_session_mappings()` funkcija
   - Dodato: Temporary mapping za test session

3. **`/my_backend/training.py`**
   - Dodato: `/test-data-loading/<session_id>` endpoint za debugging

## Kako testirati da backend radi:

```bash
# 1. Start backend
cd /my_backend
python app.py

# 2. Test health
curl http://127.0.0.1:8080/health
# Should return: {"status":"ok"}

# 3. Test session creation
curl -X POST http://127.0.0.1:8080/api/training/create-database-session \
  -H "Content-Type: application/json" \
  -d '{"sessionId": "test_session"}'

# 4. Test data loading 
curl http://127.0.0.1:8080/api/training/test-data-loading/test_form_data_session
# Should return JSON with existing time_info and zeitschritte data
```

## Plan za rešavanje problema sa Session Mapping:

### Opcija 1: Persistent Session Mapping (RECOMMENDED)
Kreirati `session_mappings` tabelu u Supabase:
```sql
CREATE TABLE session_mappings (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  string_session_id TEXT UNIQUE NOT NULL,
  uuid_session_id UUID REFERENCES sessions(id),
  created_at TIMESTAMP DEFAULT NOW()
);
```

Modifikovati `create_or_get_session_uuid()` da:
1. Proverava session_mappings tabelu prvo
2. Koristi postojeći mapping ako postoji
3. Kreira novi mapping samo ako ne postoji

### Opcija 2: Frontend UUID Strategy  
Modifikovati frontend da direktno koristi UUID format:
- Umesto `session_1751529005379_n4hr2ww` koristiti `788faa60-f0c6-4a6f-a2f0-7932246e3b8d`
- Eliminisati potrebu za mapiranje

### Opcija 3: Hybrid Approach
- Koristiti URL parametar `?session=<uuid>` za direktne UUID linkove
- Zadržati localStorage mapping za kontinuitet sessiona

## Frontend komponente koje treba proveriti:

1. **`/src/components/ui/Training.tsx`**
   - `uploadSessionId` state (linija 38)
   - Session loading useEffect (linija 43-80)
   - Props prosleđivanje komponentama (linije 327-350)

2. **`/src/components/inputs/TimeInformationInput.tsx`** 
   - `useEffect` koji poziva `timeInfoApi.getTimeInfo()` (linija 192-232)
   - Proveri da li se poziva sa ispravnim sessionId

3. **`/src/lib/supabase.ts`**
   - `getSessionUuid()` funkcija (linija 20)
   - `timeInfoApi.getTimeInfo()` funkcija (linija 172)

## Debug koraci:

1. **Proveravaj browser console** za greške u frontend API pozivima
2. **Proveravaj backend logs** (`backend.log`) za session mapping 
3. **Koristi test endpoint** za verifikaciju podataka: `/api/training/test-data-loading/<session_id>`
4. **Proveravaj localStorage** u browser dev tools za `lastSessionId`

## Trenutno testirano i funkcionalno:

- ✅ Backend server se pokreće bez grešaka
- ✅ Supabase konеkcija radi
- ✅ Session kreiranje radi  
- ✅ Podaci se čuvaju u sve tabele (time_info, zeitschritte, files)
- ✅ Podaci se mogu učitati iz baze direktno
- ✅ Backend API endpoints odgovaraju ispravno

## Sledeći koraci:

1. **Implementirati persistent session mapping** (Opcija 1)
2. **Testirati frontend komponente** sa konzistentnim session ID
3. **Verifikovati da se komponente re-renderuju** kada dobiju podatke iz baze
4. **Dodati better error handling** u frontend API funkcije

## Ključne greške za izbegavanje:

- ❌ **Ne koristiti eventlet** sa Flask-SocketIO kada se koristi httpx/Supabase
- ❌ **Ne oslanjati se samo na memorijski cache** za session mapping
- ❌ **Ne zaboraviti importovati request** u Flask aplikaciji
- ⚠️ **Session mapping se gubi pri restartu** - mora biti persistent

## Test session ID za debugging:
- String ID: `test_form_data_session` 
- UUID: `788faa60-f0c6-4a6f-a2f0-7932246e3b8d`
- Ima postojeće podatke u bazi za testiranje

---
*Napomena: Backend trenutno radi stabilno. Glavni fokus treba biti na persistent session mapping i frontend komponente koje učitavaju podatke.*