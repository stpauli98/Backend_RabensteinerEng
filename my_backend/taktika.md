# Taktika - Status i Plan za rešavanje problema sa učitavanjem podataka

## Problem koji rešavamo
**Glavni problem:** Podaci se uspešno čuvaju u bazu podataka, ali se nisu prikazivali u form fields / input fields kada se stranica osvežavala ili komponente resetovale zbog gubitka mapiranja sesija.

## Trenutno stanje (POPRAVLJENO)

### ✅ Što je uspešno rešeno:

1.  **Backend konektivnost** - REŠENO ✅
    *   Problem: Flask-SocketIO (eventlet mode) je interferisao sa httpx bibliotekom (Supabase client)
    *   Rešenje: Promenjen `async_mode` sa `'eventlet'` na `'threading'` u `app.py:31`
    *   Dodato: `allow_unsafe_werkzeug=True` za development server

2.  **Import greška** - REŠENO ✅
    *   Problem: `request` nije bio importovan u `app.py`
    *   Rešenje: Dodato `request` u import statement: `from flask import Flask, jsonify, request`

3.  **"Illegal request line" greške** - REŠENO ✅
    *   Uzrok: eventlet je mešao HTTP protokol komunikaciju
    *   Rešenje: threading mode u SocketIO

4.  **Backend API endpoints** - RADE ✅
    *   `/health` - OK
    *   `/api/training/create-database-session` - OK
    *   `/api/training/get-session-uuid/<session_id>` - OK
    *   Svi Supabase operacije rade savršeno

5.  **Podaci se čuvaju u bazi** - FUNKCIONŠE ✅
    *   `time_info` tabela: ✅
    *   `zeitschritte` tabela: ✅
    *   `files` tabela: ✅ (tested)
    *   Session UUID kreiranje: ✅

6.  **Persistent Session ID Mapping** - REŠENO ✅
    *   Problem: Frontend koristi string session ID format (`session_1751529005379_n4hr2ww`), a Backend kreira UUID session ID (`788faa60-f0c6-4a6f-a2f0-7932246e3b8d`). Mapping između njih se čuvao samo u memoriji (`session_mapping_cache = {}`) i gubio se restartom backend-a.
    *   Rešenje: Implementirana je `session_mappings` tabela u Supabase bazi koja trajno čuva vezu između string session ID-a i UUID-a. Funkcija `create_or_get_session_uuid()` sada proverava ovu tabelu pre kreiranja novog mapiranja.
    *   Dodatno: Backend endpointi (`/session-status`, `/session/<session_id>`, `/get-all-files-metadata`, `/session/<session_id>/delete`) su modifikovani da ispravno rukuju i sa string session ID-jevima i sa UUID-jevima, koristeći `get_string_id_from_uuid` funkciju za obrnuto mapiranje kada je potrebno pristupiti lokalnim fajlovima.

7.  **Razdvajanje fajlova u Supabase Storage bucket-e** - REŠENO ✅
    *   Problem: Svi uploadovani CSV fajlovi su se čuvali u `csv-files` bucket-u, bez obzira da li su "input" ili "output".
    *   Rešenje: Funkcija `save_csv_file_content` u `supabase_client.py` je modifikovana da prihvata `file_type` parametar. Na osnovu ovog parametra, fajlovi se sada čuvaju u `csv-files` (za input) ili `aus-csv-files` (za output) bucket-e. Poziv ove funkcije u `save_session_to_supabase` je takođe ažuriran da prosleđuje `file_type`.

## Fajlovi koji su modifikovani:

1.  **`/my_backend/app.py`**
    *   Dodato: `request` import
    *   Promenjen: `async_mode='threading'` (linija 31)
    *   Dodato: `allow_unsafe_werkzeug=True` (linija 138)

2.  **`/my_backend/supabase_client.py`**
    *   Dodato: Enhanced logging za troubleshooting
    *   **Uklonjeno:** `load_existing_session_mappings()` funkcija i `session_mapping_cache` (zamenjeno trajnim mapiranjem).
    *   **Modifikovano:** `create_or_get_session_uuid()` funkcija za korišćenje `session_mappings` tabele.
    *   **Dodato:** `get_string_id_from_uuid()` funkcija za obrnuto mapiranje.
    *   **Modifikovano:** `save_csv_file_content()` funkcija da podržava različite bucket-e na osnovu `file_type`.
    *   **Modifikovano:** `save_session_to_supabase()` da prosleđuje `file_type` funkciji `save_csv_file_content()`.

3.  **`/my_backend/training.py`**
    *   Dodato: `/test-data-loading/<session_id>` endpoint za debugging.
    *   **Modifikovano:** Endpointi `/session-status/<session_id>`, `/session/<session_id>`, `/get-all-files-metadata/<session_id>`, `/session/<session_id>/delete` da koriste `get_string_id_from_uuid` za ispravno rukovanje ID-jevima sesija (string ili UUID).

4.  **`/src/lib/supabase.ts`**
    *   **Modifikovano:** `sessionsApi.deleteSession()` funkcija da dohvati `type` fajla i obriše fajlove iz odgovarajućih Supabase Storage bucket-a (`csv-files` i `aus-csv-files`).

5.  **`/src/components/ui/Training.tsx`**
    *   **Modifikovano:** `handleReset` funkcija da osigura da se `sessionsApi.deleteSession()` uvek poziva sa UUID-om, koristeći `getSessionUuid` pre brisanja.

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

## Frontend komponente koje treba proveriti:

1.  **`/src/components/ui/Training.tsx`**
    *   `uploadSessionId` state (linija 38)
    *   Session loading useEffect (linija 43-80)
    *   Props prosleđivanje komponentama (linije 327-350)
    *   `handleReset` funkcija (linija 200-225)

2.  **`/src/components/inputs/TimeInformationInput.tsx`**
    *   `useEffect` koji poziva `timeInfoApi.getTimeInfo()` (linija 192-232)
    *   Proveri da li se poziva sa ispravnim sessionId

3.  **`/src/lib/supabase.ts`**
    *   `getSessionUuid()` funkcija (linija 20)
    *   `timeInfoApi.getTimeInfo()` funkcija (linija 172)
    *   `sessionsApi.deleteSession()` funkcija (linija 300-380)

## Debug koraci:

1.  **Proveravaj browser console** za greške u frontend API pozivima
2.  **Proveravaj backend logs** (`backend.log`) za session mapping
3.  **Koristi test endpoint** za verifikaciju podataka: `/api/training/test-data-loading/<session_id>`
4.  **Proveravaj localStorage** u browser dev tools za `lastSessionId`
5.  **Proveravaj Supabase Storage bucket-e** (`csv-files` i `aus-csv-files`) nakon uploada i brisanja.

## Trenutno testirano i funkcionalno:

*   ✅ Backend server se pokreće bez grešaka
*   ✅ Supabase konekcija radi
*   ✅ Session kreiranje radi
*   ✅ Podaci se čuvaju u sve tabele (`time_info`, `zeitschritte`, `files`)
*   ✅ Podaci se mogu učitati iz baze direktno
*   ✅ Backend API endpoints odgovaraju ispravno
*   ✅ Persistentno mapiranje sesija radi (string ID <-> UUID)
*   ✅ Fajlovi se čuvaju u odgovarajuće Supabase Storage bucket-e (`csv-files` za input, `aus-csv-files` za output)

## Sledeći koraci:

1.  **Verifikovati da se frontend komponente re-renderuju** kada dobiju podatke iz baze.
2.  **Dodati bolje rukovanje greškama** u frontend API funkcije.
3.  **Rešiti problem sa brisanjem fajlova iz bucket-a** (iako je logika implementirana, potrebno je testirati nakon build-a frontenda).

## Ključne greške za izbegavanje:

*   ❌ **Ne koristiti eventlet** sa Flask-SocketIO kada se koristi httpx/Supabase.
*   ❌ **Ne oslanjati se samo na memorijski cache** za session mapping (sada rešeno).
*   ❌ **Ne zaboraviti importovati request** u Flask aplikaciji.
*   ❌ **Ne prosleđivati pogrešan tip ID-a** funkcijama koje očekuju UUID (sada rešeno).

---
*Napomena: Backend trenutno radi stabilno. Glavni fokus je na ispravnom funkcionisanju frontenda i potpunom brisanju fajlova iz Storage-a.*