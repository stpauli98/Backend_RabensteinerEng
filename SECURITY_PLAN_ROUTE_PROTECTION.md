# DETALJAN PLAN ZAŠTITE BACKEND RUTA

## 📋 PREGLED

**Status:** Kritičan sigurnosni problem - 29 nezaštićenih ruta omogućavaju neautorizovan pristup podacima korisnika

**Prioritet:** 🔴 VISOK - Implementacija potrebna odmah

---

## 🔐 KATEGORIJE ZAŠTITE

### Obavezni Dekoratori:

1. **@require_auth** - Provera da li je korisnik prijavljen (Firebase token)
2. **@require_subscription** - Provera da li korisnik ima aktivnu pretplatu
3. **@check_processing_limit** - Provera quote-a za procesiranje podataka
4. **@check_training_limit** - Provera quote-a za treniranje modela
5. **@check_upload_limit** - Provera quote-a za upload fajlova

---

## 📁 CLOUD.PY - 6 RUTA (0% zaštićeno)

| Linija | Ruta | Metod | Šta radi | Dekoratori | Tracking | Prioritet | Razlog |
|--------|------|-------|----------|------------|----------|-----------|---------|
| 217 | `/upload-chunk` | POST | Prima file chunks za cloud procesiranje | `@require_auth`<br>`@require_subscription`<br>`@check_upload_limit` | `increment_upload_count()` | 🔴 KRITIČNO | Bilo ko može upload-ovati fajlove bez limite |
| 290 | `/complete` | POST | Spaja chunks i procesira podatke | `@require_auth`<br>`@require_subscription`<br>`@check_processing_limit` | `increment_processing_count()` | 🔴 KRITIČNO | Bilo ko može pokrenuti procesiranje |
| 700 | `/clouddata` | POST | Procesira temperature i load podatke | `@require_auth`<br>`@require_subscription`<br>`@check_processing_limit` | `increment_processing_count()` | 🔴 KRITIČNO | Bilo ko može procesirati podatke |
| 770 | `/interpolate-chunked` | POST | Interpolacija podataka iz chunk-ova | `@require_auth`<br>`@require_subscription`<br>`@check_processing_limit` | `increment_processing_count()` | 🔴 KRITIČNO | Bilo ko može pokrenuti interpolaciju |
| 1055 | `/prepare-save` | POST | Priprema CSV fajl za download | `@require_auth`<br>`@require_subscription` | - | 🟡 VISOKO | Bilo ko može pripremiti tuđe podatke |
| 1123 | `/download/<file_id>` | GET | Preuzima pripremljeni fajl | `@require_auth`<br>`@require_subscription` | - | 🔴 KRITIČNO | Bilo ko može preuzeti tuđe fajlove ako zna ID |

### Cloud.py Akcije:
```python
# Potrebno dodati na vrh fajla:
from flask import request, jsonify, send_file, Blueprint, Response, g
from middleware.auth import require_auth
from middleware.subscription import require_subscription, check_upload_limit, check_processing_limit
from utils.usage_tracking import increment_upload_count, increment_processing_count
```

---

## 🎓 TRAINING.PY - 23 RUTE (17% zaštićeno - samo 4 od 27)

### 🔴 KRITIČNE RUTE (Direktno menjaju/brišu podatke):

| Linija | Ruta | Metod | Šta radi | Dekoratori | Tracking | Prioritet | Razlog |
|--------|------|-------|----------|------------|----------|-----------|---------|
| **1558** | `/delete-all-sessions` | POST | **BRIŠE SVE SESIJE SVIH KORISNIKA** | `@require_auth`<br>`@require_subscription`<br>**+ Admin check!** | - | 🚨 **EKSTREMNO** | **Bilo ko može obrisati CELU BAZU PODATAKA!** |
| 902 | `/session/<id>/delete` | POST | Briše sesiju treninga | `@require_auth`<br>`@require_subscription` | - | 🔴 KRITIČNO | Bilo ko može obrisati tuđe sesije |
| 1122 | `/csv-files/<file_id>` | DELETE | Briše CSV fajl | `@require_auth`<br>`@require_subscription` | - | 🔴 KRITIČNO | Bilo ko može obrisati tuđe fajlove |
| 1090 | `/csv-files/<file_id>` | PUT | Izmena CSV fajla | `@require_auth`<br>`@require_subscription` | - | 🔴 KRITIČNO | Bilo ko može izmeniti tuđe fajlove |

### 🟡 VISOK PRIORITET (Menjaju podatke korisnika):

| Linija | Ruta | Metod | Šta radi | Dekoratori | Tracking | Prioritet |
|--------|------|-------|----------|------------|----------|-----------|
| 526 | `/finalize-session` | POST | Finalizuje sesiju nakon upload-a | `@require_auth`<br>`@require_subscription`<br>`@check_processing_limit` | `increment_processing_count()` | 🟡 VISOKO |
| 740 | `/init-session` | POST | Kreira novu sesiju treninga | `@require_auth`<br>`@require_subscription` | - | 🟡 VISOKO |
| 768 | `/save-time-info` | POST | Čuva vremenske informacije | `@require_auth`<br>`@require_subscription` | - | 🟡 VISOKO |
| 804 | `/create-database-session` | POST | Kreira sesiju u bazi | `@require_auth`<br>`@require_subscription` | - | 🟡 VISOKO |
| 866 | `/save-zeitschritte` | POST | Čuva vremenski korak | `@require_auth`<br>`@require_subscription` | - | 🟡 VISOKO |
| 1762 | `/save-evaluation-tables/<id>` | POST | Čuva evaluation tabele | `@require_auth`<br>`@require_subscription` | - | 🟡 VISOKO |
| 1298 | `/generate-plot` | POST | Generiše grafikone | `@require_auth`<br>`@require_subscription`<br>`@check_processing_limit` | `increment_processing_count()` | 🟡 VISOKO |

### 🟢 SREDNJI PRIORITET (Čitaju privatne podatke):

| Linija | Ruta | Metod | Šta radi | Dekoratori | Prioritet |
|--------|------|-------|----------|------------|-----------|
| 558 | `/list-sessions` | GET | Lista svih sesija | `@require_auth`<br>`@require_subscription` | 🟢 SREDNJE |
| 584 | `/session/<session_id>` | GET | Detalji sesije | `@require_auth`<br>`@require_subscription` | 🟢 SREDNJE |
| 645 | `/session/<id>/database` | GET | Sesija iz baze | `@require_auth`<br>`@require_subscription` | 🟢 SREDNJE |
| 706 | `/session-status/<id>` | GET | Status sesije | `@require_auth`<br>`@require_subscription` | 🟢 SREDNJE |
| 831 | `/get-session-uuid/<id>` | GET | UUID sesije | `@require_auth`<br>`@require_subscription` | 🟢 SREDNJE |
| 932 | `/get-zeitschritte/<id>` | GET | Vremenski korak | `@require_auth`<br>`@require_subscription` | 🟢 SREDNJE |
| 954 | `/get-time-info/<id>` | GET | Vremenske info | `@require_auth`<br>`@require_subscription` | 🟢 SREDNJE |
| 984 | `/csv-files/<session_id>` | GET | Lista CSV fajlova | `@require_auth`<br>`@require_subscription` | 🟢 SREDNJE |
| 1150 | `/results/<session_id>` | GET | Rezultati treninga | `@require_auth`<br>`@require_subscription` | 🟢 SREDNJE |
| 1228 | `/get-training-results/<id>` | GET | Detalji rezultata | `@require_auth`<br>`@require_subscription` | 🟢 SREDNJE |
| 1237 | `/plot-variables/<id>` | GET | Varijable za plot | `@require_auth`<br>`@require_subscription` | 🟢 SREDNJE |
| 1266 | `/visualizations/<id>` | GET | Vizualizacije | `@require_auth`<br>`@require_subscription` | 🟢 SREDNJE |
| 1624 | `/evaluation-tables/<id>` | GET | Evaluation tabele | `@require_auth`<br>`@require_subscription` | 🟢 SREDNJE |

### ✅ VEĆ ZAŠTIĆENO (4 rute):

| Linija | Ruta | Status |
|--------|------|--------|
| 382 | `/upload-chunk` | ✅ `@require_auth` `@require_subscription` `@check_processing_limit` |
| 1019 | `/csv-files` (POST) | ✅ `@require_auth` `@require_subscription` `@check_processing_limit` |
| 1436 | `/generate-datasets/<id>` | ✅ `@require_auth` `@require_subscription` `@check_processing_limit` |
| 1495 | `/train-models/<id>` | ✅ `@require_auth` `@require_subscription` `@check_training_limit` |

---

## 🚨 SPECIJALNI SLUČAJEVI

### `/delete-all-sessions` - EKSTREMNO OPASNA RUTA!

**Problem:** Ova ruta BRIŠE SVE SESIJE SVIH KORISNIKA iz baze i lokalnog storage-a!

**Trenutno stanje:** Bilo ko može poslati POST request i obrisati kompletan sistem!

**Rešenje:**
```python
@bp.route('/delete-all-sessions', methods=['POST'])
@require_auth
@require_subscription
def delete_all_sessions_endpoint():
    # DODATNA PROVERA: Samo admin može pozvati ovu rutu!
    if not is_admin_user(g.user_id):
        return jsonify({
            'success': False,
            'error': 'Unauthorized: Admin access required'
        }), 403

    # ... existing code ...
```

**Ili bolje:**
- Premestiti ovu rutu u poseban admin blueprint
- Dodati API key ili poseban admin token
- Možda čak i **sasvim ukloniti** iz production-a (koristiti samo CLI skriptu)

---

## 📊 TRACKING POTREBAN

Sledeće rute procesiraju podatke i trebaju usage tracking:

### Cloud.py:
```python
# upload-chunk (line 217)
increment_upload_count(g.user_id)

# complete (line 290)
increment_processing_count(g.user_id)

# clouddata (line 700)
increment_processing_count(g.user_id)

# interpolate-chunked (line 770)
increment_processing_count(g.user_id)
```

### Training.py:
```python
# finalize-session (line 526)
increment_processing_count(g.user_id)

# generate-plot (line 1298)
increment_processing_count(g.user_id)
```

---

## 🎯 PLAN IMPLEMENTACIJE

### Faza 1: EKSTREMNO HITNO (Ispravi odmah!)
1. ✅ **adjustments.py** - SVE RUTE (ZAVRŠENO)
2. 🔴 **training.py line 1558** - `/delete-all-sessions` (EKSTREMNO OPASNO!)
3. 🔴 **training.py DELETE/PUT rute** - Brisanje i izmena podataka

### Faza 2: KRITIČNO (Danas!)
1. 🔴 **cloud.py SVE RUTE** - Upload i procesiranje bez zaštite
2. 🔴 **training.py POST rute** - Kreiranje i menjanje podataka

### Faza 3: VISOKO (Uskoro!)
1. 🟡 **training.py GET rute** - Čitanje privatnih podataka

### Faza 4: DODATNO
1. Dodati ownership proveru - korisnik može videti samo SVOJE sesije/fajlove
2. Rate limiting za sve rute
3. Audit log za sve izmene podataka

---

## 📝 OWNERSHIP PROVERA (BUDUĆE)

Osim dekoratora, treba dodati provere vlasništva:

```python
@bp.route('/session/<session_id>', methods=['GET'])
@require_auth
@require_subscription
def get_session_endpoint(session_id):
    # Proveri da li sesija pripada ovom korisniku
    session = get_session_from_database(session_id)
    if session['user_id'] != g.user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    # ... rest of code ...
```

---

## ✅ VEĆ POPRAVLJENO

| Fajl | Status | Detalji |
|------|--------|---------|
| `adjustments.py` | ✅ 100% | Sve rute zaštićene sa auth + subscription + processing limit |
| `first_processing.py` | ✅ 100% | Zaštićeno + tracking dodat |
| `data_processing.py` | ✅ 100% | Zaštićeno + tracking dodat |
| `load_data.py` | ✅ 100% | Zaštićeno + upload tracking |

---

## 🔢 STATISTIKA

```
Ukupno backend ruta: ~35
Nezaštićeno: 29 (83%)
Zaštićeno: 6 (17%)

cloud.py: 0/6 zaštićeno (0%)
training.py: 4/27 zaštićeno (15%)
adjustments.py: ✅ 6/6 zaštićeno (100%)
first_processing.py: ✅ 1/1 zaštićeno (100%)
data_processing.py: ✅ 1/1 zaštićeno (100%)
load_data.py: ✅ 1/1 zaštićeno (100%)
```

---

## ⚠️ RIZICI AKO SE NE POPRAVI

1. **Gubitak podataka** - Bilo ko može pozvati `/delete-all-sessions`
2. **Krađa podataka** - Bilo ko može preuzeti tuđe fajlove i rezultate
3. **Zloupotreba resursa** - Neograničen upload i procesiranje bez quota
4. **GDPR kršenje** - Neautorizovan pristup ličnim podacima korisnika
5. **Financial loss** - Korisnici ne plaćaju, ali koriste premium funkcije

---

## 🎬 SLEDEĆI KORACI

**Pitanje za korisnika:**

1. Da li da počnem sa **Faza 1 (EKSTREMNO HITNO)** - delete-all-sessions + DELETE/PUT rute?
2. Da li da odmah popravim **SVE odjednom** (cloud.py + training.py)?
3. Da li želiš da `/delete-all-sessions` POTPUNO UKLONIM ili samo zaštitim sa admin checkom?
4. Da li već postoji sistem za admin proveru ili treba da napravim novi?

---

**Kreirao:** Claude Code
**Datum:** 2025-10-27
**Status:** Čeka odobrenje za implementaciju
