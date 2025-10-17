# Training API Endpoints - Kompletan Spisak

**Datum:** 2025-10-17
**Verzija:** 1.0 - Finalna Verifikacija
**Ukupno jedinstvenih endpointa:** 37

---

## 📑 Sadržaj

1. [Training Core Operations](#1-training-core-operations)
2. [Model Management](#2-model-management)
3. [Evaluation & Results](#3-evaluation--results)
4. [Visualization](#4-visualization)
5. [Plotting Interface](#5-plotting-interface)
6. [CSV File Management](#6-csv-file-management)
7. [Time Information](#7-time-information)
8. [Zeitschritte (Time Steps)](#8-zeitschritte-time-steps)
9. [Session Management](#9-session-management)
10. [Scalers](#10-scalers)
11. [Training Status/Polling](#11-training-statuspolling)
12. [Upload/Chunked Upload](#12-uploadchunked-upload)
13. [Utility Endpoints](#13-utility-endpoints)
14. [Statistika i Analiza](#statistika-i-analiza)

---

## 1. Training Core Operations

**Broj endpointa:** 7

| # | Endpoint | Metoda | Opis | Fajl | Linija |
|---|----------|--------|------|------|--------|
| 1 | `/api/training/generate-datasets/{sessionId}` | POST | Generisanje dataseta za treniranje | [trainingApiService.ts](src/features/training/services/trainingApiService.ts#L156) | 156 |
| 2 | `/api/training/train-models/{sessionId}` | POST | Pokretanje treniranja modela | [trainingApiService.ts](src/features/training/services/trainingApiService.ts#L216) | 216 |
| 3 | `/api/training/start-complete-pipeline/{sessionId}` | POST | Pokretanje kompletne pipeline | [trainingApiService.ts](src/features/training/services/trainingApiService.ts#L263) | 263 |
| 4 | `/api/training/get-training-status/{sessionId}` | GET | Dohvatanje statusa treniranja | [trainingApiService.ts](src/features/training/services/trainingApiService.ts#L273) | 273 |
| 5 | `/api/training/pipeline-overview/{sessionId}` | GET | Pregled pipeline procesa | [trainingApiService.ts](src/features/training/services/trainingApiService.ts#L280) | 280 |
| 6 | `/api/training/results/{sessionId}` | GET | Rezultati treniranja | [trainingApiService.ts](src/features/training/services/trainingApiService.ts#L287) | 287 |
| 7 | `/api/training/comprehensive-evaluation/{sessionId}` | GET | Kompletna evaluacija modela | [trainingApiService.ts](src/features/training/services/trainingApiService.ts#L294) | 294 |

### Dodatne Reference:
- **constants.ts** definiše: `TRAIN_MODELS`, `GET_TRAINING_STATUS`, `GET_TRAINING_RESULTS`
- **useViolinPlots.ts:22** koristi `GET_TRAINING_RESULTS` endpoint
- **useViolinPlots.ts:40** koristi `TRAINING_RESULTS` endpoint
- **useSessionManagement.ts:146** koristi `/results/{sessionId}`

---

## 2. Model Management

**Broj endpointa:** 5

| # | Endpoint | Metoda | Opis | Fajl | Linija |
|---|----------|--------|------|------|--------|
| 8 | `/api/training/save-model/{sessionId}` | POST | Čuvanje treniranog modela | [trainingApiService.ts](src/features/training/services/trainingApiService.ts#L337) | 337 |
| 9 | `/api/training/list-models/{sessionId}` | GET | Lista svih modela za sesiju | [trainingApiService.ts](src/features/training/services/trainingApiService.ts#L385) | 385 |
| 10 | `/api/training/download-model/{sessionId}` | GET | Download modela (generički) | [trainingApiService.ts](src/features/training/services/trainingApiService.ts#L412-413) | 412-413 |
| 11 | `/api/training/list-models-database/{sessionId}` | GET | Lista modela iz Supabase baze | [trainingApiService.ts](src/features/training/services/trainingApiService.ts#L471) | 471 |
| 12 | `/api/training/download-model-h5/{sessionId}` | GET | Download modela u H5 formatu | [trainingApiService.ts](src/features/training/services/trainingApiService.ts#L497) | 497 |

### Korištenje:
- **ModelDownloadModal.tsx:62** - `listModelsDatabase()`
- **ModelDownloadModal.tsx:84,100** - `downloadModelH5()`

---

## 3. Evaluation & Results

**Broj endpointa:** 1 (2 reference)

| # | Endpoint | Metoda | Opis | Fajl | Linija |
|---|----------|--------|------|------|--------|
| 13 | `/api/training/evaluation-tables/{sessionId}` | GET | Dohvatanje evaluacionih tabela (df_eval, df_eval_ts) | [EvaluationTables.tsx](src/features/training/components/EvaluationTables.tsx#L67) | 67 |

### Dodatne Reference:
- [shared/EvaluationTables.tsx:51](src/shared/components/tables/EvaluationTables.tsx#L51)

---

## 4. Visualization

**Broj endpointa:** 1 (2 reference)

| # | Endpoint | Metoda | Opis | Fajl | Linija |
|---|----------|--------|------|------|--------|
| 14 | `/api/training/visualizations/{sessionId}` | GET | Dohvatanje vizualizacija (violin plots) | [trainingApiService.ts](src/features/training/services/trainingApiService.ts#L300) | 300 |

### Dodatne Reference:
- [VisualizationViolinDiagramContainer.tsx:39](src/features/training/components/VisualizationViolinDiagramContainer.tsx#L39)

---

## 5. Plotting Interface

**Broj endpointa:** 2

| # | Endpoint | Metoda | Opis | Fajl | Linija |
|---|----------|--------|------|------|--------|
| 15 | `/api/training/plot-variables/{sessionId}` | GET | Lista dostupnih varijabli za plotting | [PlottingInterface.tsx](src/features/training/components/PlottingInterface.tsx#L103) | 103 |
| 16 | `/api/training/generate-plot` | POST | Generisanje custom plotova | [PlottingInterface.tsx](src/features/training/components/PlottingInterface.tsx#L239) | 239 |

---

## 6. CSV File Management

**Broj endpointa:** 4 (5 sa query parametrima)

| # | Endpoint | Metoda | Opis | Fajl | Linija |
|---|----------|--------|------|------|--------|
| 17 | `/api/training/csv-files/{sessionId}` | GET | Dohvatanje svih CSV fajlova za sesiju | [backendApi.ts](src/api/backendApi.ts#L10-11) | 10-11 |
| 17a | `/api/training/csv-files/{sessionId}?type={input\|output}` | GET | Filtriranje CSV fajlova po tipu | [backendApi.ts](src/api/backendApi.ts#L10) | 10 |
| 18 | `/api/training/csv-files` | POST | Upload/kreiranje novog CSV fajla | [backendApi.ts](src/api/backendApi.ts#L44) | 44 |
| 18a | `/api/training/csv-files` | POST | Kreiranje samo metadata bez fajla | [backendApi.ts](src/api/backendApi.ts#L58) | 58 |
| 19 | `/api/training/csv-files/{fileId}` | PUT | Ažuriranje CSV metadata | [backendApi.ts](src/api/backendApi.ts#L86) | 86 |
| 20 | `/api/training/csv-files/{fileId}` | DELETE | Brisanje CSV fajla | [backendApi.ts](src/api/backendApi.ts#L110) | 110 |

### Korištenje:
- **CSVFileUploader.tsx:72** - `getCSVFiles()`
- **CSVFileUploader.tsx:162** - `updateCSVFile()` (auto-save)
- **CSVFileUploader.tsx:453** - `deleteCSVFile()`
- **CSVFileUploader.tsx:538** - `createCSVFile()`

---

## 7. Time Information

**Broj endpointa:** 2

| # | Endpoint | Metoda | Opis | Fajl | Linija |
|---|----------|--------|------|------|--------|
| 21 | `/api/training/get-time-info/{sessionId}` | GET | Dohvatanje vremenskih informacija | [backendApi.ts](src/api/backendApi.ts#L149) | 149 |
| 22 | `/api/training/save-time-info` | POST | Čuvanje vremenskih informacija | [backendApi.ts](src/api/backendApi.ts#L169) | 169 |

### Korištenje:
- **TimeInformationInput.tsx:214** - GET poziv
- **TimeInformationInput.tsx:282** - POST poziv (auto-save)
- **constants.ts:10** - Definiše `SAVE_TIME_INFO`

---

## 8. Zeitschritte (Time Steps)

**Broj endpointa:** 2

| # | Endpoint | Metoda | Opis | Fajl | Linija |
|---|----------|--------|------|------|--------|
| 23 | `/api/training/get-zeitschritte/{sessionId}` | GET | Dohvatanje vremenskih koraka | [backendApi.ts](src/api/backendApi.ts#L204) | 204 |
| 24 | `/api/training/save-zeitschritte` | POST | Čuvanje vremenskih koraka | [backendApi.ts](src/api/backendApi.ts#L224) | 224 |

### Korištenje:
- **AllgemeineDaten.tsx:28** - GET poziv
- **AllgemeineDaten.tsx:50** - POST poziv (kreiranje default vrednosti)
- **AllgemeineDaten.tsx:110** - POST poziv (auto-save sa 500ms debounce)
- **constants.ts:11** - Definiše `SAVE_ZEITSCHRITTE`

---

## 9. Session Management

**Broj endpointa:** 7

| # | Endpoint | Metoda | Opis | Fajl | Linija |
|---|----------|--------|------|------|--------|
| 25 | `/api/training/list-sessions` | GET | Lista svih sesija | [SessionSelector.tsx](src/features/training/components/SessionSelector.tsx#L58) | 58 |
| 26 | `/api/training/session-name-change` | POST | Promena imena sesije | [SessionSelector.tsx](src/features/training/components/SessionSelector.tsx#L108) | 108 |
| 27 | `/api/training/session/{sessionId}/delete` | POST | Brisanje sesije | [SessionSelector.tsx](src/features/training/components/SessionSelector.tsx#L161) | 161 |
| 28 | `/api/training/session/{sessionId}/database` | GET | Detalji sesije iz baze sa n_dat | [useSessionManagement.ts](src/features/training/hooks/useSessionManagement.ts#L25) | 25,55 |
| 29 | `/api/training/session-status/{sessionId}` | GET | Status sesije | [useSessionManagement.ts](src/features/training/hooks/useSessionManagement.ts#L156) | 156 |
| 30 | `/api/training/create-database-session` | POST | Kreiranje sesije u bazi | [useSessionManagement.ts](src/features/training/hooks/useSessionManagement.ts#L166) | 166 |
| 31 | `/api/training/delete-all-sessions` | POST | Brisanje svih sesija | [backendApi.ts](src/api/backendApi.ts#L283) | 283 |

### Dodatne Reference:
- **supabase.ts:456** - `/session/{sessionId}/database`
- **supabase.ts:610** - `/list-sessions`
- **supabase.ts:617** - `/session/{sessionId}/delete`
- **SessionsTable.tsx:71** - `/list-sessions`
- **constants.ts:7,9** - `SESSION_STATUS`, `CREATE_DATABASE_SESSION`

---

## 10. Scalers

**Broj endpointa:** 2

| # | Endpoint | Metoda | Opis | Fajl | Linija |
|---|----------|--------|------|------|--------|
| 32 | `/api/training/scalers/{sessionId}` | GET | Preuzimanje scalers podataka | [TrainingPage.tsx](src/pages/training/TrainingPage.tsx#L343) | 343 |
| 33 | `/api/training/scalers/{sessionId}/download` | GET | Download scalers kao ZIP | [downloadUtils.ts](src/features/training/utils/downloadUtils.ts#L36) | 36 |

---

## 11. Training Status/Polling

**Broj endpointa:** 1 (multiple references)

| # | Endpoint | Metoda | Opis | Fajl | Linija |
|---|----------|--------|------|------|--------|
| 34 | `/api/training/status/{sessionId}` | GET | Status treniranja (polling) | [useTrainingPolling.ts](src/features/training/hooks/useTrainingPolling.ts#L52) | 52,110 |

### Dodatne Reference:
- **useTrainingPolling.ts:160** - Poll dataset generation status
- **useTrainingPolling.ts:210** - Poll model training status
- **constants.ts:8** - Definiše `TRAINING_STATUS`

---

## 12. Upload/Chunked Upload

**Broj endpointa:** 4

| # | Endpoint | Metoda | Opis | Fajl | Linija |
|---|----------|--------|------|------|--------|
| 35 | `/api/training/init-session` | POST | Inicijalizacija upload sesije | [ChunkedUploader.ts](src/core/services/upload/ChunkedUploader.ts#L89) | 89 |
| 36 | `/api/training/upload-chunk` | POST | Upload pojedinačnog chunka | [ChunkedUploader.ts](src/core/services/upload/ChunkedUploader.ts#L138) | 138 |
| 37 | `/api/training/finalize-session` | POST | Finalizacija upload-a | [ChunkedUploader.ts](src/core/services/upload/ChunkedUploader.ts#L237) | 237 |
| 29* | `/api/training/session-status/{sessionId}` | GET | Status sesije (duplikat #29) | [ChunkedUploader.ts](src/core/services/upload/ChunkedUploader.ts#L377) | 377 |

**Napomena:** Endpoint #29 je duplikat sa Session Management sekcijom.

---

## 13. Utility Endpoints

**Broj endpointa:** 1

| # | Endpoint | Metoda | Opis | Fajl | Linija |
|---|----------|--------|------|------|--------|
| 38 | `/api/training/get-session-uuid/{sessionId}` | GET | Dohvatanje UUID-a sesije | [supabase.ts](src/core/database/supabase.ts#L35) | 35 |

---

## Statistika i Analiza

### Ukupan Broj Endpointa

| Kategorija | Broj Jedinstvenih Endpointa |
|------------|----------------------------|
| Training Core Operations | 7 |
| Model Management | 5 |
| Evaluation & Results | 1 |
| Visualization | 1 |
| Plotting Interface | 2 |
| CSV File Management | 4 |
| Time Information | 2 |
| Zeitschritte | 2 |
| Session Management | 7 |
| Scalers | 2 |
| Training Status/Polling | 1 |
| Upload/Chunked Upload | 3 |
| Utility | 1 |
| **UKUPNO** | **37** |

### Duplikati i Višestruke Reference

Sledeći endpointi imaju više referenci u kodu:

1. **`/api/training/session-status/{sessionId}`** - 2 reference
   - useSessionManagement.ts:156
   - ChunkedUploader.ts:377

2. **`/api/training/results/{sessionId}`** - 3 reference
   - trainingApiService.ts:287
   - useSessionManagement.ts:146
   - useViolinPlots.ts:22,40

3. **`/api/training/evaluation-tables/{sessionId}`** - 2 reference
   - EvaluationTables.tsx:67
   - shared/EvaluationTables.tsx:51

4. **`/api/training/visualizations/{sessionId}`** - 2 reference
   - trainingApiService.ts:300
   - VisualizationViolinDiagramContainer.tsx:39

5. **`/api/training/list-sessions`** - 3 reference
   - SessionSelector.tsx:58
   - supabase.ts:610
   - SessionsTable.tsx:71

6. **`/api/training/session/{sessionId}/database`** - 3 reference
   - useSessionManagement.ts:25,55
   - supabase.ts:456

7. **`/api/training/status/{sessionId}`** - 2 reference
   - useTrainingPolling.ts:52,110

### API Pozivi po HTTP Metodi

| Metoda | Broj Endpointa | Procenat |
|--------|---------------|----------|
| GET | 24 | 64.9% |
| POST | 13 | 35.1% |
| PUT | 1 | 2.7% |
| DELETE | 1 | 2.7% |
| **UKUPNO** | **37** | **100%** |

### Najčešće Korišteni Endpointi

Top 5 endpointa sa najviše referenci u kodu:

1. `/api/training/results/{sessionId}` - **3 reference**
2. `/api/training/list-sessions` - **3 reference**
3. `/api/training/session/{sessionId}/database` - **3 reference**
4. `/api/training/session-status/{sessionId}` - **2 reference**
5. `/api/training/status/{sessionId}` - **2 reference**

### Endpointi sa Auto-Save Funkcijom

Sledeći endpointi implementiraju auto-save sa debounce mehanizmom:

1. **`/api/training/save-zeitschritte`** - 500ms debounce (AllgemeineDaten.tsx:110)
2. **`/api/training/csv-files/{fileId}`** (PUT) - Auto-save metadata (CSVFileUploader.tsx:162)

### Kritične Operacije

Endpointi koji zahtevaju posebnu pažnju:

- **`/api/training/delete-all-sessions`** - Briše sve sesije (backendApi.ts:283)
- **`/api/training/session/{sessionId}/delete`** - Briše pojedinačnu sesiju
- **`/api/training/csv-files/{fileId}`** (DELETE) - Briše CSV fajl

---

## Napomene za Backend Cleanup

### Endpointi za Proveru

Pre brisanja sa backend-a, proveri:

1. Da li endpoint ima više od jedne reference?
2. Da li se koristi u polling mehanizmima?
3. Da li je deo kritičnog workflow-a (upload, training, evaluation)?

### Preporuke

- **NE BRIŠI** endpointe sa više od 2 reference
- **OBAVEZNO PROVERI** polling endpointe pre brisanja
- **PAŽLJIVO** sa CRUD operacijama (CSV files, sessions)
- **TESTIRAJ** nakon brisanja nekorištenih endpointa

---

**Generisano:** 2025-10-17
**Autor:** Claude Code Analysis
**Status:** ✅ Finalna verifikacija završena
