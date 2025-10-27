# Kako Dobiti JWT Token za Testiranje

**User**: test@rabensteiner.com
**Supabase Project**: luvjebsltuttakatnzaa

---

## METODA 1: Login kroz Supabase Auth API (Najbrže)

### Korak 1: Pripremi podatke
```bash
SUPABASE_URL="https://luvjebsltuttakatnzaa.supabase.co"
SUPABASE_ANON_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx1dmplYnNsdHV0dGFrYXRuemFhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDkwMzcyMjEsImV4cCI6MjA2NDYxMzIyMX0.IZqPkAkUu0paFqRP8a6BLXk-K9h183wmX9QtdVRACEY"
EMAIL="test@rabensteiner.com"
PASSWORD="tvoj_password_ovdje"  # TREBAM PASSWORD!
```

### Korak 2: Login i dobij token
```bash
curl -X POST "${SUPABASE_URL}/auth/v1/token?grant_type=password" \
  -H "apikey: ${SUPABASE_ANON_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"email\": \"${EMAIL}\",
    \"password\": \"${PASSWORD}\"
  }" | jq -r '.access_token'
```

### Korak 3: Spremi token
```bash
# Ako je uspješno, output će biti JWT token
export FREE_USER_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6..."

# Sad možeš testirati:
curl -X POST http://localhost:8080/api/training/train-models/test123 \
  -H "Authorization: Bearer ${FREE_USER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"model_parameters": {}}'
```

**Problem**: Moraš znati password za test@rabensteiner.com!

---

## METODA 2: Supabase Dashboard (Najsigurnije)

### Koraci:
1. Otvori Supabase Dashboard: https://supabase.com/dashboard
2. Logiraj se sa svojim accountom
3. Odaberi projekt: **luvjebsltuttakatnzaa**
4. Idi na: **Authentication** → **Users**
5. Pronađi: **test@rabensteiner.com**
6. Klikni na usera
7. Skrolaj do **User UID** i kopiraj ID
8. Idi na **SQL Editor**
9. Izvršiti:

```sql
-- Generiraj access token za test user-a
SELECT auth.jwt(
  json_build_object(
    'sub', 'f4e69951-af93-4db8-9521-eadc4021e13c',
    'email', 'test@rabensteiner.com',
    'role', 'authenticated',
    'aud', 'authenticated',
    'exp', extract(epoch from now() + interval '1 hour')::integer
  )
);
```

**Output će biti JWT token koji možeš koristati!**

---

## METODA 3: Reset Password i Login (Ako Ne Znaš Password)

### Korak 1: Reset password kroz Supabase Dashboard

1. Otvori Supabase Dashboard
2. Projekt: **luvjebsltuttakatnzaa**
3. **Authentication** → **Users**
4. Pronađi: **test@rabensteiner.com**
5. Klikni **"..."** (three dots)
6. Klikni **"Send Password Reset Email"**
7. Provjeri email inbox za test@rabensteiner.com
8. Klikni link i postavi novi password (npr. "TestPassword123!")

### Korak 2: Login sa novim passwordom

```bash
curl -X POST "https://luvjebsltuttakatnzaa.supabase.co/auth/v1/token?grant_type=password" \
  -H "apikey: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx1dmplYnNsdHV0dGFrYXRuemFhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDkwMzcyMjEsImV4cCI6MjA2NDYxMzIyMX0.IZqPkAkUu0paFqRP8a6BLXk-K9h183wmX9QtdVRACEY" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@rabensteiner.com",
    "password": "TestPassword123!"
  }' | jq .
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "...",
  "user": {
    "id": "f4e69951-af93-4db8-9521-eadc4021e13c",
    "email": "test@rabensteiner.com"
  }
}
```

### Korak 3: Izvuci token
```bash
# Izvuci samo access_token
TOKEN=$(curl -s -X POST "https://luvjebsltuttakatnzaa.supabase.co/auth/v1/token?grant_type=password" \
  -H "apikey: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx1dmplYnNsdHV0dGFrYXRuemFhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDkwMzcyMjEsImV4cCI6MjA2NDYxMzIyMX0.IZqPkAkUu0paFqRP8a6BLXk-K9h183wmX9QtdVRACEY" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@rabensteiner.com",
    "password": "TestPassword123!"
  }' | jq -r '.access_token')

echo "JWT Token: $TOKEN"

# Spremi u environment variable
export FREE_USER_TOKEN="$TOKEN"
```

---

## METODA 4: Kreiraj Novi Test User (Alternativa)

Ako ne možeš pristupiti test@rabensteiner.com emailu, kreiraj novi test account:

### Korak 1: Registracija novog usera
```bash
curl -X POST "https://luvjebsltuttakatnzaa.supabase.co/auth/v1/signup" \
  -H "apikey: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx1dmplYnNsdHV0dGFrYXRuemFhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDkwMzcyMjEsImV4cCI6MjA2NDYxMzIyMX0.IZqPkAkUu0paFqRP8a6BLXk-K9h183wmX9QtdVRACEY" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "testuser2@rabensteiner.com",
    "password": "TestPassword123!"
  }' | jq .
```

### Korak 2: Dodaj Free plan novom useru

Nakon što dobiješ user_id iz responsa:

```bash
# Preko MCP Supabase ili SQL
# INSERT INTO user_subscriptions (user_id, plan_id, status, ...)
```

---

## BRZI TEST (nakon što dobiješ token)

```bash
# Spremi token
export FREE_USER_TOKEN="tvoj_jwt_token_ovdje"

# Test 1: Free user NE MOŽE training
echo "=== Test 1: Free user pokušava training ==="
curl -s -X POST http://localhost:8080/api/training/train-models/test123 \
  -H "Authorization: Bearer ${FREE_USER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"model_parameters": {}}' | jq .

# Očekivano: {"error": "Training not available", "plan": "Free"}

# Test 2: Free user MOŽE generate datasets (do 5 puta)
echo "=== Test 2: Free user generira dataset ==="
curl -s -X POST http://localhost:8080/api/training/generate-datasets/test123 \
  -H "Authorization: Bearer ${FREE_USER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"model_parameters": {}}' | jq .

# Očekivano (1-5 put): {"success": true, ...}
# Očekivano (6. put): {"error": "Processing limit reached", "current_usage": 5, "limit": 5}
```

---

## Troubleshooting

### Problem: "Invalid or expired token"
- Token je istekao (1 sat validnost)
- Generiši novi token ponovo

### Problem: "No active subscription"
- User nema aktivan plan
- Provjeri: `SELECT * FROM user_subscriptions WHERE user_id = 'f4e69951-af93-4db8-9521-eadc4021e13c' AND status = 'active';`

### Problem: "Authentication failed"
- Pogrešan password
- Reset password ili koristi Metodu 2

### Problem: Email nije dostupan
- Koristi Metodu 4 - kreiraj novi test account

---

## PREPORUKA

**Najbrži način**: 
1. Idi na Supabase Dashboard
2. Authentication → Users → test@rabensteiner.com
3. Klikni "Send Password Reset Email"
4. Postavi novi password (npr. "TestPassword123!")
5. Login sa curl komandom (Metoda 3, Korak 2)
6. Kopiraj access_token
7. Testiraj!

