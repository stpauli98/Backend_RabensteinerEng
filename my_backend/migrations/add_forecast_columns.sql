-- migrations/add_forecast_columns.sql
-- Forecast Analyse: Add missing columns to files table + api_parameters lookup
-- Applied: 2026-03-31 via Supabase MCP

-- 1. New columns on files table
ALTER TABLE files ADD COLUMN IF NOT EXISTS data_source varchar;
ALTER TABLE files ADD COLUMN IF NOT EXISTS api_source varchar;
ALTER TABLE files ADD COLUMN IF NOT EXISTS fcst_var varchar;
ALTER TABLE files ADD COLUMN IF NOT EXISTS latitude numeric;
ALTER TABLE files ADD COLUMN IF NOT EXISTS longitude numeric;
ALTER TABLE files ADD COLUMN IF NOT EXISTS feature_index integer;

-- 2. API parameters lookup table
CREATE TABLE IF NOT EXISTS api_parameters (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    api_source varchar NOT NULL,
    parameter varchar NOT NULL,
    unit varchar,
    name_user varchar NOT NULL,
    lbl varchar NOT NULL,
    created_at timestamptz DEFAULT now(),
    UNIQUE(api_source, parameter)
);

-- 3. Seed GeoSphere parameters (19 rows)
INSERT INTO api_parameters (api_source, parameter, unit, name_user, lbl) VALUES
('GeoSphere', 'cape', 'm2 s-2', 'Convective Available Potential Energy (Gewitter-/Konvektionspotenzial) [m2 s-2]', 'cape [m2 s-2]'),
('GeoSphere', 'cin', 'J kg-1', 'Convective Inhibition (Hemmung von Konvektion) [J kg-1]', 'cin [J kg-1]'),
('GeoSphere', 'grad', 'Ws m-2', 'Globalstrahlung (Energie pro Fläche über Zeitintervall) [Ws m-2]', 'grad [Ws m-2]'),
('GeoSphere', 'mnt2m', 'degree Celsius', 'Minimale Lufttemperatur in 2 m Höhe [°C]', 'mnt2m [degree Celsius]'),
('GeoSphere', 'mxt2m', 'degree Celsius', 'Maximale Lufttemperatur in 2 m Höhe [°C]', 'mxt2m [degree Celsius]'),
('GeoSphere', 'rain_acc', 'kg m-2', 'Akkumulierter Regen [kg m-2]', 'rain_acc [kg m-2]'),
('GeoSphere', 'rh2m', '%', 'Relative Luftfeuchtigkeit in 2 m Höhe [%]', 'rh2m [%]'),
('GeoSphere', 'rr_acc', 'kg m-2', 'Akkumulierter Gesamtniederschlag [kg m-2]', 'rr_acc [kg m-2]'),
('GeoSphere', 'snow_acc', 'kg m-2', 'Akkumulierter Schneefall [kg m-2]', 'snow_acc [kg m-2]'),
('GeoSphere', 'snowlmt', 'm', 'Schneefallgrenze [m]', 'snowlmt [m]'),
('GeoSphere', 'sp', 'Pa', 'Luftdruck an der Oberfläche [Pa]', 'sp [Pa]'),
('GeoSphere', 'sundur_acc', 's', 'Akkumulierte Sonnenscheindauer [s]', 'sundur_acc [s]'),
('GeoSphere', 'sy', '1', 'Dimensionsloser Modellparameter [1]', 'sy [1]'),
('GeoSphere', 't2m', 'degree Celsius', 'Lufttemperatur in 2 m Höhe [°C]', 't2m [degree Celsius]'),
('GeoSphere', 'tcc', '1', 'Gesamtbewölkung (0–1) [1]', 'tcc [1]'),
('GeoSphere', 'u10m', 'm s-1', 'Windkomponente U in 10 m Höhe [m s-1]', 'u10m [m s-1]'),
('GeoSphere', 'ugust', 'm s-1', 'Windböen U-Komponente [m s-1]', 'ugust [m s-1]'),
('GeoSphere', 'v10m', 'm s-1', 'Windkomponente V in 10 m Höhe [m s-1]', 'v10m [m s-1]'),
('GeoSphere', 'vgust', 'm s-1', 'Windböen V-Komponente [m s-1]', 'vgust [m s-1]')
ON CONFLICT (api_source, parameter) DO NOTHING;

-- 4. Seed OpenWeather parameters (12 rows)
INSERT INTO api_parameters (api_source, parameter, unit, name_user, lbl) VALUES
('OpenWeather', 'feels_like', '', 'Gefühlte Temperatur [°C]', 'feels_like [°C]'),
('OpenWeather', 'grnd_level', '', 'Luftdruck auf Bodenhöhe [hPa]', 'grnd_level [hPa]'),
('OpenWeather', 'humidity', '', 'Relative Luftfeuchtigkeit [%]', 'humidity [%]'),
('OpenWeather', 'pressure', '', 'Luftdruck [hPa]', 'pressure [hPa]'),
('OpenWeather', 'sea_level', '', 'Luftdruck auf Meereshöhe [hPa]', 'sea_level [hPa]'),
('OpenWeather', 'temp', '', 'Lufttemperatur [°C]', 'temp [°C]'),
('OpenWeather', 'temp_kf', '', 'Interner Korrekturfaktor [-]', 'temp_kf [°C]'),
('OpenWeather', 'temp_max', '', 'Maximale Lufttemperatur [°C]', 'temp_max [°C]'),
('OpenWeather', 'temp_min', '', 'Minimale Lufttemperatur [°C]', 'temp_min [°C]'),
('OpenWeather', 'deg', '', 'Windrichtung [°]', 'deg [°]'),
('OpenWeather', 'gust', '', 'Windböen [m/s]', 'gust [m/s]'),
('OpenWeather', 'speed', '', 'Windgeschwindigkeit [m/s]', 'speed [m/s]')
ON CONFLICT (api_source, parameter) DO NOTHING;

-- 5. Forecasts table
CREATE TABLE IF NOT EXISTS forecasts (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id uuid REFERENCES sessions(id),
    user_id uuid,
    status varchar DEFAULT 'pending' CHECK (status IN ('pending','running','completed','failed')),
    user_inputs jsonb,
    utc_ref timestamptz,
    result_data jsonb,
    error_message text,
    started_at timestamptz,
    completed_at timestamptz,
    created_at timestamptz DEFAULT now()
);

-- 6. Enable RLS on new tables
ALTER TABLE api_parameters ENABLE ROW LEVEL SECURITY;
ALTER TABLE forecasts ENABLE ROW LEVEL SECURITY;

-- 7. RLS policies
CREATE POLICY "Anyone can read api_parameters" ON api_parameters FOR SELECT USING (true);
CREATE POLICY "Users can read own forecasts" ON forecasts FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own forecasts" ON forecasts FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own forecasts" ON forecasts FOR UPDATE USING (auth.uid() = user_id);
