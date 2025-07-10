-- Base schema that must exist before training tables
-- These are the core tables that the existing code expects

-- Create sessions table (referenced by existing code)
CREATE TABLE IF NOT EXISTS public.sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create session mappings table (used by existing supabase_client.py)
CREATE TABLE IF NOT EXISTS public.session_mappings (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    string_session_id VARCHAR(255) NOT NULL,
    uuid_session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique mappings
    UNIQUE(string_session_id),
    UNIQUE(uuid_session_id)
);

-- Create time_info table (used by existing supabase_client.py)
CREATE TABLE IF NOT EXISTS public.time_info (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
    jahr BOOLEAN DEFAULT FALSE,
    woche BOOLEAN DEFAULT FALSE,
    monat BOOLEAN DEFAULT FALSE,
    feiertag BOOLEAN DEFAULT FALSE,
    tag BOOLEAN DEFAULT FALSE,
    zeitzone VARCHAR(100) DEFAULT 'UTC',
    category_data JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure only one time_info per session
    UNIQUE(session_id)
);

-- Create zeitschritte table (used by existing supabase_client.py)
CREATE TABLE IF NOT EXISTS public.zeitschritte (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
    eingabe VARCHAR(100),
    ausgabe VARCHAR(100),
    zeitschrittweite VARCHAR(100),
    offset VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure only one zeitschritte per session
    UNIQUE(session_id)
);

-- Create files table (used by existing supabase_client.py)
CREATE TABLE IF NOT EXISTS public.files (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
    file_name VARCHAR(255),
    bezeichnung VARCHAR(255),
    min VARCHAR(100),
    max VARCHAR(100),
    offsett VARCHAR(100),
    datenpunkte VARCHAR(100),
    numerische_datenpunkte VARCHAR(100),
    numerischer_anteil VARCHAR(100),
    datenform VARCHAR(100),
    datenanpassung VARCHAR(100),
    zeitschrittweite VARCHAR(100),
    zeitschrittweite_mittelwert VARCHAR(100),
    zeitschrittweite_min VARCHAR(100),
    skalierung VARCHAR(100) DEFAULT 'nein',
    skalierung_max VARCHAR(100),
    skalierung_min VARCHAR(100),
    zeithorizont_start VARCHAR(100),
    zeithorizont_end VARCHAR(100),
    zeitschrittweite_transferierten_daten VARCHAR(100),
    offset_transferierten_daten VARCHAR(100),
    mittelwertbildung_uber_den_zeithorizont VARCHAR(100) DEFAULT 'nein',
    storage_path VARCHAR(500),
    type VARCHAR(50),
    utc_min TIMESTAMP WITH TIME ZONE,
    utc_max TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create csv_file_refs table (used by existing supabase_client.py)
CREATE TABLE IF NOT EXISTS public.csv_file_refs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    file_id UUID NOT NULL REFERENCES public.files(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
    file_name VARCHAR(255),
    storage_path VARCHAR(500),
    file_size BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON public.sessions(created_at);

CREATE INDEX IF NOT EXISTS idx_session_mappings_string_id ON public.session_mappings(string_session_id);
CREATE INDEX IF NOT EXISTS idx_session_mappings_uuid_id ON public.session_mappings(uuid_session_id);

CREATE INDEX IF NOT EXISTS idx_time_info_session_id ON public.time_info(session_id);
CREATE INDEX IF NOT EXISTS idx_zeitschritte_session_id ON public.zeitschritte(session_id);

CREATE INDEX IF NOT EXISTS idx_files_session_id ON public.files(session_id);
CREATE INDEX IF NOT EXISTS idx_files_type ON public.files(type);

CREATE INDEX IF NOT EXISTS idx_csv_file_refs_file_id ON public.csv_file_refs(file_id);
CREATE INDEX IF NOT EXISTS idx_csv_file_refs_session_id ON public.csv_file_refs(session_id);

-- Enable RLS on the tables
ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.session_mappings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.time_info ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.zeitschritte ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.files ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.csv_file_refs ENABLE ROW LEVEL SECURITY;

-- Create policies for service role (backend can access everything)
CREATE POLICY "Service role can access all sessions" ON public.sessions
    FOR ALL USING (true);

CREATE POLICY "Service role can access all session_mappings" ON public.session_mappings
    FOR ALL USING (true);

CREATE POLICY "Service role can access all time_info" ON public.time_info
    FOR ALL USING (true);

CREATE POLICY "Service role can access all zeitschritte" ON public.zeitschritte
    FOR ALL USING (true);

CREATE POLICY "Service role can access all files" ON public.files
    FOR ALL USING (true);

CREATE POLICY "Service role can access all csv_file_refs" ON public.csv_file_refs
    FOR ALL USING (true);

-- Create updated_at trigger function if it doesn't exist
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers to automatically update updated_at columns
CREATE TRIGGER update_sessions_updated_at 
    BEFORE UPDATE ON public.sessions 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_time_info_updated_at 
    BEFORE UPDATE ON public.time_info 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_zeitschritte_updated_at 
    BEFORE UPDATE ON public.zeitschritte 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_files_updated_at 
    BEFORE UPDATE ON public.files 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_csv_file_refs_updated_at 
    BEFORE UPDATE ON public.csv_file_refs 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE public.sessions IS 'Main sessions table - each session represents a training workflow';
COMMENT ON TABLE public.session_mappings IS 'Maps string session IDs from frontend to UUID session IDs in database';
COMMENT ON TABLE public.time_info IS 'Time-based configuration for each session';
COMMENT ON TABLE public.zeitschritte IS 'Time step configuration for training';
COMMENT ON TABLE public.files IS 'File metadata for uploaded CSV files';
COMMENT ON TABLE public.csv_file_refs IS 'References to CSV files stored in Supabase storage';