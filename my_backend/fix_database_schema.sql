-- Fix database schema issues for training system

-- Check if zeitschritte table exists and fix missing columns
DO $$ 
BEGIN
    -- Check if zeitschritte table exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'zeitschritte') THEN
        -- Create zeitschritte table if it doesn't exist
        CREATE TABLE zeitschritte (
            id BIGSERIAL PRIMARY KEY,
            session_id UUID REFERENCES sessions(uuid) ON DELETE CASCADE,
            eingabe TEXT,
            ausgabe TEXT,
            zeitschrittweite TEXT,
            offset_value TEXT,  -- Use offset_value instead of offset (reserved keyword)
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        RAISE NOTICE 'Created zeitschritte table';
    ELSE
        -- Add missing columns if table exists
        -- Use offset_value instead of offset to avoid reserved keyword issues
        ALTER TABLE zeitschritte ADD COLUMN IF NOT EXISTS offset_value TEXT;
        RAISE NOTICE 'Added missing columns to zeitschritte table';
    END IF;
END $$;

-- Check if time_info table exists and fix missing columns
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'time_info') THEN
        -- Create time_info table if it doesn't exist
        CREATE TABLE time_info (
            id BIGSERIAL PRIMARY KEY,
            session_id UUID REFERENCES sessions(uuid) ON DELETE CASCADE,
            jahr BOOLEAN DEFAULT FALSE,
            monat BOOLEAN DEFAULT FALSE,
            woche BOOLEAN DEFAULT FALSE,
            tag BOOLEAN DEFAULT FALSE,
            feiertag BOOLEAN DEFAULT FALSE,
            zeitzone VARCHAR(50) DEFAULT 'UTC',
            category_data JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        RAISE NOTICE 'Created time_info table';
    END IF;
END $$;

-- Update existing training system tables with proper structure
-- Add missing columns to existing training_results table
ALTER TABLE training_results 
ADD COLUMN IF NOT EXISTS evaluation_metrics JSONB,
ADD COLUMN IF NOT EXISTS model_performance JSONB,
ADD COLUMN IF NOT EXISTS best_model JSONB,
ADD COLUMN IF NOT EXISTS summary JSONB,
ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP WITH TIME ZONE;

-- Training visualizations table  
CREATE TABLE IF NOT EXISTS training_visualizations (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(uuid) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add missing columns to existing training_visualizations table
ALTER TABLE training_visualizations 
ADD COLUMN IF NOT EXISTS plot_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS plot_type VARCHAR(100),
ADD COLUMN IF NOT EXISTS plot_data_base64 TEXT,
ADD COLUMN IF NOT EXISTS metadata JSONB;

-- Training logs table for detailed progress tracking
CREATE TABLE IF NOT EXISTS training_logs (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(uuid) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add missing columns to existing training_logs table
ALTER TABLE training_logs 
ADD COLUMN IF NOT EXISTS message TEXT,
ADD COLUMN IF NOT EXISTS level VARCHAR(20) DEFAULT 'INFO',
ADD COLUMN IF NOT EXISTS step_number INTEGER,
ADD COLUMN IF NOT EXISTS step_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS progress_percentage INTEGER;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_training_results_session_id ON training_results(session_id);
CREATE INDEX IF NOT EXISTS idx_training_results_status ON training_results(status);
CREATE INDEX IF NOT EXISTS idx_training_results_created_at ON training_results(created_at);

CREATE INDEX IF NOT EXISTS idx_training_visualizations_session_id ON training_visualizations(session_id);
CREATE INDEX IF NOT EXISTS idx_training_visualizations_plot_type ON training_visualizations(plot_type);

CREATE INDEX IF NOT EXISTS idx_training_logs_session_id ON training_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_training_logs_created_at ON training_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_training_logs_level ON training_logs(level);

CREATE INDEX IF NOT EXISTS idx_zeitschritte_session_id ON zeitschritte(session_id);
CREATE INDEX IF NOT EXISTS idx_time_info_session_id ON time_info(session_id);

-- Grant necessary permissions (adjust based on your setup)
GRANT ALL PRIVILEGES ON TABLE zeitschritte TO postgres;
GRANT ALL PRIVILEGES ON TABLE time_info TO postgres;
GRANT ALL PRIVILEGES ON TABLE training_results TO postgres;
GRANT ALL PRIVILEGES ON TABLE training_visualizations TO postgres;
GRANT ALL PRIVILEGES ON TABLE training_logs TO postgres;

-- Grant sequence permissions
GRANT ALL PRIVILEGES ON SEQUENCE zeitschritte_id_seq TO postgres;
GRANT ALL PRIVILEGES ON SEQUENCE time_info_id_seq TO postgres;
GRANT ALL PRIVILEGES ON SEQUENCE training_results_id_seq TO postgres;
GRANT ALL PRIVILEGES ON SEQUENCE training_visualizations_id_seq TO postgres;
GRANT ALL PRIVILEGES ON SEQUENCE training_logs_id_seq TO postgres;