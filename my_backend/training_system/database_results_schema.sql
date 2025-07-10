-- Database schema for training results
-- This extends the existing schema with tables needed for frontend integration

-- Training results table
CREATE TABLE IF NOT EXISTS training_results (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(uuid) ON DELETE CASCADE,
    status VARCHAR(50) DEFAULT 'running',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    error_message TEXT,
    error_traceback TEXT
);

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

-- Add constraints for required columns (only if columns exist and are empty)
-- Note: These will fail if there's existing data that violates the constraints
DO $$ 
BEGIN
    -- Try to add NOT NULL constraint to message column if it exists and has no NULL values
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'training_logs' AND column_name = 'message') THEN
        BEGIN
            ALTER TABLE training_logs ALTER COLUMN message SET NOT NULL;
        EXCEPTION WHEN OTHERS THEN
            -- Ignore if constraint cannot be added (e.g., existing NULL values)
            NULL;
        END;
    END IF;
    
    -- Try to add NOT NULL constraint to level column if it exists and has no NULL values
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'training_logs' AND column_name = 'level') THEN
        BEGIN
            ALTER TABLE training_logs ALTER COLUMN level SET NOT NULL;
        EXCEPTION WHEN OTHERS THEN
            -- Ignore if constraint cannot be added
            NULL;
        END;
    END IF;
END $$;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_training_results_session_id ON training_results(session_id);
CREATE INDEX IF NOT EXISTS idx_training_results_status ON training_results(status);
CREATE INDEX IF NOT EXISTS idx_training_results_created_at ON training_results(created_at);

CREATE INDEX IF NOT EXISTS idx_training_visualizations_session_id ON training_visualizations(session_id);
CREATE INDEX IF NOT EXISTS idx_training_visualizations_plot_type ON training_visualizations(plot_type);

CREATE INDEX IF NOT EXISTS idx_training_logs_session_id ON training_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_training_logs_created_at ON training_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_training_logs_level ON training_logs(level);

-- Update existing training_progress table to ensure it has the right structure
ALTER TABLE training_progress 
ADD COLUMN IF NOT EXISTS step_number INTEGER,
ADD COLUMN IF NOT EXISTS step_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS error_message TEXT;

-- Row Level Security (RLS) policies for security
ALTER TABLE training_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_visualizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_logs ENABLE ROW LEVEL SECURITY;

-- RLS policies (adjust based on your authentication setup)
-- These are basic policies - modify based on your authentication system
-- Note: PostgreSQL doesn't support IF NOT EXISTS for policies, so these will fail if policies already exist

DO $$ 
BEGIN
    -- Training results policies
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'training_results' AND policyname = 'Users can view their own training results') THEN
        CREATE POLICY "Users can view their own training results" ON training_results
            FOR SELECT USING (true); -- Adjust based on your user system
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'training_results' AND policyname = 'Users can insert their own training results') THEN
        CREATE POLICY "Users can insert their own training results" ON training_results
            FOR INSERT WITH CHECK (true); -- Adjust based on your user system
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'training_results' AND policyname = 'Users can update their own training results') THEN
        CREATE POLICY "Users can update their own training results" ON training_results
            FOR UPDATE USING (true); -- Adjust based on your user system
    END IF;

    -- Training visualizations policies
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'training_visualizations' AND policyname = 'Users can view their own visualizations') THEN
        CREATE POLICY "Users can view their own visualizations" ON training_visualizations
            FOR SELECT USING (true);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'training_visualizations' AND policyname = 'Users can insert their own visualizations') THEN
        CREATE POLICY "Users can insert their own visualizations" ON training_visualizations
            FOR INSERT WITH CHECK (true);
    END IF;

    -- Training logs policies
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'training_logs' AND policyname = 'Users can view their own logs') THEN
        CREATE POLICY "Users can view their own logs" ON training_logs
            FOR SELECT USING (true);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'training_logs' AND policyname = 'Users can insert their own logs') THEN
        CREATE POLICY "Users can insert their own logs" ON training_logs
            FOR INSERT WITH CHECK (true);
    END IF;
END $$;

-- Comments for documentation
COMMENT ON TABLE training_results IS 'Stores complete training results including evaluation metrics and model performance';
COMMENT ON TABLE training_visualizations IS 'Stores base64-encoded plots and visualizations from training';
COMMENT ON TABLE training_logs IS 'Stores detailed logs and progress information during training';

COMMENT ON COLUMN training_results.evaluation_metrics IS 'JSONB containing all evaluation metrics (wape, smape, mase, mae, etc.)';
COMMENT ON COLUMN training_results.model_performance IS 'JSONB containing model-specific performance data';
COMMENT ON COLUMN training_results.best_model IS 'JSONB containing information about the best performing model';
COMMENT ON COLUMN training_visualizations.plot_data_base64 IS 'Base64-encoded PNG image data for frontend display';
COMMENT ON COLUMN training_logs.progress_percentage IS 'Progress percentage (0-100) for real-time updates';
COMMENT ON COLUMN training_logs.created_at IS 'Timestamp when the log entry was created';