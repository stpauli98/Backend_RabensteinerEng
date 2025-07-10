-- Database schema for training results
-- This extends the existing schema with tables needed for frontend integration

-- Training results table
CREATE TABLE IF NOT EXISTS training_results (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(uuid) ON DELETE CASCADE,
    evaluation_metrics JSONB,
    model_performance JSONB,
    best_model JSONB,
    summary JSONB,
    status VARCHAR(50) DEFAULT 'running',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    error_traceback TEXT
);

-- Training visualizations table  
CREATE TABLE IF NOT EXISTS training_visualizations (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(uuid) ON DELETE CASCADE,
    plot_name VARCHAR(255) NOT NULL,
    plot_type VARCHAR(100) NOT NULL,
    plot_data_base64 TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Training logs table for detailed progress tracking
CREATE TABLE IF NOT EXISTS training_logs (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(uuid) ON DELETE CASCADE,
    message TEXT NOT NULL,
    level VARCHAR(20) NOT NULL DEFAULT 'INFO',
    step_number INTEGER,
    step_name VARCHAR(255),
    progress_percentage INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_training_results_session_id ON training_results(session_id);
CREATE INDEX IF NOT EXISTS idx_training_results_status ON training_results(status);
CREATE INDEX IF NOT EXISTS idx_training_results_created_at ON training_results(created_at);

CREATE INDEX IF NOT EXISTS idx_training_visualizations_session_id ON training_visualizations(session_id);
CREATE INDEX IF NOT EXISTS idx_training_visualizations_plot_type ON training_visualizations(plot_type);

CREATE INDEX IF NOT EXISTS idx_training_logs_session_id ON training_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_training_logs_timestamp ON training_logs(timestamp);
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

CREATE POLICY IF NOT EXISTS "Users can view their own training results" ON training_results
    FOR SELECT USING (true); -- Adjust based on your user system

CREATE POLICY IF NOT EXISTS "Users can insert their own training results" ON training_results
    FOR INSERT WITH CHECK (true); -- Adjust based on your user system

CREATE POLICY IF NOT EXISTS "Users can update their own training results" ON training_results
    FOR UPDATE USING (true); -- Adjust based on your user system

CREATE POLICY IF NOT EXISTS "Users can view their own visualizations" ON training_visualizations
    FOR SELECT USING (true);

CREATE POLICY IF NOT EXISTS "Users can insert their own visualizations" ON training_visualizations
    FOR INSERT WITH CHECK (true);

CREATE POLICY IF NOT EXISTS "Users can view their own logs" ON training_logs
    FOR SELECT USING (true);

CREATE POLICY IF NOT EXISTS "Users can insert their own logs" ON training_logs
    FOR INSERT WITH CHECK (true);

-- Comments for documentation
COMMENT ON TABLE training_results IS 'Stores complete training results including evaluation metrics and model performance';
COMMENT ON TABLE training_visualizations IS 'Stores base64-encoded plots and visualizations from training';
COMMENT ON TABLE training_logs IS 'Stores detailed logs and progress information during training';

COMMENT ON COLUMN training_results.evaluation_metrics IS 'JSONB containing all evaluation metrics (wape, smape, mase, mae, etc.)';
COMMENT ON COLUMN training_results.model_performance IS 'JSONB containing model-specific performance data';
COMMENT ON COLUMN training_results.best_model IS 'JSONB containing information about the best performing model';
COMMENT ON COLUMN training_visualizations.plot_data_base64 IS 'Base64-encoded PNG image data for frontend display';
COMMENT ON COLUMN training_logs.progress_percentage IS 'Progress percentage (0-100) for real-time updates';