-- Database schema extensions for training system
-- These tables extend the existing schema to support training results
-- 
-- IMPORTANT: Run base_schema.sql FIRST before running this file!
-- The base_schema.sql creates the sessions table and other core tables
-- that this file references.

-- Table for storing training results
CREATE TABLE IF NOT EXISTS public.training_results (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL DEFAULT 'pending', -- pending, running, completed, failed
    results JSONB, -- Complete training results
    evaluation_metrics JSONB, -- Evaluation metrics for all models
    model_comparison JSONB, -- Model comparison data
    training_metadata JSONB, -- Training configuration and metadata
    best_model_info JSONB, -- Information about the best performing model
    error_message TEXT, -- Error message if training failed
    error_traceback TEXT, -- Full error traceback for debugging
    
    -- Process tracking for audit trail
    processing_started_by VARCHAR(100), -- Process ID that started training
    processing_info JSONB, -- Additional processing information
    heartbeat_history JSONB, -- History of heartbeats for monitoring
    
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for tracking training progress (optimized for session isolation)
CREATE TABLE IF NOT EXISTS public.training_progress (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
    overall_progress INTEGER DEFAULT 0 CHECK (overall_progress >= 0 AND overall_progress <= 100),
    current_step VARCHAR(100),
    total_steps INTEGER DEFAULT 7,
    completed_steps INTEGER DEFAULT 0,
    step_details JSONB, -- Detailed progress for each step
    model_progress JSONB, -- Progress for individual models
    estimated_time_remaining INTEGER, -- Estimated seconds remaining
    
    -- Session isolation and monitoring fields
    status VARCHAR(20) DEFAULT 'idle' CHECK (status IN ('idle', 'running', 'completed', 'failed', 'abandoned')),
    process_id VARCHAR(100), -- ID of the process handling this session
    process_info JSONB, -- Additional process information (PID, hostname, etc.)
    last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Timing information
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure only one progress record per session
    UNIQUE(session_id)
);

-- Table for storing training logs (optimized for batch operations)
CREATE TABLE IF NOT EXISTS public.training_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
    level VARCHAR(20) NOT NULL DEFAULT 'INFO', -- DEBUG, INFO, WARNING, ERROR, CRITICAL
    message TEXT NOT NULL,
    step VARCHAR(100), -- Which step generated this log
    model_name VARCHAR(50), -- Which model generated this log (if applicable)
    details JSONB, -- Additional structured log data
    
    -- Batch processing optimization
    batch_id UUID, -- Group related log entries for batch processing
    sequence_number INTEGER, -- Order within batch
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for storing training visualizations
CREATE TABLE IF NOT EXISTS public.training_visualizations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
    plot_type VARCHAR(50) NOT NULL, -- violin, forecast, comparison, history, residual
    plot_name VARCHAR(100) NOT NULL, -- Specific plot identifier
    dataset_name VARCHAR(100), -- Which dataset this plot belongs to
    model_name VARCHAR(50), -- Which model this plot belongs to (if applicable)
    image_data TEXT, -- Base64 encoded image data
    storage_path VARCHAR(500), -- Path in Supabase storage if stored as file
    plot_metadata JSONB, -- Metadata about the plot (size, format, etc.)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create optimized indexes for session isolation and performance
CREATE INDEX IF NOT EXISTS idx_training_results_session_id ON public.training_results(session_id);
CREATE INDEX IF NOT EXISTS idx_training_results_status ON public.training_results(status);
CREATE INDEX IF NOT EXISTS idx_training_results_created_at ON public.training_results(created_at);
CREATE INDEX IF NOT EXISTS idx_training_results_processing ON public.training_results(processing_started_by, status);

-- Optimized indexes for training_progress (session isolation focused)
CREATE INDEX IF NOT EXISTS idx_training_progress_session_id ON public.training_progress(session_id);
CREATE INDEX IF NOT EXISTS idx_training_progress_status_heartbeat ON public.training_progress(status, last_heartbeat);
CREATE INDEX IF NOT EXISTS idx_training_progress_process_id ON public.training_progress(process_id);
CREATE INDEX IF NOT EXISTS idx_training_progress_heartbeat ON public.training_progress(last_heartbeat);

-- Indexes for log batch processing
CREATE INDEX IF NOT EXISTS idx_training_logs_session_id ON public.training_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_training_logs_batch_sequence ON public.training_logs(batch_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_training_logs_level_created ON public.training_logs(level, created_at);

CREATE INDEX IF NOT EXISTS idx_training_visualizations_session_id ON public.training_visualizations(session_id);
CREATE INDEX IF NOT EXISTS idx_training_visualizations_plot_type ON public.training_visualizations(plot_type);

-- Add Row Level Security (RLS) policies if needed
-- Note: Adjust these based on your authentication setup

-- Enable RLS on the tables
ALTER TABLE public.training_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.training_progress ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.training_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.training_visualizations ENABLE ROW LEVEL SECURITY;

-- Create policies for service role (backend can access everything)
CREATE POLICY "Service role can access all training_results" ON public.training_results
    FOR ALL USING (true);

CREATE POLICY "Service role can access all training_progress" ON public.training_progress
    FOR ALL USING (true);

CREATE POLICY "Service role can access all training_logs" ON public.training_logs
    FOR ALL USING (true);

CREATE POLICY "Service role can access all training_visualizations" ON public.training_visualizations
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
CREATE TRIGGER update_training_results_updated_at 
    BEFORE UPDATE ON public.training_results 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_training_progress_updated_at 
    BEFORE UPDATE ON public.training_progress 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE public.training_results IS 'Stores complete training results and metadata for each session';
COMMENT ON TABLE public.training_progress IS 'Tracks real-time progress of training sessions';
COMMENT ON TABLE public.training_logs IS 'Stores detailed logs from training processes';
COMMENT ON TABLE public.training_visualizations IS 'Stores generated plots and charts from training results';

COMMENT ON COLUMN public.training_results.results IS 'Complete JSONB structure containing all training results';
COMMENT ON COLUMN public.training_results.evaluation_metrics IS 'Evaluation metrics for all trained models';
COMMENT ON COLUMN public.training_results.model_comparison IS 'Comparison data between different models';
COMMENT ON COLUMN public.training_results.best_model_info IS 'Information about the best performing model';

COMMENT ON COLUMN public.training_progress.step_details IS 'JSONB array containing progress for each training step';
COMMENT ON COLUMN public.training_progress.model_progress IS 'JSONB object containing progress for individual models';

COMMENT ON COLUMN public.training_visualizations.image_data IS 'Base64 encoded image data for small plots';
COMMENT ON COLUMN public.training_visualizations.storage_path IS 'Supabase storage path for larger image files';

-- Additional comments for session isolation features
COMMENT ON COLUMN public.training_progress.status IS 'Session status: idle, running, completed, failed, abandoned';
COMMENT ON COLUMN public.training_progress.process_id IS 'Unique identifier of the process handling this session';
COMMENT ON COLUMN public.training_progress.last_heartbeat IS 'Last activity timestamp for dead session detection';
COMMENT ON COLUMN public.training_logs.batch_id IS 'Groups log entries for efficient batch processing';

-- Function for automatic session cleanup (dead session detection)
CREATE OR REPLACE FUNCTION cleanup_abandoned_sessions()
RETURNS INTEGER AS $$
DECLARE
    cleanup_count INTEGER;
BEGIN
    -- Mark sessions as abandoned if no heartbeat for 5 minutes and status is running
    UPDATE public.training_progress 
    SET status = 'abandoned', 
        updated_at = NOW()
    WHERE status = 'running' 
    AND last_heartbeat < NOW() - INTERVAL '5 minutes';
    
    GET DIAGNOSTICS cleanup_count = ROW_COUNT;
    
    RETURN cleanup_count;
END;
$$ LANGUAGE plpgsql;

-- Function for session acquisition (atomic locking)
CREATE OR REPLACE FUNCTION acquire_session_lock(
    p_session_id UUID,
    p_process_id VARCHAR(100),
    p_process_info JSONB DEFAULT NULL
)
RETURNS BOOLEAN AS $$
DECLARE
    lock_acquired BOOLEAN := FALSE;
BEGIN
    -- Try to acquire lock on idle session
    UPDATE public.training_progress 
    SET status = 'running',
        process_id = p_process_id,
        process_info = p_process_info,
        started_at = NOW(),
        last_heartbeat = NOW(),
        updated_at = NOW()
    WHERE session_id = p_session_id 
    AND status IN ('idle', 'abandoned');
    
    -- Check if we successfully acquired the lock
    lock_acquired := FOUND;
    
    -- If no existing record, create one
    IF NOT lock_acquired THEN
        INSERT INTO public.training_progress (
            session_id, status, process_id, process_info, 
            started_at, last_heartbeat
        ) VALUES (
            p_session_id, 'running', p_process_id, p_process_info,
            NOW(), NOW()
        )
        ON CONFLICT (session_id) DO NOTHING;
        
        lock_acquired := FOUND;
    END IF;
    
    RETURN lock_acquired;
END;
$$ LANGUAGE plpgsql;

-- Function for updating heartbeat
CREATE OR REPLACE FUNCTION update_session_heartbeat(
    p_session_id UUID,
    p_process_id VARCHAR(100)
)
RETURNS BOOLEAN AS $$
DECLARE
    heartbeat_updated BOOLEAN := FALSE;
BEGIN
    UPDATE public.training_progress 
    SET last_heartbeat = NOW(),
        updated_at = NOW()
    WHERE session_id = p_session_id 
    AND process_id = p_process_id
    AND status = 'running';
    
    heartbeat_updated := FOUND;
    
    RETURN heartbeat_updated;
END;
$$ LANGUAGE plpgsql;