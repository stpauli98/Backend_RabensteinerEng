-- Migration: 007_training_results_to_storage.sql
-- Purpose: Add Storage-related columns to training_results table
-- Date: 2025-10-22

-- ============================================================================
-- ADD NEW COLUMNS FOR STORAGE
-- ============================================================================

-- Add columns for Storage file tracking
ALTER TABLE public.training_results
ADD COLUMN IF NOT EXISTS results_file_path TEXT,
ADD COLUMN IF NOT EXISTS file_size_bytes BIGINT,
ADD COLUMN IF NOT EXISTS compressed BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS results_metadata JSONB DEFAULT '{}'::jsonb;

-- ============================================================================
-- UPDATE COLUMN COMMENTS
-- ============================================================================

COMMENT ON COLUMN training_results.results IS
'DEPRECATED: Use results_file_path instead. Large training results are now stored in Storage bucket for better performance and to avoid timeout issues.';

COMMENT ON COLUMN training_results.results_file_path IS
'Path to training results JSON file in training-results storage bucket. Format: {session_id}/training_results_{timestamp}.json[.gz]';

COMMENT ON COLUMN training_results.file_size_bytes IS
'Size of the results file in bytes (compressed if applicable)';

COMMENT ON COLUMN training_results.compressed IS
'Whether the results file is gzip compressed';

COMMENT ON COLUMN training_results.results_metadata IS
'Quick-access metadata for SQL queries without downloading full results. Contains: {accuracy, loss, epochs_completed, model_type, dataset_count, training_split}';

COMMENT ON TABLE training_results IS
'Training results storage. New records store full results in Storage bucket (results_file_path), old records may have results in JSONB column.';

-- ============================================================================
-- CREATE INDEXES FOR PERFORMANCE
-- ============================================================================

-- Index for metadata queries (allows queries on nested JSON fields)
CREATE INDEX IF NOT EXISTS idx_training_results_metadata
ON training_results USING gin(results_metadata);

-- Index for file path lookup
CREATE INDEX IF NOT EXISTS idx_training_results_file_path
ON training_results(results_file_path)
WHERE results_file_path IS NOT NULL;

-- Index for finding results with storage files
CREATE INDEX IF NOT EXISTS idx_training_results_has_storage
ON training_results(session_id, created_at DESC)
WHERE results_file_path IS NOT NULL;

-- ============================================================================
-- HELPER FUNCTION FOR METADATA EXTRACTION
-- ============================================================================

-- Function to extract metadata from results JSONB (for migration purposes)
CREATE OR REPLACE FUNCTION extract_results_metadata(results_json JSONB)
RETURNS JSONB
LANGUAGE plpgsql
IMMUTABLE
AS $$
BEGIN
    RETURN jsonb_build_object(
        'accuracy', results_json->'metrics'->>'accuracy',
        'loss', results_json->'metrics'->>'loss',
        'epochs_completed', results_json->'parameters'->>'EP',
        'model_type', results_json->>'model_type',
        'dataset_count', results_json->>'dataset_count',
        'training_split', results_json->>'training_split'
    );
END;
$$;

COMMENT ON FUNCTION extract_results_metadata IS
'Extracts quick-access metadata from full training results JSONB. Used for migrating old records to new schema.';

-- ============================================================================
-- VERIFICATION
-- ============================================================================

DO $$
DECLARE
    col_count INTEGER;
    idx_count INTEGER;
BEGIN
    -- Verify columns were added
    SELECT COUNT(*) INTO col_count
    FROM information_schema.columns
    WHERE table_schema = 'public'
    AND table_name = 'training_results'
    AND column_name IN ('results_file_path', 'file_size_bytes', 'compressed', 'results_metadata');

    IF col_count = 4 THEN
        RAISE NOTICE '✅ All 4 new columns added successfully';
    ELSE
        RAISE WARNING '⚠️  Expected 4 columns, found %', col_count;
    END IF;

    -- Verify indexes were created
    SELECT COUNT(*) INTO idx_count
    FROM pg_indexes
    WHERE schemaname = 'public'
    AND tablename = 'training_results'
    AND indexname LIKE 'idx_training_results_%';

    IF idx_count >= 3 THEN
        RAISE NOTICE '✅ Indexes created successfully (% indexes)', idx_count;
    ELSE
        RAISE WARNING '⚠️  Expected at least 3 indexes, found %', idx_count;
    END IF;
END $$;

-- ============================================================================
-- OPTIONAL: MIGRATE EXISTING DATA (if needed)
-- ============================================================================

-- Uncomment this section if you want to migrate existing training_results to extract metadata
/*
UPDATE public.training_results
SET results_metadata = extract_results_metadata(results)
WHERE results IS NOT NULL
AND results_metadata = '{}'::jsonb;
*/
