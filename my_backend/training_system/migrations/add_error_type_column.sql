-- Migration: Add error_type column to training_results table
-- Created: 2025-07-29
-- Purpose: Fix database schema error where error_type column is missing

-- Add error_type column if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'training_results' 
        AND column_name = 'error_type'
        AND table_schema = 'public'
    ) THEN
        ALTER TABLE public.training_results 
        ADD COLUMN error_type VARCHAR(50);
        
        -- Add comment for the new column
        COMMENT ON COLUMN public.training_results.error_type IS 'Type of error (training_failed, data_processing, database_error, etc.)';
        
        RAISE NOTICE 'Added error_type column to training_results table';
    ELSE
        RAISE NOTICE 'error_type column already exists in training_results table';
    END IF;
END $$;
