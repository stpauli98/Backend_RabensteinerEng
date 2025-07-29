-- Migration: Add dataset columns to training_results table
-- This migration adds missing columns that are required for dataset generation tracking

-- Add dataset information columns to training_results table
DO $$ 
BEGIN
    -- Add dataset_count column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'training_results' 
        AND column_name = 'dataset_count'
    ) THEN
        ALTER TABLE public.training_results ADD COLUMN dataset_count INTEGER;
        RAISE NOTICE 'Added dataset_count column to training_results table';
    ELSE
        RAISE NOTICE 'dataset_count column already exists in training_results table';
    END IF;

    -- Add train_dataset_size column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'training_results' 
        AND column_name = 'train_dataset_size'
    ) THEN
        ALTER TABLE public.training_results ADD COLUMN train_dataset_size INTEGER;
        RAISE NOTICE 'Added train_dataset_size column to training_results table';
    ELSE
        RAISE NOTICE 'train_dataset_size column already exists in training_results table';
    END IF;

    -- Add val_dataset_size column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'training_results' 
        AND column_name = 'val_dataset_size'
    ) THEN
        ALTER TABLE public.training_results ADD COLUMN val_dataset_size INTEGER;
        RAISE NOTICE 'Added val_dataset_size column to training_results table';
    ELSE
        RAISE NOTICE 'val_dataset_size column already exists in training_results table';
    END IF;

    -- Add test_dataset_size column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'training_results' 
        AND column_name = 'test_dataset_size'
    ) THEN
        ALTER TABLE public.training_results ADD COLUMN test_dataset_size INTEGER;
        RAISE NOTICE 'Added test_dataset_size column to training_results table';
    ELSE
        RAISE NOTICE 'test_dataset_size column already exists in training_results table';
    END IF;

    -- Add dataset_generation_time column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'training_results' 
        AND column_name = 'dataset_generation_time'
    ) THEN
        ALTER TABLE public.training_results ADD COLUMN dataset_generation_time FLOAT;
        RAISE NOTICE 'Added dataset_generation_time column to training_results table';
    ELSE
        RAISE NOTICE 'dataset_generation_time column already exists in training_results table';
    END IF;

    -- Add datasets_info column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'training_results' 
        AND column_name = 'datasets_info'
    ) THEN
        ALTER TABLE public.training_results ADD COLUMN datasets_info JSONB;
        RAISE NOTICE 'Added datasets_info column to training_results table';
    ELSE
        RAISE NOTICE 'datasets_info column already exists in training_results table';
    END IF;

END $$;

-- Add comments for the new columns
COMMENT ON COLUMN public.training_results.dataset_count IS 'Number of datasets generated during the training session';
COMMENT ON COLUMN public.training_results.train_dataset_size IS 'Size of the training dataset';
COMMENT ON COLUMN public.training_results.val_dataset_size IS 'Size of the validation dataset';
COMMENT ON COLUMN public.training_results.test_dataset_size IS 'Size of the test dataset';
COMMENT ON COLUMN public.training_results.dataset_generation_time IS 'Time taken to generate datasets in seconds';
COMMENT ON COLUMN public.training_results.datasets_info IS 'JSONB metadata about generated datasets including names, shapes, and properties';

-- Verify migration completed successfully
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_schema = 'public' 
AND table_name = 'training_results' 
AND column_name IN ('dataset_count', 'train_dataset_size', 'val_dataset_size', 'test_dataset_size', 'dataset_generation_time', 'datasets_info')
ORDER BY column_name;