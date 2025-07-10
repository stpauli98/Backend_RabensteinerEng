-- Fix foreign key references to match actual table structure

-- First, check the actual structure of sessions table
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'sessions';

-- Check the actual structure of zeitschritte table
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'zeitschritte';

-- Fix zeitschritte table foreign key reference
-- Drop the existing constraint if it exists
DO $$ 
BEGIN
    -- Drop existing foreign key constraint if it exists
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'zeitschritte_session_id_fkey' 
        AND table_name = 'zeitschritte'
    ) THEN
        ALTER TABLE zeitschritte DROP CONSTRAINT zeitschritte_session_id_fkey;
        RAISE NOTICE 'Dropped existing foreign key constraint';
    END IF;
    
    -- Add correct foreign key constraint
    ALTER TABLE zeitschritte 
    ADD CONSTRAINT zeitschritte_session_id_fkey 
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE;
    
    RAISE NOTICE 'Added correct foreign key constraint';
END $$;

-- Fix time_info table foreign key reference
DO $$ 
BEGIN
    -- Drop existing foreign key constraint if it exists
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'time_info_session_id_fkey' 
        AND table_name = 'time_info'
    ) THEN
        ALTER TABLE time_info DROP CONSTRAINT time_info_session_id_fkey;
        RAISE NOTICE 'Dropped existing time_info foreign key constraint';
    END IF;
    
    -- Add correct foreign key constraint
    ALTER TABLE time_info 
    ADD CONSTRAINT time_info_session_id_fkey 
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE;
    
    RAISE NOTICE 'Added correct time_info foreign key constraint';
END $$;

-- Fix training_results table foreign key reference
DO $$ 
BEGIN
    -- Drop existing foreign key constraint if it exists
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'training_results_session_id_fkey' 
        AND table_name = 'training_results'
    ) THEN
        ALTER TABLE training_results DROP CONSTRAINT training_results_session_id_fkey;
        RAISE NOTICE 'Dropped existing training_results foreign key constraint';
    END IF;
    
    -- Add correct foreign key constraint
    ALTER TABLE training_results 
    ADD CONSTRAINT training_results_session_id_fkey 
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE;
    
    RAISE NOTICE 'Added correct training_results foreign key constraint';
END $$;

-- Fix training_visualizations table foreign key reference  
DO $$ 
BEGIN
    -- Drop existing foreign key constraint if it exists
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'training_visualizations_session_id_fkey' 
        AND table_name = 'training_visualizations'
    ) THEN
        ALTER TABLE training_visualizations DROP CONSTRAINT training_visualizations_session_id_fkey;
        RAISE NOTICE 'Dropped existing training_visualizations foreign key constraint';
    END IF;
    
    -- Add correct foreign key constraint
    ALTER TABLE training_visualizations 
    ADD CONSTRAINT training_visualizations_session_id_fkey 
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE;
    
    RAISE NOTICE 'Added correct training_visualizations foreign key constraint';
END $$;

-- Fix training_logs table foreign key reference
DO $$ 
BEGIN
    -- Drop existing foreign key constraint if it exists
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'training_logs_session_id_fkey' 
        AND table_name = 'training_logs'
    ) THEN
        ALTER TABLE training_logs DROP CONSTRAINT training_logs_session_id_fkey;
        RAISE NOTICE 'Dropped existing training_logs foreign key constraint';
    END IF;
    
    -- Add correct foreign key constraint
    ALTER TABLE training_logs 
    ADD CONSTRAINT training_logs_session_id_fkey 
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE;
    
    RAISE NOTICE 'Added correct training_logs foreign key constraint';
END $$;