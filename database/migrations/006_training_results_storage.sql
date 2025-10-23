-- Migration: 006_training_results_storage.sql
-- Purpose: Setup Storage bucket and RLS policies for training results
-- Date: 2025-10-22

-- ============================================================================
-- STORAGE BUCKET SETUP
-- ============================================================================

-- Kreiranje training-results bucket-a (if not exists)
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'training-results',
    'training-results',
    false,  -- Private bucket
    52428800,  -- 50MB file size limit
    ARRAY['application/json', 'application/gzip', 'application/octet-stream']
)
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- RLS POLICIES FOR STORAGE
-- ============================================================================

-- Drop existing policies if they exist (for idempotency)
DROP POLICY IF EXISTS "Users can upload their training results" ON storage.objects;
DROP POLICY IF EXISTS "Users can read their training results" ON storage.objects;
DROP POLICY IF EXISTS "Users can delete their training results" ON storage.objects;
DROP POLICY IF EXISTS "Service role has full access to training results" ON storage.objects;

-- Policy 1: Users can upload their own training results
-- File path format: {session_id}/training_results_*.json[.gz]
CREATE POLICY "Users can upload their training results"
ON storage.objects FOR INSERT
WITH CHECK (
    bucket_id = 'training-results' AND
    (
        -- Allow if user owns the session
        auth.uid() IN (
            SELECT user_id FROM public.sessions
            WHERE id::text = (storage.foldername(name))[1]
        )
        OR
        -- Allow service role
        auth.jwt() ->> 'role' = 'service_role'
    )
);

-- Policy 2: Users can read their own training results
CREATE POLICY "Users can read their training results"
ON storage.objects FOR SELECT
USING (
    bucket_id = 'training-results' AND
    (
        -- Allow if user owns the session
        auth.uid() IN (
            SELECT user_id FROM public.sessions
            WHERE id::text = (storage.foldername(name))[1]
        )
        OR
        -- Allow service role
        auth.jwt() ->> 'role' = 'service_role'
    )
);

-- Policy 3: Users can delete their own training results
CREATE POLICY "Users can delete their training results"
ON storage.objects FOR DELETE
USING (
    bucket_id = 'training-results' AND
    (
        -- Allow if user owns the session
        auth.uid() IN (
            SELECT user_id FROM public.sessions
            WHERE id::text = (storage.foldername(name))[1]
        )
        OR
        -- Allow service role
        auth.jwt() ->> 'role' = 'service_role'
    )
);

-- Policy 4: Service role has full access (backup policy)
CREATE POLICY "Service role has full access to training results"
ON storage.objects FOR ALL
USING (
    bucket_id = 'training-results' AND
    auth.jwt() ->> 'role' = 'service_role'
);

-- ============================================================================
-- VERIFICATION
-- ============================================================================

-- Verify bucket was created
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM storage.buckets WHERE id = 'training-results') THEN
        RAISE NOTICE '✅ Storage bucket "training-results" created successfully';
    ELSE
        RAISE EXCEPTION '❌ Failed to create storage bucket';
    END IF;
END $$;

-- Verify policies were created
DO $$
DECLARE
    policy_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO policy_count
    FROM pg_policies
    WHERE schemaname = 'storage' AND tablename = 'objects'
    AND policyname LIKE '%training results%';

    IF policy_count >= 4 THEN
        RAISE NOTICE '✅ RLS policies created successfully (% policies)', policy_count;
    ELSE
        RAISE WARNING '⚠️  Expected 4 policies, found %', policy_count;
    END IF;
END $$;
