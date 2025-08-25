-- Migration 001: Fix offset column naming
-- Date: 2025-08-23
-- Description: Rename offsett columns to offset for consistency

-- 1. Fix zeitschritte table
ALTER TABLE zeitschritte 
RENAME COLUMN offsett TO "offset";

-- 2. Fix files table  
ALTER TABLE files 
RENAME COLUMN offsett TO "offset";

-- Verify the changes
-- SELECT column_name FROM information_schema.columns 
-- WHERE table_name IN ('zeitschritte', 'files') 
-- AND column_name LIKE '%offset%';