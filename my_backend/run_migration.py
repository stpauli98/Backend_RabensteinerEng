#!/usr/bin/env python3
"""
Database Migration Runner
Runs the dataset columns migration safely
"""

import os
import sys
from supabase_client import get_supabase_client

def run_migration():
    """Run the add_error_type_column migration"""
    
    # Get Supabase client
    supabase = get_supabase_client()
    
    if not supabase:
        print("❌ Error: Could not connect to Supabase")
        sys.exit(1)
    
    # Read migration SQL
    migration_file = os.path.join(os.path.dirname(__file__), 'training_system', 'migrations', 'add_error_type_column.sql')
    
    if not os.path.exists(migration_file):
        print(f"❌ Error: Migration file not found: {migration_file}")
        sys.exit(1)
    
    with open(migration_file, 'r') as f:
        migration_sql = f.read()
    
    try:
        print("🔄 Running database migration: add_error_type_column.sql")
        print("📝 Migration SQL:")
        print("=" * 50)
        print(migration_sql[:500] + "..." if len(migration_sql) > 500 else migration_sql)
        print("=" * 50)
        
        # Execute migration using raw SQL
        result = supabase.rpc('execute_sql', {'sql': migration_sql})
        
        print("✅ Migration completed successfully!")
        print("📊 Result:", result)
        
        # Verify the column was added
        verify_sql = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'training_results' 
        AND column_name = 'error_type'
        ORDER BY column_name;
        """
        
        verification_result = supabase.rpc('execute_sql', {'sql': verify_sql})
        print("🔍 Verification - error_type column added:")
        print(verification_result)
        
    except Exception as e:
        print(f"❌ Error running migration: {e}")
        print("💡 You may need to run this migration manually in the Supabase SQL editor")
        print("📋 Migration SQL to run manually:")
        print("=" * 50)
        print(migration_sql)
        print("=" * 50)
        sys.exit(1)

if __name__ == "__main__":
    run_migration()