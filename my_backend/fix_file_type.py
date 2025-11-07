#!/usr/bin/env python3
"""
Fix file type mismatch: Update 444.csv from 'input' to 'output'
Session: 7c08a59a-bcef-471f-a982-6dbca9b3fd80
"""

from utils.database import get_supabase_client
from datetime import datetime

def fix_file_type():
    supabase = get_supabase_client()
    session_id = '7c08a59a-bcef-471f-a982-6dbca9b3fd80'
    file_name = '444.csv'

    print(f"🔧 Fixing file type for {file_name} in session {session_id}")

    # Check current state
    print("\n📊 Current file types:")
    response = supabase.table('files').select('file_name, type').eq('session_id', session_id).execute()
    for file in response.data:
        print(f"  - {file['file_name']}: {file['type']}")

    # Update 444.csv to 'output'
    print(f"\n✏️ Updating {file_name} type to 'output'...")
    update_response = supabase.table('files')\
        .update({
            'type': 'output',
            'updated_at': datetime.utcnow().isoformat()
        })\
        .eq('session_id', session_id)\
        .eq('file_name', file_name)\
        .execute()

    if update_response.data:
        print(f"✅ Successfully updated {file_name} to type 'output'")
    else:
        print(f"❌ Failed to update {file_name}")
        return False

    # Verify the fix
    print("\n📊 Updated file types:")
    verify_response = supabase.table('files').select('file_name, type').eq('session_id', session_id).execute()
    for file in verify_response.data:
        print(f"  - {file['file_name']}: {file['type']}")

    # Count files by type
    input_files = [f for f in verify_response.data if f['type'] == 'input']
    output_files = [f for f in verify_response.data if f['type'] == 'output']

    print(f"\n✅ File counts: {len(input_files)} input files, {len(output_files)} output files")

    return len(output_files) > 0

if __name__ == '__main__':
    success = fix_file_type()
    if success:
        print("\n✅ File type fix completed successfully!")
        print("🚀 Ready to re-run training")
    else:
        print("\n❌ File type fix failed")
