#!/usr/bin/env python3
"""
Fix zeithorizont values for training session files
Session: 7c08a59a-bcef-471f-a982-6dbca9b3fd80
"""

from utils.database import get_supabase_client
from datetime import datetime

def fix_zeithorizont_values():
    supabase = get_supabase_client()
    session_id = '7c08a59a-bcef-471f-a982-6dbca9b3fd80'

    print(f"🔧 Fixing zeithorizont values for session {session_id}")

    # Check current state
    print("\n📊 Current state:")
    response = supabase.table('files').select('file_name, type, zeithorizont_start, zeithorizont_end').eq('session_id', session_id).execute()
    for file in response.data:
        print(f"  {file['file_name']}: type={file['type']}, start={file.get('zeithorizont_start')}, end={file.get('zeithorizont_end')}")

    # Update input file (222.csv): -1 to 0 (1 hour backward to current)
    print("\n✏️ Updating 222.csv (input): zeithorizont_start=-1, zeithorizont_end=0")
    update_response = supabase.table('files')\
        .update({
            'zeithorizont_start': '-1',
            'zeithorizont_end': '0',
            'updated_at': datetime.utcnow().isoformat()
        })\
        .eq('session_id', session_id)\
        .eq('file_name', '222.csv')\
        .execute()

    if update_response.data:
        print(f"✅ Successfully updated 222.csv")
    else:
        print(f"❌ Failed to update 222.csv")
        return False

    # Update output file (444.csv): 0 to 1 (current to 1 hour forward)
    print("\n✏️ Updating 444.csv (output): zeithorizont_start=0, zeithorizont_end=1")
    update_response = supabase.table('files')\
        .update({
            'zeithorizont_start': '0',
            'zeithorizont_end': '1',
            'updated_at': datetime.utcnow().isoformat()
        })\
        .eq('session_id', session_id)\
        .eq('file_name', '444.csv')\
        .execute()

    if update_response.data:
        print(f"✅ Successfully updated 444.csv")
    else:
        print(f"❌ Failed to update 444.csv")
        return False

    # Verify the fix
    print("\n📊 Updated state:")
    verify_response = supabase.table('files').select('file_name, type, zeithorizont_start, zeithorizont_end').eq('session_id', session_id).execute()
    for file in verify_response.data:
        print(f"  {file['file_name']}: type={file['type']}, start={file.get('zeithorizont_start')}, end={file.get('zeithorizont_end')}")

    return True

if __name__ == '__main__':
    success = fix_zeithorizont_values()
    if success:
        print("\n✅ Zeithorizont fix completed successfully!")
        print("🚀 Ready to retry training")
    else:
        print("\n❌ Zeithorizont fix failed")
