"""
One-time migration script to clean up orphaned storage files.

Usage:
    python -m scripts.cleanup_orphaned_storage           # Dry run
    python -m scripts.cleanup_orphaned_storage --delete  # Actually delete
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.database.client import get_supabase_admin_client


def cleanup_orphaned_training_results(dry_run: bool = True) -> dict:
    """Find and delete training result files with no corresponding DB record."""
    supabase = get_supabase_admin_client()

    print(f"\n{'='*60}")
    print(f"TRAINING RESULTS CLEANUP (dry_run={dry_run})")
    print(f"{'='*60}\n")

    db_response = supabase.table('training_results').select('results_file_path').execute()
    valid_paths = {r['results_file_path'] for r in db_response.data if r.get('results_file_path')}
    print(f"Found {len(valid_paths)} valid paths in database")

    try:
        storage_folders = supabase.storage.from_('training-results').list()
        print(f"Found {len(storage_folders)} session folders in storage")
    except Exception as e:
        print(f"ERROR: Could not list storage bucket: {e}")
        return {'orphaned_count': 0, 'orphaned_bytes': 0, 'deleted': 0}

    orphaned = []
    total_orphaned_bytes = 0

    for folder in storage_folders:
        session_id = folder['name']
        try:
            files = supabase.storage.from_('training-results').list(session_id)
        except Exception:
            continue

        for f in files:
            full_path = f"{session_id}/{f['name']}"
            file_size = f.get('metadata', {}).get('size', 0) if f.get('metadata') else 0

            if full_path not in valid_paths:
                orphaned.append({'path': full_path, 'size': file_size})
                total_orphaned_bytes += file_size

    print(f"\nOrphaned: {len(orphaned)} files ({total_orphaned_bytes/1024/1024:.1f} MB)")

    deleted_count = 0
    if not dry_run and orphaned:
        for item in orphaned:
            try:
                supabase.storage.from_('training-results').remove([item['path']])
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {item['path']}: {e}")
        print(f"Deleted: {deleted_count}/{len(orphaned)} files")

    return {'orphaned_count': len(orphaned), 'orphaned_bytes': total_orphaned_bytes, 'deleted': deleted_count}


def cleanup_orphaned_trained_models(dry_run: bool = True) -> dict:
    """Find and delete trained model files with no corresponding session."""
    supabase = get_supabase_admin_client()

    print(f"\n{'='*60}")
    print(f"TRAINED MODELS CLEANUP (dry_run={dry_run})")
    print(f"{'='*60}\n")

    db_response = supabase.table('sessions').select('id').execute()
    valid_session_ids = {str(r['id']) for r in db_response.data if r.get('id')}
    print(f"Found {len(valid_session_ids)} valid sessions in database")

    try:
        storage_folders = supabase.storage.from_('trained-models').list()
        print(f"Found {len(storage_folders)} session folders in storage")
    except Exception as e:
        print(f"ERROR: Could not list storage bucket: {e}")
        return {'orphaned_count': 0, 'orphaned_bytes': 0, 'deleted': 0}

    orphaned = []
    total_orphaned_bytes = 0

    for folder in storage_folders:
        session_id = folder['name']
        if session_id not in valid_session_ids:
            try:
                files = supabase.storage.from_('trained-models').list(session_id)
                for f in files:
                    file_size = f.get('metadata', {}).get('size', 0) if f.get('metadata') else 0
                    orphaned.append({'path': f"{session_id}/{f['name']}", 'size': file_size})
                    total_orphaned_bytes += file_size
            except Exception:
                continue

    print(f"\nOrphaned: {len(orphaned)} files ({total_orphaned_bytes/1024/1024:.1f} MB)")

    deleted_count = 0
    if not dry_run and orphaned:
        for item in orphaned:
            try:
                supabase.storage.from_('trained-models').remove([item['path']])
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {item['path']}: {e}")
        print(f"Deleted: {deleted_count}/{len(orphaned)} files")

    return {'orphaned_count': len(orphaned), 'orphaned_bytes': total_orphaned_bytes, 'deleted': deleted_count}


def main():
    dry_run = '--delete' not in sys.argv

    print("\n" + "="*60)
    print("ORPHANED STORAGE CLEANUP SCRIPT")
    print("="*60)
    print(f"\nMode: {'DRY RUN' if dry_run else 'DELETE MODE'}")

    if not dry_run:
        confirm = input("\nType 'YES' to confirm deletion: ")
        if confirm != 'YES':
            print("Aborted.")
            return

    results = cleanup_orphaned_training_results(dry_run=dry_run)
    models = cleanup_orphaned_trained_models(dry_run=dry_run)

    print(f"\n{'='*60}")
    print("TOTAL")
    print(f"{'='*60}")
    total = results['orphaned_count'] + models['orphaned_count']
    total_mb = (results['orphaned_bytes'] + models['orphaned_bytes']) / 1024 / 1024
    total_deleted = results['deleted'] + models['deleted']
    print(f"Orphaned: {total} files ({total_mb:.1f} MB)")
    print(f"Deleted: {total_deleted}")

    if dry_run and total > 0:
        print(f"\nRun with --delete to delete files")


if __name__ == '__main__':
    main()
