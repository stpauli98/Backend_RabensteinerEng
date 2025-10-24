"""Cleanup service for temporary files"""
import os
import time
import logging

logger = logging.getLogger(__name__)

temp_files = {}

def cleanup_old_files():
    """Clean up files older than 60 minutes from temp_uploads directory"""
    success = True
    errors = []
    deleted_count = 0
    current_time = time.time()
    EXPIRY_TIME = 60 * 60
    
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'api', 'temp_uploads')
    
    try:
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                try:
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > EXPIRY_TIME:
                        os.remove(file_path)
                        deleted_count += 1
                except Exception as e:
                    success = False
                    errors.append(f"Error with {name}: {str(e)}")
                    logger.error(f"Error cleaning up file {name}: {str(e)}")
            
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    os.rmdir(dir_path)
                except OSError:
                    pass
        
        for file_id, file_info in list(temp_files.items()):
            if not os.path.exists(file_info['path']):
                del temp_files[file_id]
    
        logger.info(f"Cleaned up {deleted_count} files older than 60 minutes")
        return {
            "success": success,
            "message": f"Cleaned up {deleted_count} files older than 60 minutes",
            "deleted_count": deleted_count,
            "errors": errors if errors else None
        }
                
    except Exception as e:
        logger.error(f"Error in cleanup_old_files: {str(e)}")
        return {"error": str(e)}
