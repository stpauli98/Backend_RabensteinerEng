"""
Data loader module for training system
Handles loading session data from database and downloading files
"""

import os
import pandas as pd
from supabase import create_client
from typing import Dict, List, Optional, Tuple
import logging

# TODO: Import from your existing supabase_client
# from supabase_client import supabase

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading training session data from database and file storage
    """
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.temp_dir = "temp_training_data"
        self._ensure_temp_dir()
    
    def _ensure_temp_dir(self):
        """Create temporary directory for downloaded files"""
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def load_session_data(self, session_id: str) -> Dict:
        """
        Load all session data from database
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict containing all session data
        """
        try:
            # Load session info
            session_data = self._load_session_info(session_id)
            
            # Load time configuration
            time_info = self._load_time_info(session_id)
            
            # Load zeitschritte configuration
            zeitschritte = self._load_zeitschritte(session_id)
            
            # Load file metadata
            files_info = self._load_files_info(session_id)
            
            return {
                'session': session_data,
                'time_info': time_info,
                'zeitschritte': zeitschritte,
                'files': files_info
            }
            
        except Exception as e:
            logger.error(f"Error loading session data for {session_id}: {str(e)}")
            raise
    
    def _load_session_info(self, session_id: str) -> Dict:
        """Load basic session information"""
        try:
            # TODO: Implement actual database query
            # This is placeholder based on your database schema
            
            response = self.supabase.table('sessions').select('*').eq('session_id', session_id).execute()
            
            if not response.data:
                raise ValueError(f"Session {session_id} not found")
            
            return response.data[0]
            
        except Exception as e:
            logger.error(f"Error loading session info: {str(e)}")
            raise
    
    def _load_time_info(self, session_id: str) -> Dict:
        """Load time configuration for the session"""
        try:
            response = self.supabase.table('time_info').select('*').eq('session_id', session_id).execute()
            
            if not response.data:
                # Return default time info if none exists
                return {
                    'jahr': True,
                    'monat': True,
                    'woche': True,
                    'feiertag': True,
                    'zeitzone': 'UTC'
                }
            
            return response.data[0]
            
        except Exception as e:
            logger.error(f"Error loading time info: {str(e)}")
            raise
    
    def _load_zeitschritte(self, session_id: str) -> Dict:
        """Load zeitschritte (time steps) configuration"""
        try:
            response = self.supabase.table('zeitschritte').select('*').eq('session_id', session_id).execute()
            
            if not response.data:
                # Return default zeitschritte if none exists
                return {
                    'eingabe': '24',
                    'ausgabe': '1',
                    'zeitschrittweite': '1',
                    'offset': '0'
                }
            
            return response.data[0]
            
        except Exception as e:
            logger.error(f"Error loading zeitschritte: {str(e)}")
            raise
    
    def _load_files_info(self, session_id: str) -> List[Dict]:
        """Load file metadata for the session"""
        try:
            response = self.supabase.table('files').select('*').eq('session_id', session_id).execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error loading files info: {str(e)}")
            raise
    
    def download_session_files(self, session_id: str) -> Dict[str, str]:
        """
        Download all CSV files for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict mapping file types to local file paths
        """
        try:
            files_info = self._load_files_info(session_id)
            downloaded_files = {}
            
            for file_info in files_info:
                file_type = file_info.get('type', 'unknown')
                file_name = file_info.get('fileName', 'unknown.csv')
                storage_path = file_info.get('storage_path', '')
                
                # Download file from storage
                local_path = self._download_file(storage_path, file_name, session_id)
                downloaded_files[file_type] = local_path
            
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Error downloading session files: {str(e)}")
            raise
    
    def _download_file(self, storage_path: str, file_name: str, session_id: str) -> str:
        """
        Download a single file from storage
        
        Args:
            storage_path: Path in storage bucket
            file_name: Name of the file
            session_id: Session identifier
            
        Returns:
            Local file path
        """
        try:
            # Create local file path
            local_file_path = os.path.join(self.temp_dir, f"{session_id}_{file_name}")
            
            # TODO: Implement actual file download from Supabase storage
            # This is placeholder based on your storage structure
            
            # Download from 'csv-files' bucket
            response = self.supabase.storage.from_('csv-files').download(storage_path)
            
            # Save to local file
            with open(local_file_path, 'wb') as f:
                f.write(response)
            
            logger.info(f"Downloaded {storage_path} to {local_file_path}")
            return local_file_path
            
        except Exception as e:
            logger.error(f"Error downloading file {storage_path}: {str(e)}")
            raise
    
    def prepare_file_paths(self, session_id: str) -> Tuple[List[str], List[str]]:
        """
        Prepare input and output file paths for training
        
        Args:
            session_id: Session identifier
            
        Returns:
            Tuple of (input_files, output_files) lists
        """
        try:
            downloaded_files = self.download_session_files(session_id)
            
            input_files = []
            output_files = []
            
            for file_type, file_path in downloaded_files.items():
                if file_type == 'input':
                    input_files.append(file_path)
                elif file_type == 'output':
                    output_files.append(file_path)
            
            return input_files, output_files
            
        except Exception as e:
            logger.error(f"Error preparing file paths: {str(e)}")
            raise
    
    def load_csv_data(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV data from file path
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Pandas DataFrame
        """
        try:
            # TODO: Add proper CSV loading with error handling
            # Based on how your CSV files are structured
            
            df = pd.read_csv(file_path)
            
            # Basic validation
            if df.empty:
                raise ValueError(f"CSV file {file_path} is empty")
            
            logger.info(f"Loaded CSV data from {file_path}: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV data from {file_path}: {str(e)}")
            raise
    
    def cleanup_temp_files(self, session_id: str):
        """
        Clean up temporary files for a session
        
        Args:
            session_id: Session identifier
        """
        try:
            import glob
            
            # Find all files for this session
            pattern = os.path.join(self.temp_dir, f"{session_id}_*")
            files_to_remove = glob.glob(pattern)
            
            for file_path in files_to_remove:
                os.remove(file_path)
                logger.info(f"Removed temporary file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {str(e)}")


# Factory function to create data loader
def create_data_loader(supabase_client) -> DataLoader:
    """
    Create and return a DataLoader instance
    
    Args:
        supabase_client: Configured Supabase client
        
    Returns:
        DataLoader instance
    """
    return DataLoader(supabase_client)