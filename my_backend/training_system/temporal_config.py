"""
Temporal features configuration - Dynamic configuration from database
Contains the T class structure that receives configuration from database/user settings
This replaces hardcoded values with dynamic configuration
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TemporalFeatureConfig:
    """Individual temporal feature configuration"""
    
    def __init__(self, config_data: Dict[str, Any] = None):
        """Initialize temporal feature from database/user config"""
        if config_data:
            self.IMP = config_data.get('IMP', False)
            self.LT = config_data.get('LT', False)
            self.SPEC = config_data.get('SPEC', 'Zeithorizont')
            self.TH_STRT = config_data.get('TH_STRT', 0)
            self.TH_END = config_data.get('TH_END', 0)
            self.SCAL = config_data.get('SCAL', True)
            self.SCAL_MAX = config_data.get('SCAL_MAX', 1)
            self.SCAL_MIN = config_data.get('SCAL_MIN', 0)
            self.CNTRY = config_data.get('CNTRY', 'Österreich')  # For holidays
        else:
            # Default fallback values
            self.IMP = False
            self.LT = False
            self.SPEC = 'Zeithorizont'
            self.TH_STRT = 0
            self.TH_END = 0
            self.SCAL = True
            self.SCAL_MAX = 1
            self.SCAL_MIN = 0
            self.CNTRY = 'Österreich'


class T:
    """
    Dynamic Temporal features configuration class
    Receives configuration from database instead of hardcoded values
    Maintains same interface as original for backwards compatibility
    """
    
    def __init__(self, temporal_config_data: Dict[str, Any] = None):
        """
        Initialize temporal configuration from database/user settings
        
        Args:
            temporal_config_data: Dictionary containing temporal feature configurations
                                Format: {
                                    'Y': {'IMP': True, 'TH_STRT': -24, ...},
                                    'M': {'IMP': False, 'TH_STRT': -1, ...},
                                    'W': {'IMP': True, 'TH_STRT': -24, ...},
                                    'D': {'IMP': False, 'TH_STRT': -24, ...},
                                    'H': {'IMP': True, 'TH_STRT': -100, 'CNTRY': 'Österreich', ...}
                                }
        """
        
        if temporal_config_data:
            # Initialize from database/user config
            self.Y = TemporalFeatureConfig(temporal_config_data.get('Y'))
            self.M = TemporalFeatureConfig(temporal_config_data.get('M'))
            self.W = TemporalFeatureConfig(temporal_config_data.get('W'))
            self.D = TemporalFeatureConfig(temporal_config_data.get('D'))
            self.H = TemporalFeatureConfig(temporal_config_data.get('H'))
            logger.info("Temporal configuration loaded from database")
        else:
            # Default fallback configuration (same as original hardcoded)
            self.Y = TemporalFeatureConfig({
                'IMP': False, 'LT': False, 'SPEC': 'Zeithorizont',
                'TH_STRT': -24, 'TH_END': 0, 'SCAL': True,
                'SCAL_MAX': 1, 'SCAL_MIN': 0
            })
            self.M = TemporalFeatureConfig({
                'IMP': False, 'LT': False, 'SPEC': 'Zeithorizont', 
                'TH_STRT': -1, 'TH_END': 0, 'SCAL': True,
                'SCAL_MAX': 1, 'SCAL_MIN': 0
            })
            self.W = TemporalFeatureConfig({
                'IMP': False, 'LT': False, 'SPEC': 'Aktuelle Zeit',
                'TH_STRT': -24, 'TH_END': 0, 'SCAL': True,
                'SCAL_MAX': 1, 'SCAL_MIN': 0
            })
            self.D = TemporalFeatureConfig({
                'IMP': False, 'LT': True, 'SPEC': 'Zeithorizont',
                'TH_STRT': -24, 'TH_END': 0, 'SCAL': True,
                'SCAL_MAX': 1, 'SCAL_MIN': 0
            })
            self.H = TemporalFeatureConfig({
                'IMP': False, 'LT': False, 'SPEC': 'Aktuelle Zeit',
                'TH_STRT': -100, 'TH_END': 0, 'SCAL': True,
                'SCAL_MAX': 1, 'SCAL_MIN': 0, 'CNTRY': 'Österreich'
            })
            logger.info("Using default temporal configuration")
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert temporal configuration to dictionary format for database storage"""
        return {
            'Y': {
                'IMP': self.Y.IMP, 'LT': self.Y.LT, 'SPEC': self.Y.SPEC,
                'TH_STRT': self.Y.TH_STRT, 'TH_END': self.Y.TH_END,
                'SCAL': self.Y.SCAL, 'SCAL_MAX': self.Y.SCAL_MAX, 'SCAL_MIN': self.Y.SCAL_MIN
            },
            'M': {
                'IMP': self.M.IMP, 'LT': self.M.LT, 'SPEC': self.M.SPEC,
                'TH_STRT': self.M.TH_STRT, 'TH_END': self.M.TH_END,
                'SCAL': self.M.SCAL, 'SCAL_MAX': self.M.SCAL_MAX, 'SCAL_MIN': self.M.SCAL_MIN
            },
            'W': {
                'IMP': self.W.IMP, 'LT': self.W.LT, 'SPEC': self.W.SPEC,
                'TH_STRT': self.W.TH_STRT, 'TH_END': self.W.TH_END,
                'SCAL': self.W.SCAL, 'SCAL_MAX': self.W.SCAL_MAX, 'SCAL_MIN': self.W.SCAL_MIN
            },
            'D': {
                'IMP': self.D.IMP, 'LT': self.D.LT, 'SPEC': self.D.SPEC,
                'TH_STRT': self.D.TH_STRT, 'TH_END': self.D.TH_END,
                'SCAL': self.D.SCAL, 'SCAL_MAX': self.D.SCAL_MAX, 'SCAL_MIN': self.D.SCAL_MIN
            },
            'H': {
                'IMP': self.H.IMP, 'LT': self.H.LT, 'SPEC': self.H.SPEC,
                'TH_STRT': self.H.TH_STRT, 'TH_END': self.H.TH_END,
                'SCAL': self.H.SCAL, 'SCAL_MAX': self.H.SCAL_MAX, 'SCAL_MIN': self.H.SCAL_MIN,
                'CNTRY': self.H.CNTRY
            }
        }
    
    def validate_config(self) -> bool:
        """Validate temporal configuration"""
        try:
            # Check that all required attributes exist
            for feature in [self.Y, self.M, self.W, self.D, self.H]:
                required_attrs = ['IMP', 'LT', 'SPEC', 'TH_STRT', 'TH_END', 'SCAL', 'SCAL_MAX', 'SCAL_MIN']
                for attr in required_attrs:
                    if not hasattr(feature, attr):
                        return False
            
            # Additional validation for holiday feature
            if not hasattr(self.H, 'CNTRY'):
                return False
                
            return True
        except Exception as e:
            logger.error(f"Temporal config validation failed: {e}")
            return False
    
    @classmethod
    def load_from_database(cls, supabase_client, session_id: str) -> 'T':
        """
        Load temporal configuration from database for specific session
        
        Args:
            supabase_client: Supabase client instance
            session_id: Training session ID
            
        Returns:
            T instance with database configuration
        """
        try:
            # Query temporal configuration from database
            response = supabase_client.table('temporal_configurations').select('*').eq('session_id', session_id).execute()
            
            if response.data and len(response.data) > 0:
                # Convert database response to temporal config format
                db_config = response.data[0]
                temporal_config_data = {
                    'Y': db_config.get('y_config', {}),
                    'M': db_config.get('m_config', {}),
                    'W': db_config.get('w_config', {}),
                    'D': db_config.get('d_config', {}),
                    'H': db_config.get('h_config', {})
                }
                
                logger.info(f"Loaded temporal configuration from database for session {session_id}")
                return cls(temporal_config_data)
            else:
                logger.info(f"No temporal configuration found for session {session_id}, using defaults")
                return cls()  # Return default configuration
                
        except Exception as e:
            logger.error(f"Error loading temporal configuration from database: {e}")
            return cls()  # Return default configuration on error
    
    def save_to_database(self, supabase_client, session_id: str) -> bool:
        """
        Save temporal configuration to database
        
        Args:
            supabase_client: Supabase client instance  
            session_id: Training session ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_dict = self.to_dict()
            
            # Prepare database record
            db_record = {
                'session_id': session_id,
                'y_config': config_dict['Y'],
                'm_config': config_dict['M'],
                'w_config': config_dict['W'],
                'd_config': config_dict['D'],
                'h_config': config_dict['H'],
                'created_at': 'now()',
                'updated_at': 'now()'
            }
            
            # Upsert (insert or update) configuration
            response = supabase_client.table('temporal_configurations').upsert(db_record).execute()
            
            if response.data:
                logger.info(f"Saved temporal configuration to database for session {session_id}")
                return True
            else:
                logger.error("Failed to save temporal configuration to database")
                return False
                
        except Exception as e:
            logger.error(f"Error saving temporal configuration to database: {e}")
            return False


# Factory function to create temporal configuration
def create_temporal_config(config_data: Dict[str, Any] = None) -> T:
    """
    Create temporal configuration instance
    
    Args:
        config_data: Optional configuration data from database/user
        
    Returns:
        T instance
    """
    return T(config_data)


# Create default instance for backwards compatibility
temporal_config = T()