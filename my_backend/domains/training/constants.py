"""
Constants for training domain.

Centralized definitions for TIME components and other training-related constants.
These are the KNOWN possible time component names based on the original training.py structure.
"""

# TIME component names - these are the possible time features that can be generated
# Based on original training.py class T structure (Y, M, W, D, Holiday)
TIME_COMPONENT_NAMES = [
    'Y_sin', 'Y_cos',  # Year sin/cos
    'M_sin', 'M_cos',  # Month sin/cos
    'W_sin', 'W_cos',  # Week sin/cos
    'D_sin', 'D_cos',  # Day sin/cos
    'Holiday'          # Holiday flag
]

# Mapping from time_info table column names to component names
# These match the actual column names in the time_info Supabase table
TIME_INFO_TO_COMPONENTS = {
    'jahr': ['Y_sin', 'Y_cos'],      # Year sin/cos components
    'monat': ['M_sin', 'M_cos'],     # Month sin/cos components
    'woche': ['W_sin', 'W_cos'],     # Week sin/cos components
    'tag': ['D_sin', 'D_cos'],       # Day sin/cos components
    'feiertag': ['Holiday']          # Holiday flag
}

# Default timestep configuration (matching MTS class in original training.py)
DEFAULT_TIMESTEP_CONFIG = {
    'eingabe': 97,           # MTS.I_N - input timesteps
    'ausgabe': 97,           # MTS.O_N - output timesteps
    'zeitschrittweite': 15,  # MTS.DELT - timestep width in minutes
    'offset': 0              # MTS.OFST - offset in minutes
}

# Columns to exclude when identifying data features (not actual data columns)
TIMESTAMP_COLUMN_NAMES = ['timestamp', 'utc', 'zeit', 'datetime', 'date', 'time']


def get_active_time_components(time_info: dict) -> list:
    """
    Get list of active TIME component names based on time_info configuration.

    Args:
        time_info: Dictionary with boolean flags for each time component type
                   e.g., {'jahreszeitlicheSinCosKomponente': True, ...}

    Returns:
        List of active time component names, e.g., ['Y_sin', 'Y_cos', 'W_sin', 'W_cos']
    """
    if not time_info:
        return []

    active_components = []
    for key, component_names in TIME_INFO_TO_COMPONENTS.items():
        if time_info.get(key, False):
            active_components.extend(component_names)

    return active_components


def calculate_time_deltas(zeitschritte: dict, n_averaging_levels: int = 12) -> list:
    """
    Calculate time deltas based on zeitschritte configuration.

    In original training.py, deltas are calculated as: delt * n for n in range(1, n_avg+1)
    where delt comes from zeitschrittweite.

    Args:
        zeitschritte: Dictionary with 'zeitschrittweite' key (timestep width in minutes)
        n_averaging_levels: Number of averaging levels (default 12 as in original)

    Returns:
        List of time deltas in minutes, e.g., [15, 30, 45, ..., 180] for 15-min timesteps
    """
    delt = float(zeitschritte.get('zeitschrittweite', DEFAULT_TIMESTEP_CONFIG['zeitschrittweite']))
    return [delt * n for n in range(1, n_averaging_levels + 1)]
