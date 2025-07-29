import sys
import os
import uuid

# Add the parent directory to the path so we can import from my_backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supabase_client import save_time_info

def example_save_time_info():
    """
    Example demonstrating how to save time information to the database
    """
    # Create a new session ID (or use an existing one)
    session_id = str(uuid.uuid4())
    
    # Create a sample time_info dictionary with all fields
    time_info = {
        # Vremenske komponente
        "jahr": True,
        "woche": True,
        "monat": True,
        "feiertag": False,
        "lokalzeit": True,
        
        # Lokalizacija
        "land": "Deutschland",
        "zeitzone": "Europe/Berlin",
        
        # Napredne opcije
        "detaillierteBerechnung": True,
        "datenform": "Zeit Horizont",
        
        # Vremenski horizont
        "zeithorizontStart": 0,
        "zeithorizontEnd": 24,
        
        # Skaliranje
        "skalierung": "ja",
        "skalierungMin": 0.0,
        "skalierungMax": 100.0
    }
    
    # Save the time information
    success = save_time_info(session_id, time_info)
    
    if success:
        print(f"Successfully saved time information for session {session_id}")
    else:
        print(f"Failed to save time information for session {session_id}")
    
    return session_id

if __name__ == "__main__":
    example_save_time_info()
