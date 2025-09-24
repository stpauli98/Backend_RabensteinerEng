from flask import Flask, request, jsonify
import pandas as pd
from io import StringIO
from flask_cors import CORS
import json
import pytz

app = Flask(__name__)
CORS(app)


def validate_delimiter(file_content, delimiter):
    """Überprüft, ob der Delimiter für den angegebenen CSV-Inhalt korrekt ist"""
    try:
        # Lese die ersten paar Zeilen
        sample_content = '\n'.join(file_content.split('\n')[:5])
        
        # Versuche, den Inhalt mit dem angegebenen Delimiter zu parsen
        df = pd.read_csv(StringIO(sample_content), delimiter=delimiter, nrows=1)
        
        # Wenn wir nur eine Spalte haben, ist der Delimiter wahrscheinlich falsch
        if len(df.columns) <= 1:
            detected_delimiter = None
            # Versuche, den richtigen Delimiter zu erkennen
            for test_delimiter in [',', ';', '\t', '|']:
                try:
                    test_df = pd.read_csv(StringIO(sample_content), delimiter=test_delimiter, nrows=1)
                    if len(test_df.columns) > 1:
                        detected_delimiter = test_delimiter
                        break
                except:
                    continue
                    
            if detected_delimiter:
                return False, f"Falscher Delimiter. Die CSV-Datei verwendet '{detected_delimiter}' als Delimiter."
            else:
                return False, "Falscher Delimiter. Wir konnten den richtigen Delimiter nicht erkennen."
                
        return True, None
        
    except Exception as e:
        return False, f"Fehler bei der Delimiter-Überprüfung: {str(e)}"


def load_raw_data(files, delimiter, timezone):
    print(f"Starte load_raw_data mit Delimiter: {delimiter}")
    df = pd.DataFrame()
    
    for file in files:
        print(f"Verarbeite Datei: {file.filename}")
        file_content = file.read().decode('utf-8')
        
        # Überprüfe den Delimiter
        print(f"Überprüfe Delimiter: {delimiter}")
        is_valid, error_message = validate_delimiter(file_content, delimiter)
        if not is_valid:
            print(f"Delimiter-Überprüfung fehlgeschlagen: {error_message}")
            return None, error_message
            
        # Setze die Dateiposition nach dem Lesen zurück
        file.seek(0)
        
        csv_data = StringIO(file_content)
        
        try:
            # Versuche, nur die ersten paar Zeilen zu lesen
            print("Versuche, die ersten 5 Zeilen zu lesen...")
            sample_df = pd.read_csv(csv_data, delimiter=delimiter, nrows=5)
            print(f"Spalten gefunden: {sample_df.columns.tolist()}")
            
            # Wenn wir nur eine Spalte haben, ist der Delimiter wahrscheinlich falsch
            if sample_df.shape[1] == 1:
                print(f"Nur eine Spalte gefunden, Delimiter '{delimiter}' ist wahrscheinlich falsch")
                return None, f"Das eingegebene Trennzeichen '{delimiter}' stimmt nicht mit dem Trennzeichen in der Datei überein. Überprüfen Sie das Trennzeichen und versuchen Sie es erneut."

            # Setze die Dateiposition auf den Anfang für das vollständige Lesen
            csv_data.seek(0)
            
            # Lese die Datei in Teilen für große Dateien
            print("Lese die Datei in Teilen...")
            chunk_size = 50000  # Anpassen Sie diesen Wert nach Bedarf
            chunks = []
            
            for chunk in pd.read_csv(csv_data, delimiter=delimiter, chunksize=chunk_size):
                chunk = chunk.dropna(axis=1, how='all')  # Entferne leere Spalten
                chunks.append(chunk)
                print(f"Gelesener Chunk-Größe {len(chunk)}")
            
            temp_df = pd.concat(chunks, ignore_index=True)
            print(f"Datei erfolgreich verarbeitet. Größe: {temp_df.shape}")
            
            df = pd.concat([df, temp_df], ignore_index=True)
            
        except pd.errors.ParserError as e:
            print(f"Parser-Fehler: {str(e)}")
            return None, f"Fehler beim Parsen der CSV-Datei: {str(e)}"
        except Exception as e:
            print(f"Unerwarteter Fehler: {str(e)}")
            return None, f"Fehler bei der Verarbeitung der Datei: {str(e)}"
    
    # Speichere den gesamten DataFrame für die spätere Verarbeitung
    print(f"Gesamtzeilen im DataFrame: {len(df)}")
    return df, None


def detect_column_type(column_name, sample_data):
    """Detektiert den Spaltentyp basierend auf Namen und Daten"""
    column_name_lower = column_name.lower()
    
    # Datum/Zeit Erkennung basierend auf Spaltennamen
    if any(word in column_name_lower for word in ['datum', 'date', 'zeit', 'time', 'uhrzeit']):
        # Prüfe ob es ein Datums- oder Zeitformat ist
        try:
            sample = str(sample_data.iloc[0])
            if any(c in sample for c in [':', '.']):
                if any(c in sample for c in ['/', '-', '.']):
                    return 'date'  # Enthält sowohl Datum als auch Zeit
                return 'time'  # Nur Zeit
            return 'date'  # Nur Datum
        except:
            return 'unknown'
    
    # Wert Erkennung
    if any(word in column_name_lower for word in ['wert', 'messwert', 'value', 'leistung', 'durchfluss', 'volumen']):
        try:
            # Versuche zu prüfen ob es sich um numerische Werte handelt
            pd.to_numeric(sample_data.iloc[0])
            return 'value'
        except:
            pass
    
    # Versuche numerische Konvertierung
    try:
        pd.to_numeric(sample_data.iloc[0])
        return 'value'
    except:
        return 'unknown'


def create_column_mapping(df, selected_columns):
    """Erstellt eine Spaltenzuordnung basierend auf den erkannten Typen"""
    print("\nErkenne Spaltentypen...")
    
    # Erkenne Spaltentypen
    column_types = {}
    for col in df.columns:
        column_types[col] = detect_column_type(col, df[col])
    
    print(f"\nErkannte Spaltentypen: {column_types}")
    
    # Erstelle die Spaltenzuordnung basierend auf den erkannten Typen
    mapping = {}
    value_column_found = False
    
    print("\nSelected columns received:", selected_columns)
    print("Column types:", column_types)
    
    # Prvo mapiraj datum i vreme
    for col in df.columns:
        if column_types.get(col) == 'date':
            mapping[col] = 'Datum'
            print(f"Mapping date column: {col} -> Datum")
        elif column_types.get(col) == 'time':
            mapping[col] = 'Zeit'
            print(f"Mapping time column: {col} -> Zeit")
    
    # Zatim mapiraj vrednosnu kolonu
    for col in df.columns:
        if column_types.get(col) == 'value':
            print(f"Checking value column: {col}")
            # Ako je izabrana column3, koristi nju
            if selected_columns.get('column3'):
                print(f"Column3 is selected: {selected_columns['column3']}")
                if col == selected_columns['column3']:
                    print(f"Using column3: {col} as Wert")
                    mapping[col] = 'Wert'
                    value_column_found = True
                    break
            # Ako je izabrana column2 i nije vreme, koristi nju
            elif selected_columns.get('column2'):
                print(f"Column2 is selected: {selected_columns['column2']}")
                if col == selected_columns['column2'] and column_types.get(selected_columns['column2']) != 'time':
                    print(f"Using column2: {col} as Wert")
                    mapping[col] = 'Wert'
                    value_column_found = True
                    break
    
    # Ako nije pronađena vrednosna kolona, uzmi prvu dostupnu
    if not value_column_found:
        print("No value column selected, using first available value column")
        for col in df.columns:
            if column_types.get(col) == 'value':
                mapping[col] = 'Wert'
                print(f"Using first value column: {col} as Wert")
                value_column_found = True
                break

    print("\nErstellte Spaltenzuordnung:", mapping)
    
    # Benenne die Spalten um
    df = df.rename(columns=mapping)
    print("Spalten nach Umbenennung:", df.columns.tolist())
    
    return mapping


# Definiere die unterstützten Datumsformate
date_formats = [
    "%Y-%m-%d %H:%M:%S",     # 2022-01-01 00:00:00
    "%d.%m.%Y %H:%M:%S",     # 31.12.2022 00:00:00
    "%Y-%m-%d %H:%M",        # 2022-01-01 00:00
    "%d.%m.%Y %H:%M",        # 31.12.2022 00:00
    "%Y-%m-%d",              # 2022-01-01
    "%d.%m.%Y",              # 31.12.2022
    "%d/%m/%Y %H:%M:%S",     # 31/12/2022 00:00:00
    "%Y/%m/%d %H:%M:%S",     # 2022/12/31 00:00:00
    "%d-%m-%Y %H:%M:%S",     # 31-12-2022 00:00:00
]


@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        file = request.files['file']
        delimiter = request.form.get('delimiter', ',')
        timezone = request.form.get('timezone', 'UTC')
        selected_columns = json.loads(request.form.get('selected_columns', '{}'))
        value_column_name = request.form.get('valueColumnName', 'Wert')
        
        print("=== Starte Datei-Upload ===")
        print(f"Empfangene Datei: {file.filename}")
        print(f"Empfangener Delimiter: {delimiter}")
        print(f"Empfangene dropdown_count: {request.form.get('dropdown_count', '2')}")
        print(f"Empfangene Zeitzone: {timezone}")
        print(f"Empfangene selected_columns: {selected_columns}")
        print(f"Empfangener Wertname: {value_column_name}")

        # Überprüfe die Eingaben
        if not file:
            print("Fehler: Keine Datei bereitgestellt")
            return jsonify({"error": "Keine Datei bereitgestellt"}), 400

        if not delimiter:
            print("Fehler: Delimiter fehlt")
            return jsonify({"error": "Delimiter fehlt"}), 400

        # Überprüfe, ob mindestens eine Datums- oder Wertspalte ausgewählt wurde
        if not selected_columns.get("column1") and not selected_columns.get("column2"):
            print("Fehler: Keine Spalten ausgewählt")
            return jsonify({"error": "Bitte wählen Sie mindestens eine Datums- und eine Wertspalte aus"}), 400

        # Lade die Daten
        df, error = load_raw_data([file], delimiter, timezone)  
        
        if error:
            print(f"Fehler in load_raw_data: {error}")
            return jsonify({"error": error}), 400

        if df is None or df.empty:
            print("Fehler: DataFrame ist leer")
            return jsonify({"error": "Keine Daten in der bereitgestellten Datei gefunden."}), 400

        print(f"Anfängliche DataFrame-Spalten: {df.columns.tolist()}")

        # Erstelle Spaltenzuordnung
        column_mapping = create_column_mapping(df, selected_columns)
        
        # Überprüfe, ob die ausgewählten Spalten im DataFrame vorhanden sind
        for col_name in column_mapping.keys():
            if col_name not in df.columns:
                print(f"Fehler: Spalte {col_name} nicht gefunden. Verfügbare Spalten: {df.columns.tolist()}")
                return jsonify({"error": f"Die ausgewählte Spalte '{col_name}' wurde in der Datei nicht gefunden. Verfügbare Spalten: {', '.join(df.columns.tolist())}"}), 400

        # Überprüfe speziell die Datumsspalte
        date_col = selected_columns.get("column1")
        if not date_col:
            print("Fehler: Keine Datumsspalte ausgewählt")
            return jsonify({"error": "Bitte wählen Sie eine Datumsspalte aus"}), 400

        # Wenn Zeit ausgewählt wurde, überprüfe ob sie existiert
        time_col = selected_columns.get("column2")
        if time_col and column_mapping.get(time_col) == "Zeit" and time_col not in df.columns:
            print(f"Fehler: Zeitspalte {time_col} nicht gefunden")
            return jsonify({"error": f"Die ausgewählte Zeitspalte '{time_col}' wurde in der Datei nicht gefunden"}), 400

        # Umbenenne die Spalten
        df = df.rename(columns=column_mapping)
        print(f"Spalten nach Umbenennung: {df.columns.tolist()}")
        
        # Überprüfe, ob alle benötigten Spalten vorhanden sind
        if request.form.get('dropdown_count', '2') == '3':
            needed_columns = ["Datum", "Zeit", "Wert"]
        else:
            if "Zeit" in column_mapping.values():
                needed_columns = ["Datum", "Zeit", "Wert"]
            else:
                needed_columns = ["Datum", "Wert"]

        missing_cols = [col for col in needed_columns if col not in df.columns]
        if missing_cols:
            print(f"Fehlende Spalten: {missing_cols}")
            print(f"Verfügbare Spalten: {df.columns.tolist()}")
            return jsonify({"error": f"Fehlende erforderliche Spalten: {', '.join(missing_cols)}"}), 400

        # Behalte nur die benötigten Spalten
        df = df[needed_columns]
        print(f"\nDatentypen vor der Konvertierung:")
        print(df.dtypes)
        print(f"\nBeispieldaten vor der Konvertierung:\n{df.head().to_dict()}")
        
        try:
            print("\nVersuche, in UTC zu konvertieren")
            
            def format_time(time_val):
                """Konvertiere die Uhrzeit in das Format HH:MM:SS"""
                if isinstance(time_val, str):
                    time_val = time_val.strip()
                    
                    if ':' in time_val:
                        parts = time_val.split(':')
                        if len(parts) == 3:
                            return time_val
                        if len(parts) == 2:
                            return f"{parts[0]}:{parts[1]}:00"
                    
                    if '.' in time_val:
                        parts = time_val.split('.')
                        if len(parts) == 2:
                            return f"{parts[0]}:{parts[1]}:00"
                        if len(parts) == 3:
                            return f"{parts[0]}:{parts[1]}:{parts[2]}"
                
                elif isinstance(time_val, (int, float)):
                    return f"{int(time_val):02d}:00:00"
                
                raise ValueError(f"Ungültiges Zeitformat: {time_val}")
            
            try:
                print("Versuche flexibles Datum-Parsing...")
                
                # Säubere die Datumsspalte
                df["Datum"] = df["Datum"].str.strip()
                print(f"Datumswerte: {df['Datum'].unique()}")
                
                # Liste der zu versuchenden Datumsformate
                date_formats = [
                    "%Y-%m-%d %H:%M:%S",     # 2022-01-01 00:00:00
                    "%d.%m.%Y %H:%M:%S",     # 31.12.2022 00:00:00
                    "%Y-%m-%d %H:%M",        # 2022-01-01 00:00
                    "%d.%m.%Y %H:%M",        # 31.12.2022 00:00
                    "%Y-%m-%d",              # 2022-01-01
                    "%d.%m.%Y",              # 31.12.2022
                    "%d/%m/%Y %H:%M:%S",     # 31/12/2022 00:00:00
                    "%Y/%m/%d %H:%M:%S",     # 2022/12/31 00:00:00
                    "%d-%m-%Y %H:%M:%S",     # 31-12-2022 00:00:00
                ]
                
                success = False
                for date_format in date_formats:
                    try:
                        print(f"Versuche Format: {date_format}")
                        df["UTC"] = pd.to_datetime(df["Datum"], format=date_format, errors='raise')
                        success = True
                        print(f"Format {date_format} erfolgreich!")
                        break
                    except ValueError:
                        continue
                
                if not success:
                    print("Kein Format hat funktioniert, versuche automatische Erkennung...")
                    df["UTC"] = pd.to_datetime(df["Datum"], errors='raise')
                
                # Prüfe auf NaT (Not a Time) Werte
                if df["UTC"].isna().any():
                    print("Einige Datumsangaben konnten nicht geparst werden")
                    print(f"Problematische Werte: {df.loc[df['UTC'].isna(), 'Datum'].unique()}")
                    raise ValueError("Ungültiges Datumsformat gefunden")
                    
                print(f"Datum erfolgreich geparst: {df['UTC'].iloc[0]} (Beispiel)")

                # Lokalisiere die Uhrzeit in UTC
                print("Lokalisiere die Uhrzeit in UTC...")
                df["UTC"] = df["UTC"].dt.tz_localize('UTC')
                
                # Konvertiere in die gewünschte Zeitzone, wenn nicht UTC
                if timezone != 'UTC':
                    print(f"Konvertiere in {timezone}...")
                    df["UTC"] = df["UTC"].dt.tz_convert(timezone)
                
                # Formatiere die Uhrzeit mit Zeitzone
                df["UTC"] = df["UTC"].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
                
                # Behalte nur die benötigten Spalten
                df = df[["UTC", "Wert"]]
                df = df.rename(columns={"Wert": value_column_name})
                print("\nUTC-Konvertierung erfolgreich")
                print(f"Endgültige Beispieldaten:\n{df.head().to_dict()}")
                
                # Konvertiere DataFrame in das erwartete Format
                headers = df.columns.tolist()
                data = df.values.tolist()
                formatted_data = [headers] + data  # Kombiniere Headers und Daten
                
                return jsonify({
                    "success": True, 
                    "data": formatted_data,
                    "fullData": formatted_data
                }), 200
                
            except Exception as e:
                print(f"Flexibles Parsing fehlgeschlagen: {str(e)}")
                print(f"Datumswerte: {df['Datum'].unique()}")
                return jsonify({"error": f"Fehler beim Parsen des Datums: {str(e)}"}), 400
        except Exception as e:
            print(f"\nFehler während der UTC-Konvertierung: {str(e)}")
            print(f"Datentypen:")
            print(df.dtypes)
            print(f"Beispieldaten:\n{df.head().to_dict()}")
            return jsonify({"error": f"Fehler bei der Konvertierung der Uhrzeit: {str(e)}"}), 400

    except Exception as e:
        print(f"Unerwarteter Fehler: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
    
if __name__ == "__main__":
    app.run(debug=True, port=5001)
