import pandas as pd
import numpy as np
import datetime
import copy
import statistics
from datetime import datetime as dat
from flask import Flask, request, jsonify
from flask_cors import CORS
from io import StringIO


app = Flask(__name__)
CORS(app)

@app.route('/api/erste-bearbeitung', methods=['POST'])
def erste_bearbeitung():
    try:
        # Get and validate request data
        file_content = request.form.get('fileContent')
        if not file_content:
            return jsonify({"error": "Keine Datei gefunden"}), 400

        # Get and convert numeric parameters with validation
        try:
            tss = float(request.form.get('tss', '0'))
            offset = float(request.form.get('offset', '0'))
            mode_input = request.form.get('mode', '')
            intrpl_max = float(request.form.get('intrplMax', '60'))
        except ValueError:
            return jsonify({"error": "TSS, Offset und Interpolationsmaximum müssen gültige Zahlen sein"}), 400

        # Validate parameter values
        if tss <= 0:
            return jsonify({"error": "TSS muss größer als 0 sein"}), 400
        if offset < 0:
            return jsonify({"error": "Offset muss größer oder gleich 0 sein"}), 400
        if mode_input == "intrpl" and intrpl_max <= 0:
            return jsonify({"error": "Interpolationsmaximum muss größer als 0 sein"}), 400
        if mode_input not in ["intrpl", "nearest", "nearest (mean)"]:
            return jsonify({"error": f"Ungültiger Modus: {mode_input}"}), 400

        print("\nParametri nakon konverzije:", f"tss={tss}, offset={offset}, mode={mode_input}, intrpl_max={intrpl_max}")

        # Ispis primljenih podataka
        print("\nPrimljeni podaci:")
        print("----------------")
        print(f"TSS: {tss}")
        print(f"Offset: {offset}")
        print(f"Mode: {mode_input}")
        print(f"IntrplMax: {intrpl_max}")
        print("\nPrvih 50 karaktera file content-a:")
        print(file_content[:200] if file_content else "Nema sadržaja")
        print("----------------\n")

    

        ###############################################################################
        # DATEN LADEN #################################################################
        ###############################################################################

        df = pd.read_csv(StringIO(file_content), delimiter=",", header=0)
        print("Zaglavlja kolona:", df.columns)

        # Pronađi vrednosnu kolonu (bilo koja kolona koja nije UTC)
        value_column = next((col for col in df.columns if col != 'UTC'), None)
        if not value_column:
            print("Greška: Nije pronađena vrednosna kolona")
            return jsonify({"error": "Keine Wertespalte gefunden"}), 400

        print(f"Koristi se vrednosna kolona: {value_column}")

        # VORBEREITUNG DER ROHDATEN ###################################################
        
        # Duplikate (Zeilen mit gleicher UTC) in den Rohdaten löschen
        df1 = df.drop_duplicates(["UTC"]).reset_index(drop = True)

        # Rohdaten nach UTC ordnen
        df2 = copy.deepcopy(df1)
        df2.sort_values(by = ["UTC"], inplace = True)

        # Reset des Indexes in den Rohdaten
        df3 = copy.deepcopy(df2)
        df3.reset_index(drop = True, inplace = True)

        # ZEITGRENZEN #################################################################

        # Zeitgrenzen in den Rohdaten
        try:
            if len(df3) == 0:
                print("Greška: DataFrame je prazan")
                return jsonify({"error": "Keine Daten gefunden"}), 400

            print(f"Prvi red podataka: {df3.iloc[0]}")
            print(f"Primer UTC vrednosti: {df3['UTC'][0]}")
            print(f"Primer {value_column} vrednosti: {df3[value_column][0]}")
            
            def is_numeric(value):
                try:
                    float(value)
                    return True
                except (ValueError, TypeError):
                    return False

            # Proveri da li su vrednosti numeričke
            non_numeric = df3[value_column].apply(lambda x: not is_numeric(x))
            if non_numeric.any():
                print(f"Pronađene ne-numeričke vrednosti u {value_column}:", 
                      df3[value_column][non_numeric].head())
            
            time_min_raw = datetime.datetime.strptime(df3["UTC"][0], '%Y-%m-%d %H:%M:%S')
            time_max_raw = datetime.datetime.strptime(df3["UTC"][len(df3)-1], '%Y-%m-%d %H:%M:%S')
            
            print(f"Vremenski opseg: {time_min_raw} do {time_max_raw}")
        except Exception as e:
            print(f"Greška pri obradi vremena: {str(e)}")
            return jsonify({"error": f"Fehler bei der Zeitverarbeitung: {str(e)}"}), 400

        ###############################################################################
        # EINGABEN ####################################################################
        ###############################################################################

        # Gewünschte Zeitschrittweite der aufbereiteten Daten [min]
        #tss = 3
        # Gewünschter Offset der aufbereiteten Daten [min]
        #offset = 1
        # Methode der Datenaufbereitung    
        #mode = "intrpl"
        """
            mean............Mittelwertbildung
            intrpl..........Lineare Interpolation
            .........Nähester Wert
                            Algorithmus nimmt den nächstgelegenen Messwert aus den
                            Rohdaten. Wenn zwei Messwerte der Rohdaten minimal und
                            gleich weit vom aktuellen Zeitpunkt des kontinuierlichen
                            Zeitstempels entfernt sind, wird der erste dieser Messwerte
                            für die aufbereiteten Daten verwendet.
            
            nearest (mean)..Algorithmus nimmt den nächstgelegenen Messwert aus den
                            Rohdaten. Wenn zwei Messwerte der Rohdaten minimal und
                            gleich weit vom aktuellen Zeitpunkt des kontinuierlichen
                            Zeitstempels entfernt sind, wird der Mittelwert dieser
                            Messwerte für die aufbereiteten Daten verwendet.

        """

        # Maximale Zeitschrittweite der linearen Interpolation [min]
        #intrpl_max = 3

        ###############################################################################
        # KONTINUIERLICHER ZEITSTEMPEL ################################################
        ###############################################################################

        # Offset der unteren Zeitgrenze in der Rohdaten
        offset_strt = datetime.timedelta(minutes=time_min_raw.minute,
                                       seconds=time_min_raw.second,
                                       microseconds=time_min_raw.microsecond)

        # Realer Offset in den aufbereiteten Daten [min]
        # Ensure positive offset within TSS range
        offset = abs(offset) % tss if offset >= 0 else 0

        # Untere Zeitgrenze in den aufbereiteten Daten
        time_min = time_min_raw - offset_strt
        if offset > 0:
            # Add offset to align with the requested time grid
            time_min += datetime.timedelta(minutes=offset)

        # Generate continuous timestamp
        time_list = []
        current_time = time_min
        
        while current_time <= time_max_raw:
            time_list.append(current_time)
            current_time += datetime.timedelta(minutes=tss)

        if not time_list:
            return jsonify({"error": "Keine gültigen Zeitpunkte generiert"}), 400

        print(f"Generisano {len(time_list)} vremenskih tačaka")
        print(f"Prvi timestamp: {time_list[0]}")
        print(f"Poslednji timestamp: {time_list[-1]}")

        ###############################################################################
        # AUFBEREREITUNG DER DATEN ####################################################
        ###############################################################################

        # Zähler für den Durchlauf der Rohdaten
        i_raw = 0

        # Initialisierung der Liste mit den aufbereiteten Messwerten
        value_list = []

        # METHODE: MITTELWERTBILDUNG ##################################################

        if mode_input == "mean":
            
            # Schleife durchläuft alle Zeitschritte des kontinuierlichen Zeitstempels
            for i in range(0, len(time_list)):
                        
                # Zeitgrenzen für die Mittelwertbildung (Untersuchungsraum)
                time_int_min = time_list[i]-datetime.timedelta(minutes = tss/2)
                time_int_max = time_list[i]+datetime.timedelta(minutes = tss/2)
                
                # Berücksichtigung angrenzender Untersuchungsräume
                if i > 0: i_raw -= 1
                if i > 0 and datetime.datetime.strptime(df3["UTC"][i_raw], '%Y-%m-%d %H:%M:%S') < time_int_min: i_raw += 1                           
                
                # Initialisierung der Liste mit den Messwerten im Untersuchungsraum
                value_int_list = []
                
                # Auflistung numerischer Messwerte im Untersuchungsraum
                while i_raw < len(df3) and datetime.datetime.strptime(df3["UTC"][i_raw], '%Y-%m-%d %H:%M:%S') <= time_int_max and datetime.datetime.strptime(df3["UTC"][i_raw], '%Y-%m-%d %H:%M:%S') >= time_int_min:
                    if is_numeric(df3[value_column][i_raw]):
                        value_int_list.append(float(df3[value_column][i_raw]))
                    i_raw += 1
                    
                # Mittelwertbildung über die numerischen Messwerte im Untersuchungsraum
                if len(value_int_list) > 0:
                    value_list.append(statistics.mean(value_int_list))
                else:
                    value_list.append("nan")

        # METHODE: LINEARE INTERPOLATION ##############################################

        elif mode_input == "intrpl":
            
            # Zähler für den Durchlauf der Rohdaten
            i_raw = 0
            
            # Richtung des Schleifendurchlaufs
            direct = 1
            """
                direct = 1...Schleife läuft vorwärts
                direct = -1..Schleife läuft rückwärts
            """
            
            
            # Schleife durchläuft alle Zeitschritte des kontinuierlichen Zeitstempels
            for i in range(0, len(time_list)):
                        
                # Schleife durchläuft die Rohdaten von vorne bis hinten zur Auffindung des nachfolgenden Wertes
                if direct == 1:
                    
                    loop = True
                    while i_raw < len(df3) and loop == True:
                        
                        # Der aktuelle Zeitpunkt in den Rohdaten liegt nach dem aktuellen Zeitpunkt
                        # im kontinuierlichen Zeitstempel oder ist mit diesem identisch.
                        if datetime.datetime.strptime(df3["UTC"][i_raw], '%Y-%m-%d %H:%M:%S') >= time_list[i]:
                            
                            # Der aktuelle Zeitpunkt in den Rohdaten liegt nach dem aktuellen Zeitpunkt
                            # im kontinuierlichen Zeitstempel oder ist mit diesem identisch und der 
                            # dazugehörige Messwert ist numerisch
                            if is_numeric(df3[value_column][i_raw]):
                                
                                # UTC und Messwert vom nachfolgenden Wert übernehmen 
                                time_next = datetime.datetime.strptime(df3["UTC"][i_raw], '%Y-%m-%d %H:%M:%S')
                                value_next = float(df3[value_column][i_raw])
                                loop = False
                                
                            else:
                                
                                # Zähler aktuallisieren, wenn Messwert nicht numerisch
                                i_raw += 1
                        
                        else:
                            
                            # Zähler aktuallisieren, wenn der aktuelle Zeitpunkt in den Rohdaten vor dem aktuellen
                            # Zeitpunkt im kontinuierlichen Zeitstempel liegt.
                            i_raw += 1
                    
                    # Die gesamten Rohdaten wurden durchlaufen und es wurde kein gültiger Messwert gefunden
                    if i_raw+1 > len(df3):
                        value_list.append("nan")
                        
                        # Zähler für die Rohdaten auf Null setzen und Schleifenrichtung festlegen
                        i_raw = 0
                        direct = 1
                    else:
                        
                        # Schleifenrichtung umdrehen
                        direct = -1
                
                # Finden des vorangegangenen Wertes
                if direct == -1:
                    
                    loop = True
                    while i_raw >= 0 and loop == True:
                        
                        # Der aktuelle Zeitpunkt in den Rohdaten liegt vor dem aktuellen Zeitpunkt
                        # im kontinuierlichen Zeitstempel oder ist mit diesem identisch.
                        if datetime.datetime.strptime(df3["UTC"][i_raw], '%Y-%m-%d %H:%M:%S') <= time_list[i]:
                            
                            # Der aktuelle Zeitpunkt in den Rohdaten liegt vor dem aktuellen Zeitpunkt
                            # im kontinuierlichen Zeitstempel oder ist mit diesem identisch und der 
                            # dazugehörige Messwert ist numerisch
                            if is_numeric(df3[value_column][i_raw]):
                                
                                # UTC und Messwert vom vorangegangenen Wert übernehmen 
                                time_prior = datetime.datetime.strptime(df3["UTC"][i_raw], '%Y-%m-%d %H:%M:%S')
                                value_prior = float(df3[value_column][i_raw])
                                loop = False
                            else:
                                # Zähler aktuallisieren, wenn Messwert nicht numerisch
                                i_raw -= 1
                        else:
                            # Zähler aktuallisieren, wenn der aktuelle Zeitpunkt in den Rohdaten nach dem aktuellen
                            # Zeitpunkt im kontinuierlichen Zeitstempel liegt.
                            i_raw -= 1
                    
                    # Die gesamten Rohdaten wurden durchlaufen und es wurde kein gültiger Messwert gefunden
                    if i_raw < 0:
                        value_list.append("nan")
                        
                        # Zähler für die Rohdaten auf Null setzen und Schleifenrichtung festlegen
                        i_raw = 0
                        direct = 1
                        
                    # Es wurde ein gültige Messwerte vor dem aktuellen Zeitpunkt im kontinuierlichen Zeitstempel und nach diesem gefunden
                    else:
                        delta_time = time_next-time_prior
                        
                        #print(str(time_list[i])+": "+str(time_prior)+" ; "+str(value_prior)+" ; "+str(time_next)+" ; "+str(value_next)+" ; "+str(delta_time))
                
                        # Zeitabstand zwischen den entsprechenden Messwerten in den Rohdaten [sec]
                        delta_time_sec = delta_time.total_seconds()
                        delta_value = value_prior-value_next
                
                        # Zeitpunkte fallen zusammen oder gleichbleibender Messwert - Keine lineare Interpolation notwendig
                        if delta_time_sec == 0 or (delta_value == 0 and delta_time_sec <= intrpl_max*60):
                            value_list.append(value_prior)
                        
                        # Zeitabstand zu groß - Keine lineare Interpolation möglich
                        elif delta_time_sec > intrpl_max*60:
                            value_list.append("nan")
                        
                        # Lineare Interpolation
                        else:
                            delta_time_prior_sec = (time_list[i]-time_prior).total_seconds()   
                            value_list.append(value_prior-delta_value/delta_time_sec*delta_time_prior_sec)
                        
                            
                        direct = 1
                    

        # METHODE: ZEITLICH NÄCHSTLIEGENDER MESSWERT ##################################       

        elif mode_input == "nearest" or mode_input == "nearest (mean)":
            i_raw = 0  # Reset index counter
            
            # Schleife durchläuft alle Zeitschritte des kontinuierlichen Zeitstempels
            for i in range(0, len(time_list)):
                try:
                    # Zeitgrenzen für die Untersuchung (Untersuchungsraum)
                    time_int_min = time_list[i] - datetime.timedelta(minutes=tss/2)
                    time_int_max = time_list[i] + datetime.timedelta(minutes=tss/2)
                    
                    # Find values within the time window
                    value_int_list = []
                    delta_time_int_list = []
                    
                    # Scan through data points within the time window
                    while i_raw < len(df3):
                        current_time = datetime.datetime.strptime(df3["UTC"][i_raw], '%Y-%m-%d %H:%M:%S')
                        
                        if current_time > time_int_max:
                            break
                            
                        if current_time >= time_int_min:
                            if is_numeric(df3[value_column][i_raw]):
                                value_int_list.append(float(df3[value_column][i_raw]))
                                delta_time_int_list.append(abs((time_list[i] - current_time).total_seconds()))
                        
                        i_raw += 1
                    
                    # If we've moved past the window, back up to catch potential values in the next window
                    if i_raw > 0:
                        i_raw -= 1
                    
                    # Process values based on mode
                    if value_int_list:
                        if mode_input == "nearest":
                            # Find the value with minimum time difference
                            min_time = min(delta_time_int_list)
                            min_idx = delta_time_int_list.index(min_time)
                            value_list.append(value_int_list[min_idx])
                        else:  # nearest (mean)
                            # Find all values with the minimum time difference
                            min_time = min(delta_time_int_list)
                            nearest_values = [
                                value_int_list[idx]
                                for idx, delta in enumerate(delta_time_int_list)
                                if abs(delta - min_time) < 0.001  # Small tolerance for float comparison
                            ]
                            value_list.append(statistics.mean(nearest_values))
                    else:
                        value_list.append("nan")
                        
                except Exception as e:
                    print(f"Error processing time step {i}: {str(e)}")
                    value_list.append("nan")
                    
        # DATENRAHMEN MIT DEN AUFBEREITETEN DATEN #####################################

        # Ispisivanje dužina lista za dijagnostiku
        print(f"Dužina time_list: {len(time_list)}")
        print(f"Dužina value_list: {len(value_list)}")

        df4 = pd.DataFrame({"UTC": time_list, value_column: value_list}) 

        # Formatiramo UTC kolonu u željeni format
        df4['UTC'] = df4['UTC'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

        return jsonify(df4.to_dict(orient="records")), 200

    except Exception as e:
        print(f"Greška: {str(e)}")
        return jsonify({"error": str(e)}), 400   


if __name__ == '__main__':
    print("Starting Flask server on port 5005...")
    app.run(port=5006, debug=True)   