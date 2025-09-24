import pandas as pd
import math
from datetime import datetime as dat
from flask import Flask, request, jsonify
from flask_cors import CORS
from io import StringIO
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/api/zweite-bearbeitung', methods=['POST'])
def zweite_bearbeitung():
    try:
        # Dobijanje podataka iz zahtjeva
        file_content = request.form.get('file')
        EL0 = request.form.get('radioValueNull')
        ELNN = request.form.get('radioValueNotNull')
        EQ_MAX = float(request.form.get('eqMax'))
        CHG_MAX = float(request.form.get('chgMax'))
        LG_MAX = float(request.form.get('lgMax'))
        GAP_MAX = float(request.form.get('gapMax'))

       # Ispis primljenih podataka
        print("\nPrimljeni podaci:")
        print("----------------")
        print(f"EQ_MAX: {EQ_MAX}")
        print(f"EL0: {EL0}")
        print(f"ELNN: {ELNN}")
        print(f"CHG_MAX: {CHG_MAX}")
        print(f"LG_MAX: {LG_MAX}")
        print(f"GAP_MAX: {GAP_MAX}")
        print("\nPrvih 50 karaktera file content-a:")
        print(file_content[:200] if file_content else "Nema sadržaja")
        print("----------------\n")

        if EL0 == "ja":
            EL0 = 1
        else:
            EL0 = 0

        if ELNN == "ja":
            ELNN = 1
        else:
            ELNN = 0

        print(f"EL0: {EL0}")
        print(f"ELNN: {ELNN}")


        ##############################################################################
        # DATEN LADEN #################################################################
        ##############################################################################

        df = pd.read_csv(StringIO(file_content), delimiter = "," , header = 0)
        print("\nNakon učitavanja:")
        print(df.head())
        
        # Pretpostavljamo da je prva kolona vrijeme, a druga podaci
        time_column = df.columns[0]  # Prva kolona je vrijeme
        data_column = df.columns[1]  # Druga kolona su podaci
        print(f"\nKolona sa vremenom: {time_column}")
        print(f"Kolona sa podacima: {data_column}")
        
        # Konvertujemo kolonu sa podacima u numerički format
        df[data_column] = pd.to_numeric(df[data_column], errors='coerce')
        print("\nNakon konverzije u numerički format:")
        print(df.head())

        UTC_fmt = "%Y-%m-%d %H:%M:%S"

        ##############################################################################
        # ELIMINIERUNG VON MESSAUSFÄLLEN (GLEICHBLEIBENDE MESSWERTE) ##################
        ##############################################################################

        if "EQ_MAX" in locals():
            print("\nPočinjem EQ_MAX obradu...")
            print(f"EQ_MAX vrednost: {EQ_MAX}")
            
            # Status des Identifikationsrahmens (frm...frame)
            frm = 0
            """
                0...Im aktuellen Zeitschritt ist kein Identifikationsrahmen für
                    gleichbleibende Messwerte offen
                1...Im aktuellen Zeitschritt ist ein Identifikationsrahmen für 
                    gleichbleibende Messwerte offen
            """
            
            # Durchlauf des gesamten Datenrahmens             
            for i in range (1, len(df)):
                
                # Aktueller Messwert ist gleich dem letzen Messwert und 
                # Identifikationsrahmen ist geschlossen → Identifikationsrahmen wird
                # geöffnet
                if df.iloc[i-1][data_column] == df.iloc[i][data_column] and frm == 0:           
                    
                    idx_strt = i-1
                    frm = 1
                
                # Aktueller Messwert ist ungleich dem letzten Messwert und
                # Identifikationsrahmen ist offen → Identifikationsrahmen wird
                # geschlossen und ausgewertet
                elif df.iloc[i-1][data_column] != df.iloc[i][data_column] and frm == 1:
                    
                    idx_end = i-1
                    
                    # Länge des Identifikationsrahmens [min]
                    frm_width = (dat.strptime(df.iloc[idx_end][time_column], UTC_fmt)-\
                                dat.strptime(df.iloc[idx_strt][time_column], UTC_fmt)).total_seconds()/60
                    
                    # Der Identifikationsrahmen ist zu groß, so dass eine Messlücke
                    # eingefügt werden muss
                    if frm_width >= EQ_MAX:
                        for i_frm in range (idx_strt, idx_end+1):
                            df.at[i_frm, data_column] = np.nan
                    frm = 0
        
                # Ende des Datensatzes ist erreicht und Identifikationsrahmen ist offen
                elif i == len(df)-1 and frm == 1: 
                    idx_end = i
                    
                    # Länge des Identifikationsrahmens [min]
                    frm_width = (dat.strptime(df.iloc[idx_end][time_column], UTC_fmt)-\
                                dat.strptime(df.iloc[idx_strt][time_column], UTC_fmt)).total_seconds()/60
                    
                    # Der Identifikationsrahmen ist zu groß, so dass eine Messlücke
                    # eingefügt werden muss
                    if frm_width >= EQ_MAX:
                        for i_frm in range (idx_strt, idx_end+1):
                            df.at[i_frm, data_column] = np.nan
                            
        print("\nNakon EQ_MAX obrade:")
        print(df.head())

        ##############################################################################
        # ELIMINIERUNG VON NULLWERTEN #################################################
        ##############################################################################

        if EL0 == 1:
            print("\nPočinjem EL0 obradu...")
            
            # Durchlauf des gesamten Datenrahmens
            for i in range (0, len(df)):  
                if df.iloc[i][data_column] == 0:
                    df.at[i, data_column] = np.nan

        print("\nNakon EL0 obrade:")
        print(df.head())

        ##############################################################################
        # ELIMINIERUNG VON NICHT NUMERISCHEN WERTEN ###################################
        ##############################################################################

        if ELNN == 1:
            print("\nPočinjem ELNN obradu...")
            
            # Durchlauf des gesamten Datenrahmens
            for i in range (0, len(df)):  
                try:
                    # Pokušaj konverzije u float
                    float_value = float(df.iloc[i][data_column])
                    # Ako je NaN, ostavi ga kao NaN
                    if pd.isna(float_value):
                        df.at[i, data_column] = np.nan
                except (ValueError, TypeError):
                    # Ako konverzija ne uspije, postavi na NaN
                    df.at[i, data_column] = np.nan      

            print("\nNakon ELNN obrade:")
            print(df.head())

        ##############################################################################
        # ELIMINIERUNG VON AUSREISSERN ################################################
        ##############################################################################

        if "CHG_MAX" and "LG_MAX" in locals():
            print("\nPočinjem CHG_MAX i LG_MAX obradu...")
            print(f"CHG_MAX vrednost: {CHG_MAX}")
            print(f"LG_MAX vrednost: {LG_MAX}")
        
            # Status des Identifikationsrahmens (frm...frame)
            frm = 0
            """
                0...Im aktuellen Zeitschritt ist kein Identifikationsrahmen für
                    Ausreisser offen
                1...Im aktuellen Zeitschritt ist ein Identifikationsrahmen für 
                    Ausreisser offen
            """
            
            # Durchlauf des gesamten Datenrahmens                         
            for i in range (1, len(df)):  
                
                # nan im aktuellen Zeitschritt und Identifikationsrahmen ist im
                # aktuellen Zeitschritt nicht offen
                if pd.isna(df.iloc[i][data_column]) and frm == 0:
                    pass
                
                # nan im aktuellen Zeitschritt und Identifikationsrahmen ist im
                # aktuellen Zeitschritt offen → Identifikationsrahmen wird geschlossen
                # und ausgewertet
                elif pd.isna(df.iloc[i][data_column]) and frm == 1:
                    idx_end = i-1
        
                    for i_frm in range (idx_strt, idx_end+1):
                        df.at[i_frm, data_column] = np.nan

                    # Identifikationsrahmen wird geschlossen
                    frm = 0
                
                # nan im letzten Zeitschritt
                elif pd.isna(df.iloc[i-1][data_column]):
                    pass
                
                # Kein nan im letzten und aktuellen Zeitschritt
                else:
                    # Änderung des Messwertes im aktuellen Zeitschritt
                    chg = abs(df.iloc[i][data_column] - df.iloc[i-1][data_column])
                    
                    # Zeitschrittweite vom letzten zum aktuellen Zeitschritt [min]
                    t = (dat.strptime(df.iloc[i][time_column], UTC_fmt)-\
                        dat.strptime(df.iloc[i-1][time_column], UTC_fmt)).total_seconds()/60
                    
                    # Änderung im aktuellen Zeitschritt ist zu groß und
                    # Identifikationsrahmen ist geschlossen → Identifikationsrahmen
                    # wird geöffnet
                    if chg/t > CHG_MAX and frm == 0:
                        idx_strt = i
                        frm = 1
                        
                    # Änderung im aktuellen Zeitschritt ist zu groß und
                    # Identifikationsrahmen ist offen → nan einfügen
                    elif chg/t > CHG_MAX and frm == 1:
                        idx_end = i-1
                        
                        for i_frm in range (idx_strt, idx_end+1):
                            df.at[i_frm, data_column] = np.nan
                        
                        # Identifikationsrahmen wird geschlossen
                        frm = 0
                    
                    # Identifikationsrahmen ist offen und die maximale Breite des 
                    # Identifikationsrahmens wurde erreicht → Identifikationsrahmen
                    # wird geschlossen
                    elif frm == 1 and (dat.strptime(df.iloc[i][time_column], \
                                                    UTC_fmt)-\
                                    dat.strptime(df.iloc[idx_strt][time_column], \
                                                    UTC_fmt)).total_seconds()/60 > LG_MAX:
                        frm = 0
                            
        print("\nNakon CHG_MAX i LG_MAX obrade:")
        print(df.head())

        ##############################################################################
        # SCHLIESSEN VON MESSLÜCKEN ###################################################
        ##############################################################################

        if "GAP_MAX" in locals():
            print("\nPočinjem GAP_MAX obradu...")
            print(f"GAP_MAX vrednost: {GAP_MAX}")

            # Status des Identifikationsrahmens (frm...frame)
            frm = 0
            """
                0...Im aktuellen Zeitschritt ist kein Identifikationsrahmen für
                    Messlücken offen
                1...Im aktuellen Zeitschritt ist ein Identifikationsrahmen für 
                    Messlücken offen
            """
            
            # Durchlauf des gesamten Datenrahmens                   
            for i in range (1, len(df)):
                
                # Kein Messwert für den aktuellen Zeitschritt vorhanden und 
                # Identifikationsrahmen ist geschlossen → Identifikationsrahmen wird 
                # geöffnet
                if pd.isna(df.iloc[i][data_column]) and frm == 0:
                    idx_strt = i
                    frm = 1
                    
                # Messwert für den aktuellen Zeitschritt vorhanden und 
                # Identifikationsrahmen ist offen → Identifikationsrahmen wird
                # geschlossen und ausgewertet
                elif not pd.isna(df.iloc[i][data_column]) and frm == 1:
                    idx_end = i-1
                    
                    # Länge des Identifikationsrahmens [min]
                    frm_width = (dat.strptime(df.iloc[idx_end+1][time_column], \
                                            UTC_fmt)-\
                                dat.strptime(df.iloc[idx_strt-1][time_column], \
                                            UTC_fmt)).total_seconds()/60
                    
                    # Der Identifikationsrahmen ist klein genug, um die Anwendung einer
                    # linearen Interpolation zu erlauben, um die Messlücke zu füllen
                    if frm_width <= GAP_MAX:
                        
                        # Absolute Änderung des Messwertes
                        dif = float(df.iloc[idx_end+1][data_column])-\
                            float(df.iloc[idx_strt-1][data_column])

                        # Änderung des Messwertes pro Minute
                        dif_min = dif/frm_width
                        
                        # Lineare Interpolation
                        for i_frm in range (idx_strt, idx_end+1):
                            

                            gap_min = (dat.strptime(df.iloc[i_frm][time_column], UTC_fmt)-\
                                    dat.strptime(df.iloc[idx_strt-1][time_column], UTC_fmt)).total_seconds()/60
                            

                            df.at[i_frm, data_column] = float(df.iloc[idx_strt-1][data_column])+\
                                gap_min*dif_min                    

                            i_frm += 1
                    frm = 0

                    # Ende des Datensatzes ist erreicht und Identifikationsrahmen ist offen

        print("\nNakon GAP_MAX obrade:")
        print(df.head())

        # Na kraju funkcije, prije except bloka:
        # Konvertujemo DataFrame u format pogodan za JSON
        # Prvo zamijenimo np.nan sa None da bi se moglo serijalizovati u JSON
        df = df.replace({np.nan: None})
        
        print("\nRezultati obrade:")
        print("----------------")
        print(f"Broj redova nakon obrade: {len(df)}")
        print(f"Kolone: {df.columns.tolist()}")
        print("\nPrvih 5 redova:")
        print(df.head())
        
        processed_data = {
            'data': df.to_dict('records'),  # Konvertuje DataFrame u listu dictionary-ja
            'message': 'Daten wurden erfolgreich verarbeitet'
        }
        
        print("\nPrvi red processed_data:")
        print(processed_data['data'][0] if processed_data['data'] else "Nema podataka")
        
        return jsonify(processed_data)

    except Exception as e:
        print(f"FEHLER: {e}")
        return jsonify({'error': str(e)}), 400  
        
if __name__ == '__main__':
    print("Startign Flash server od Port 5005..")
    app.run(port=5005, debug=True)            