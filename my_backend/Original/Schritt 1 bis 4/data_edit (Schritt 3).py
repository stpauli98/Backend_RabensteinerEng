import pandas as pd
import math

###############################################################################
# DATEN LADEN #################################################################
###############################################################################

#FILE_NAME = "load_grid_kW_Krumpendorf"
FILE_NAME = "t_RL"

df = pd.read_csv("historical/data_2/"+FILE_NAME+".csv", delimiter = ";")

UTC_fmt = "%Y-%m-%d %H:%M:%S"

# Konvertieren eines UTC-Strings im Format '%Y-%m-%d %H:%M:%S' in ein 
# datetime-Objekt
df["UTC"] = pd.to_datetime(df["UTC"], format = UTC_fmt)

###############################################################################
# EINGABEN ####################################################################
###############################################################################

# Zulässige Zeitspanne mit unverändertem Messwert (EQ...equal) [min]
EQ_MAX = 20

# Maximalwert [#]
#ELMAX = 1000

# Minimalwert [#]
ELMIN = 30

# Nullwerte entfernen (EL...elimination)
EL0 = "nein"

# Nicht numerische Werte entfernen (EL...elimination, NN...non numeric)
ELNN = "ja"

# Zulässige zeitliche Änderung (CHG...change) des Messwerts (Ausreißererkennung) [#/min]
CHG_MAX = 2

# Zulässigen Dauer (LG...length) von Extremwerten (Ausreißererkennung) [min]
LG_MAX = 30

# Zulässige Dauer von Messausfällen zur Schließung mittels linearer Interpolation [min]
GAP_MAX = 1000000

###############################################################################
# ELIMINIERUNG VON MESSAUSFÄLLEN (GLEICHBLEIBENDE MESSWERTE) ##################
###############################################################################

if "EQ_MAX" in locals():
    
    print("Eliminierung von Messausfällen (gleichbleibende Messwerte)")
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
        
        #print(str(i+1)+"/"+str(len(df)))
                
        # Aktueller Messwert ist gleich dem letzen Messwert und 
        # Identifikationsrahmen ist geschlossen → Identifikationsrahmen wird
        # geöffnet
        if df.iloc[i-1][1] == df.iloc[i][1] and frm == 0:           
            
            idx_strt = i-1
            frm = 1
        
        # Aktueller Messwert ist ungleich dem letzten Messwert und
        # Identifikationsrahmen ist offen → Identifikationsrahmen wird
        # geschlossen und ausgewertet
        elif df.iloc[i-1][1] != df.iloc[i][1] and frm == 1:
            
            idx_end = i-1
            
            # Länge des Identifikationsrahmens [min]
            frm_width = (df.iloc[idx_end][0]-
                         df.iloc[idx_strt][0]).total_seconds()/60
            
            # Der Identifikationsrahmen ist zu groß, so dass eine Messlücke
            # eingefügt werden muss
            if frm_width >= EQ_MAX:
                for i_frm in range (idx_strt, idx_end+1):
                    df.iloc[i_frm, 1] = "nan"
            frm = 0
   
        # Ende des Datensatzes ist erreicht und Identifikationsrahmen ist offen
        elif i == len(df)-1 and frm == 1: 
            idx_end = i
            
            # Länge des Identifikationsrahmens [min]
            frm_width = (df.iloc[idx_end][0]-
                         df.iloc[idx_strt][0]).total_seconds()/60
            
            # Der Identifikationsrahmen ist zu groß, so dass eine Messlücke
            # eingefügt werden muss
            if frm_width >= EQ_MAX:
                for i_frm in range (idx_strt, idx_end+1):
                    df.iloc[i_frm, 1] = "nan"

###############################################################################
# WERTE ÜBER DEM OBEREN GRENZWERT ENTFERNEN ###################################
###############################################################################
        
if "ELMAX" in locals():
                
    # Durchlauf des gesamten Datenrahmens
    for i in range (len(df)): 
                    
        try:
            if float(df.iloc[i][1]) > ELMAX:
                df.iloc[i, 1] = "nan"
        except:
            df.iloc[i, 1] = "nan"

###############################################################################
# WERTE UNTER DEM UNTEREN GRENZWERT ENTFERNEN #################################
###############################################################################
            
if "ELMIN" in locals():
            
    # Durchlauf des gesamten Datenrahmens
    for i in range (len(df)): 
                    
        try:
            if float(df.iloc[i][1]) < ELMIN:
                df.iloc[i, 1] = "nan"
        except:
            df.iloc[i, 1] = "nan"

###############################################################################
# ELIMINIERUNG VON NULLWERTEN #################################################
###############################################################################

if EL0 == "ja":
    
    print("Eliminierung von Nullwerten")
    
    # Durchlauf des gesamten Datenrahmens
    for i in range (0, len(df)):
        
        #print(str(i+1)+"/"+str(len(df)))
        
        if df.iloc[i][1] == 0:
            df.iloc[i, 1] = "nan"

###############################################################################
# ELIMINIERUNG VON NICHT NUMERISCHEN WERTEN ###################################
###############################################################################

if ELNN == "ja":

    print("Eliminierung von nicht numerischen Werten")    

    # Durchlauf des gesamten Datenrahmens
    for i in range (0, len(df)):
        
        #print(str(i+1)+"/"+str(len(df)))
        
        try:
            float(df.iloc[i][1])
            if math.isnan(float(df.iloc[i][1])) == True:
               df.iloc[i, 1] = "nan" 
        except:
            df.iloc[i, 1] = "nan"

###############################################################################
# ELIMINIERUNG VON AUSREISSERN ################################################
###############################################################################

if "CHG_MAX" and "LG_MAX" in locals():
   
    print("Eliminierung von Ausreissern") 
   
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
        
        #print(str(i+1)+"/"+str(len(df)))
        
        # nan im aktuellen Zeitschritt und Identifikationsrahmen ist im
        # aktuellen Zeitschritt nicht offen
        if df.iloc[i][1] == "nan" and frm == 0:
            pass
        
        # nan im aktuellen Zeitschritt und Identifikationsrahmen ist im
        # aktuellen Zeitschritt offen
        elif df.iloc[i][1] == "nan" and frm == 1:
            idx_end = i-1
 
            for i_frm in range (idx_strt, idx_end+1):
                df.iloc[i_frm, 1] = "nan"

            # Identifikationsrahmen wird geschlossen
            frm = 0
        
        # nan im letzten Zeitschritt
        elif  df.iloc[i-1][1] == "nan":
            pass
        
        # Kein nan im letzten und aktuellen Zeitschritt
        else:    
            # Änderung des Messwertes im aktuellen Zeitschritt
            chg = abs(float(df.iloc[i][1])-float(df.iloc[i-1][1]))
            
            # Zeitschrittweite vom letzten zum aktuellen Zeitschritt [min]
            t = (df.iloc[i][0]-df.iloc[i-1][0]).total_seconds()/60
            
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
                    df.iloc[i_frm, 1] = "nan"
                
                # Identifikationsrahmen wird geschlossen
                frm = 0
            
            # Identifikationsrahmen ist offen und die maximale Breite des 
            # Identifikationsrahmens wurde erreicht → Identifikationsrahmen
            # wird geschlossen
            elif    frm == 1 \
                and (df.iloc[i][0]-
                     df.iloc[idx_strt][0]).total_seconds()/60 > LG_MAX:
                    
                frm = 0
                    
###############################################################################
# SCHLIESSEN VON MESSLÜCKEN ###################################################
###############################################################################

if "GAP_MAX" in locals():

    print("Schließen von Messlücken")    

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
        
        print(str(i+1)+"/"+str(len(df)))
        
        # Kein Messwert für den aktuellen Zeitschritt vorhanden und 
        # Identifikationsrahmen ist geschlossen → Identifikationsrahmen wird 
        # geöffnet
        if df.iloc[i][1] == "nan" and frm == 0:
            idx_strt = i
            frm = 1
            
        # Messwert für den aktuellen Zeitschritt vorhanden und 
        # Identifikationsrahmen ist offen → Identifikationsrahmen wird
        # geschlossen und ausgewertet
        elif df.iloc[i][1] != "nan" and frm == 1:

            idx_end = i-1
            
            # Länge des Identifikationsrahmens [min]
            frm_width = (df.iloc[idx_end+1][0]-
                         df.iloc[idx_strt-1][0]).total_seconds()/60
            
            # Der Identifikationsrahmen ist klein genug, um die Anwendung einer
            # linearen Interpolation zu erlauben, um die Messlücke zu füllen
            if frm_width <= GAP_MAX:
                
                # Absolute Änderung des Messwertes
                dif = float(df.iloc[idx_end+1][1])-\
                    float(df.iloc[idx_strt-1][1])

                # Änderung des Messwertes pro Minute
                dif_min = dif/frm_width
                
                # Lineare Interpolation
                for i_frm in range (idx_strt, idx_end+1):
                    
                    gap_min = (df.iloc[i_frm][0]-
                               df.iloc[idx_strt-1][0]).total_seconds()/60
                    
                    df.iloc[i_frm, 1] = float(df.iloc[idx_strt-1][1])+\
                        gap_min*dif_min                    

                    i_frm += 1
            frm = 0

df.to_csv("historical/data_3/"+FILE_NAME+".csv", index = False, sep = ";")