import pandas as pd
import numpy as np
import datetime
import copy
import statistics

###############################################################################
# DATEN LADEN #################################################################
###############################################################################

#FILE_NAME = "load_grid_kW_Krumpendorf"
#FILE_NAME = "t_out_grad_C_Krumpendorf"
FILE_NAME = "t_RL"

df = pd.read_csv("historical/data_1/"+FILE_NAME+".csv", delimiter = ";")

UTC_fmt = "%Y-%m-%d %H:%M:%S"

# VORBEREITUNG DER ROHDATEN ###################################################
  
# Duplikate (Zeilen mit gleicher UTC) in den Rohdaten löschen (Nur der erste 
# Eintrag wird übernommen)
df1 = df.drop_duplicates(["UTC"]).reset_index(drop = True)

# Konvertieren eines UTC-Strings im Format '%Y-%m-%d %H:%M:%S' in ein 
# datetime-Objekt
df2 = copy.deepcopy(df1)
df2["UTC"] = pd.to_datetime(df2["UTC"], format = "%Y-%m-%d %H:%M:%S")

# Rohdaten nach UTC ordnen
df3 = copy.deepcopy(df2)
df3.sort_values(by = ["UTC"], inplace = True)

# Reset des Indexes in den Rohdaten
df4 = copy.deepcopy(df3)
df4.reset_index(drop = True, inplace = True)

# ZEITGRENZEN #################################################################

# Zeitgrenzen in den Rohdaten
time_min_raw = df4["UTC"][0]
time_max_raw = df4["UTC"][len(df4)-1]

###############################################################################
# EINGABEN ####################################################################
###############################################################################

# Gewünschte Zeitschrittweite der aufbereiteten Daten [min]
tss = 3

# Gewünschter Offset der aufbereiteten Daten [min]
offset = 0
# Methode der Datenaufbereitung    
mode = "mean"
"""
    mean............Mittelwertbildung
    intrpl..........Lineare Interpolation
    nearest.........Nähester Wert
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
#intrpl_max = 10
intrpl_max = 10

###############################################################################
# KONTINUIERLICHER ZEITSTEMPEL ################################################
###############################################################################

# Offset der unteren Zeitgrenze in der Rohdaten
offset_strt = datetime.timedelta(minutes = time_min_raw.minute,
                                 seconds = time_min_raw.second,
                                 microseconds = time_min_raw.microsecond)

# Realer Offset in den aufbereiteten Daten [min]
loop = True
i = 0
while offset >= tss:
    offset -= tss

# Untere Zeitgrenze in den aufbereiteten Daten
i = 0
loop = True
while loop == True:
    a = datetime.timedelta(minutes = i*tss+offset)
    if a >= offset_strt:
        loop = False
    i += 1
time_min = time_min_raw-offset_strt+a

# KONTINUIERLICHER ZEITSTEMPEL ################################################

# Zähler für den Durchlauf des kontinuierlichen Zeitstempels
i = 0

# Initialisierung der Liste mit dem kontinuierlichen Zeitstempel
time_list = []

# Kontinuierlichen Zeitstempel erzeugen
loop = True
while loop == True:
    a = datetime.timedelta(minutes = i*tss)
    time = time_min+a
    if time <= time_max_raw:
        time_list.append(time)
    else:
        loop = False
    i += 1

###############################################################################
# AUFBEREREITUNG DER DATEN ####################################################
###############################################################################

# Zähler für den Durchlauf der Rohdaten
i_raw = 0

# Initialisierung der Liste mit den aufbereiteten Messwerten
value_list = []

# METHODE: MITTELWERTBILDUNG ##################################################

if mode == "mean":
    
    # Schleife durchläuft alle Zeitschritte des kontinuierlichen Zeitstempels
    for i in range(len(time_list)):
        
        print(str(i)+"/"+str(len(time_list)))
                
        # Zeitgrenzen für die Mittelwertbildung (Untersuchungsraum)
        time_int_min = time_list[i]-datetime.timedelta(minutes = tss/2)
        time_int_max = time_list[i]+datetime.timedelta(minutes = tss/2)
        
        # Berücksichtigung angrenzender Untersuchungsräume
        if i > 0: i_raw -= 1
        if i > 0 and df4["UTC"][i_raw] < time_int_min: i_raw += 1                           
        
        # Initialisierung der Liste mit den Messwerten im Untersuchungsraum
        value_int_list = []
        
        # Auflistung numerischer Messwerte im Untersuchungsraum
        while i_raw < len(df4) \
            and df4["UTC"][i_raw] <= time_int_max \
            and df4["UTC"][i_raw] >= time_int_min:
                
            try:
                value_int_list.append(float(df4.iloc[i_raw, 1]))
            except:
                pass
            i_raw += 1
            
        # Mittelwertbildung über die numerischen Messwerte im Untersuchungsraum
        if len(value_int_list) > 0:
            value_list.append(statistics.mean(value_int_list))
        else:
            value_list.append("nan")

# METHODE: LINEARE INTERPOLATION ##############################################

elif mode == "intrpl":
    
    # Zähler für den Durchlauf der Rohdaten
    i_raw = 0
        
    # Richtung des Schleifendurchlaufs
    direct = 1
    """
        direct = 1...Schleife läuft vorwärts
        direct = -1..Schleife läuft rückwärts
    """
    
    
    # Schleife durchläuft alle Zeitschritte des kontinuierlichen Zeitstempels.
    for i in range(0, len(time_list)):
        
        print(i)
        
        # Schleife durchläuft die Rohdaten von vorne bis hinten zur Auffindung 
        # des nachfolgenden Wertes.
        if direct == 1:
            
            loop = True
            while i_raw < len(df4) and loop == True:
                
                # Der aktuelle Zeitpunkt in den Rohdaten liegt nach dem 
                # aktuellen Zeitpunkt im kontinuierlichen Zeitstempel oder ist 
                # mit diesem identisch.
                if df4["UTC"][i_raw] >= time_list[i]:
                    
                    # Der aktuelle Zeitpunkt in den Rohdaten liegt nach dem 
                    # aktuellen Zeitpunkt im kontinuierlichen Zeitstempel oder 
                    # ist mit diesem identisch und der dazugehörige Messwert 
                    # ist numerisch.
                    try:
                        
                        # UTC und Messwert vom nachfolgenden Wert übernehmen. 
                        time_next = df4["UTC"][i_raw]
                        value_next = float(df4.iloc[i_raw, 1])
                        loop = False
                        
                    except:
                        
                        # Zähler aktuallisieren, wenn Messwert nicht numerisch.
                        i_raw += 1
                
                else:
                    
                    # Zähler aktuallisieren, wenn der aktuelle Zeitpunkt in 
                    # den Rohdaten vor dem aktuellen Zeitpunkt im 
                    # kontinuierlichen Zeitstempel liegt.
                    i_raw += 1
            
            # Die gesamten Rohdaten wurden durchlaufen und es wurde kein 
            # gültiger Messwert gefunden.
            if i_raw+1 > len(df4):
                value_list.append("nan")
                
                # Zähler für die Rohdaten auf Null setzen und Schleifenrichtung
                # festlegen.
                i_raw = 0
                direct = 1
            else:
                
                # Schleifenrichtung umdrehen
                direct = -1
        
        # Finden des vorangegangenen Wertes
        if direct == -1:
            
            loop = True
            while i_raw >= 0 and loop == True:
                
                # Der aktuelle Zeitpunkt in den Rohdaten liegt vor dem 
                # aktuellen Zeitpunkt im kontinuierlichen Zeitstempel oder ist 
                # mit diesem identisch.
                if df4["UTC"][i_raw] <= time_list[i]:
                    
                    # Der aktuelle Zeitpunkt in den Rohdaten liegt vor dem 
                    # aktuellen Zeitpunkt im kontinuierlichen Zeitstempel oder 
                    # ist mit diesem identisch und der dazugehörige Messwert 
                    # ist numerisch.
                    try:
                        
                        # UTC und Messwert vom vorangegangenen Wert übernehmen 
                        time_prior = df4["UTC"][i_raw]
                        value_prior = float(df4.iloc[i_raw, 1])
                        
                        loop = False
                    except:
                        # Zähler aktuallisieren, wenn Messwert nicht numerisch
                        i_raw -= 1
                else:
                    # Zähler aktuallisieren, wenn der aktuelle Zeitpunkt in den 
                    # Rohdaten nach dem aktuellen Zeitpunkt im kontinuierlichen 
                    # Zeitstempel liegt.
                    i_raw -= 1
              
            # Die gesamten Rohdaten wurden durchlaufen und es wurde kein 
            # gültiger Messwert gefunden.
            if i_raw < 0:
                value_list.append("nan")
                
                # Zähler für die Rohdaten auf Null setzen und Schleifenrichtung 
                # festlegen.
                i_raw = 0
                direct = 1
                
            # Es wurde ein gültiger Messwert vor dem aktuellen Zeitpunkt im 
            # kontinuierlichen Zeitstempel und nach diesem gefunden.
            else:
                delta_time = time_next-time_prior
                
                # Zeitabstand zwischen den entsprechenden Messwerten in den 
                # Rohdaten [sec]
                delta_time_sec = delta_time.total_seconds()
                delta_value = value_prior-value_next
          
                # Zeitpunkte fallen zusammen oder gleichbleibender Messwert 
                # → Keine lineare Interpolation notwendig
                if delta_time_sec == 0 \
                    or (delta_value == 0 and delta_time_sec <= intrpl_max*60):
                    
                    value_list.append(value_prior)
                
                # Zeitabstand zu groß → Keine lineare Interpolation möglich
                elif delta_time_sec > intrpl_max*60:
                    value_list.append("nan")
                
                # Lineare Interpolation
                else:
                    delta_time_prior_sec = (time_list[i]-time_prior).\
                        total_seconds()   
                    value_list.append(value_prior-delta_value/delta_time_sec*
                                      delta_time_prior_sec)
                
                    
                direct = 1
            
# METHODE: ZEITLICH NÄCHSTLIEGENDER MESSWERT ##################################       

elif mode == "nearest" or mode == "nearest (mean)":
   
    # Schleife durchläuft alle Zeitschritte des kontinuierlichen Zeitstempels
    for i in range(len(time_list)):
        
        # Zeitgrenzen für die Untersuchung (Untersuchungsraum)
        time_int_min = time_list[i]-datetime.timedelta(minutes = tss/2)
        time_int_max = time_list[i]+datetime.timedelta(minutes = tss/2)
        
        # Berücksichtigung angrenzender Untersuchungsräume
        if i > 0: i_raw -= 1
        if i > 0 and df4["UTC"][i_raw] < time_int_min: i_raw += 1  
        
        # Initialisierung der Listen für den Untersuchungsraum
        value_int_list, delta_time_int_list = ([] for i in range(2))
        
        # Auflistung numerischer Messwerte mit Zeifdifferenzen (zum jeweiligen 
        # Zeitpunkt des kontinuierlichen Zeitstempels) im Untersuchungsraum
        while   i_raw < len(df4) \
            and df4["UTC"][i_raw] <= time_int_max \
            and df4["UTC"][i_raw] >= time_int_min:
                
            try:
                value_int_list.append(float(df4.iloc[i_raw, 1]))
                delta_time_int_list.append(abs(time_list[i]-df4["UTC"][i_raw]))
            except:
                pass
            i_raw += 1
        
        # Zeitlich nächstliegenden Messwert eruieren (Wenn zwei Messwerte zum 
        # jeweiligen Zeitpunkt des kontinuierlichen Zeitstempels den gleichen 
        # minimalen zeitlichen Abstand aufweisen, wird der erste Messwert 
        # übernommen)
        if len(value_int_list) > 0:
            
            if mode == "nearest":
                delta_time_int_array = np.array(delta_time_int_list)
                item_index = np.where(delta_time_int_array == 
                                      min(delta_time_int_list))
                value_list.append(value_int_list[item_index[0][0]])
            elif mode == "nearest (mean)":
                delta_time_int_array = np.array(delta_time_int_list)
                item_index = np.where(delta_time_int_array == 
                                      min(delta_time_int_list))
                value_int_mean_list = []
                for i1 in range(0, len(item_index[0])):
                    value_int_mean_list.append(value_int_list[item_index\
                                                              [0][i1]])
                value_list.append(statistics.mean(value_int_mean_list))
        else:
            value_list.append("nan")

# DATENRAHMEN MIT DEN AUFBEREITETEN DATEN #####################################

df5 = pd.DataFrame({"UTC": time_list, df4.columns[1]: value_list}) 


df5.to_csv("historical/data_2/"+FILE_NAME+".csv", index = False, sep = ";")