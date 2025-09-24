import pandas as pd
import numpy as np
import math
import sys
import datetime
import statistics
from datetime import datetime as dat

UTC_fmt = "%Y-%m-%d %H:%M:%S"

###############################################################################
###############################################################################
# DATEN LADEN #################################################################
###############################################################################
###############################################################################

files = ["t_VL.csv",
         "t_RL.csv"]

df = []
info = []
for i in range (len(files)):
    df.append(pd.read_csv("historical/data_3/"+files[i], delimiter = ";"))

###############################################################################
# ZEITSTEMPEL IN DATETIME KONVERTIERTEN #######################################
###############################################################################

for i in range (len(files)):

    #df[i]["UTC"] = [dat.strptime(ts, UTC_fmt) for ts in df[i].iloc[:,0]]
    
    # Konvertieren eines UTC-Strings im Format '%Y-%m-%d %H:%M:%S' in ein 
    # datetime-Objekt
    df[i]["UTC"] = pd.to_datetime(df[i]["UTC"], format = UTC_fmt)

###############################################################################
# INFORMATIONEN ZU DEN DATEN ##################################################
###############################################################################

for i in range (len(files)):
    
    # GRUNDLEGENDE INFORMATIONEN ##############################################
    
    info.append([files[i],                                  # Name der Datei
                 df[i].columns.values[1],                   # Name der Messreihe
                 min(df[i]["UTC"]),                         # Startzeit (UTC)
                 max(df[i]["UTC"]),                         # Endzeit (UTC)
                 (max(df[i]["UTC"])-min(df[i]["UTC"])).\
                     total_seconds()/(60*(len(df[i])-1))])  # Zeitschrittweite [min]
                                               
    # OFFSET [MIN] ############################################################

    ofst = ((min(df[i]["UTC"]))-
            (min(df[i]["UTC"])).replace(minute = 0, 
                                        second = 0, 
                                        microsecond = 0)).total_seconds()/60
    while ofst-info[i][-1]>= 0:
       ofst -= info[i][-1]
    info[i].append(ofst)
    
    # ANZAHL DER DATENPUNKTE ##################################################
    info[i].append(len(df[i]))
    
    # ANZAHL DER NUMERISCHEN DATENPUNKTE ######################################
    x = 0
    for i1 in range(len(df[i])):
        try:
            float(df[i].iloc[i1][1])
            if math.isnan(float(df[i].iloc[i1][1])) == True:
               x += 1  
        except:
            x += 1  
    info[i].append(info[i][-1]-x)
    
    # ANTEIL AN NUMERISCHEN DATENPUNKTEN [%] ##################################
    info[i].append(info[i][-1]/info[i][-2]*100)

info_df = pd.DataFrame(info, columns = ["Name der Datei",
                                        "Name der Messreihe",
                                        "Startzeit (UTC)",
                                        "Endzeit (UTC)",
                                        "Zeitschrittweite [min]",
                                        "Offset [min]",
                                        "Anzahl der Datenpunkte",
                                        "Anzahl der numerischen Datenpunkte",
                                        "Anteil an numerischen Datenpunkten"])

###############################################################################
###############################################################################
# EINGABEN ####################################################################
###############################################################################
###############################################################################

# Startzeit (UTC)
#t_strt = "2022-03-02 00:05:27"

# Endzeit (UTC)
#t_end  = dat.strptime("2023-04-02 00:00:00", UTC_fmt)

# Zeitschrittweite [min]
tss = 15

# Offset [min]
ofst = 0

###############################################################################
# DATENGRENZEN ################################################################
###############################################################################

# Gemeinsame Start- und Endzeit der Daten
t_strt_f = info_df.iloc[:,2].max()
t_end_f  = info_df.iloc[:,3].min()

# Startzeit
if "t_strt" in locals():
   try:
       t_strt = dat.strptime(t_strt, UTC_fmt)
   except:
       print("Eingabe der Startzeit (UTC) im Format 'JJJJ-MM-TT hh:mm:ss'")
       sys.exit()
    
   if t_strt_f > t_strt:
       t_strt = t_strt_f
else:
    t_strt = t_strt_f

# Endzeit
if "t_end" in locals():
   try:
       t_end = dat.strptime(t_end, UTC_fmt)
   except:
       print("Eingabe der Endzeit (UTC) im Format 'JJJJ-MM-TT hh:mm:ss'")
       sys.exit()
    
   if t_end_f < t_end:
       t_end = t_end_f
else:
    t_end= t_end_f

if t_strt > t_end:
    print("Keine zeitliche Überschneidung der Datensätze!")
    sys.exit()

###############################################################################
# ÜBERPRÜFUNG DER EINGEGEBENEN WERTE ##########################################
###############################################################################    

# Zeitschrittweite
if "tss" in locals():
    try:
        tss = float(tss)
        if tss <= 0:
            print("Geben Sie einen positiven Wert als Zeitschrittweite ein!")
            sys.exit()
    except:
        print("Geben Sie einen numerischen Wert als Zeitschrittweite ein!")
        sys.exit()
else:
    print("Geben Sie die gewünschte Zeitschrittweite ein!")
    sys.exit()

# Offset
if "ofst" in locals():
    try:
        ofst = float(ofst)
        if ofst < 0:
            while ofst < 0:
                ofst += tss
    except:
        print("Geben Sie einen numerischen Wert als Zeitverschiebung (Offset) ein!")
        sys.exit()   
else:
    print("Geben Sie die gewünschte Zeitverschiebung (Offset) auf die volle Stunde ein!")
    sys.exit()

###############################################################################
# KONTINUIERLICHER ZEITSTEMPEL ################################################
###############################################################################

t_ref = t_strt.replace(minute = 0, second = 0, microsecond = 0)
t_ref += datetime.timedelta(minutes = ofst)

while t_ref < t_strt:
    t_ref += datetime.timedelta(minutes = tss)

t_list = []
while t_ref <= t_end:
    t_list.append(t_ref)
    t_ref += datetime.timedelta(minutes = tss)


###############################################################################
# ABFRAGEN DER MODI ###########################################################
############################################################################### 

mode = []
intrpl_max = []
nearest_max = []
for i in range (len(files)):
    if info_df["Zeitschrittweite [min]"][i] != tss or info_df["Offset [min]"][i] != ofst:
       
        # load 
        if i == 0:
            mode.append("mean")
        
        # temp_out
        elif i == 1:
            mode.append("mean")
      
        # Maximale Zeitschrittweite der linearen Interpolation [min]
        
        # load
        if i == 0:
            intrpl_max.append(10)
        
        # temp_out
        elif i == 1:
            intrpl_max.append(100)
       
        # Maximale Zeitschrittweite bei der Suche des nächstliegenden Messwertes [min]
        
        # load
        if i == 0:
            nearest_max.append(10)
            
        # temp_out
        if i == 1:
            nearest_max.append(100)
            
        
    else:
        mode.append("")



###############################################################################
###############################################################################
# DATEN VERARBEITEN ###########################################################
###############################################################################
###############################################################################

df_final = []
   
for i in range (len(files)):
    
    # KEINE ANPASSUNG NOTWENDIG ###############################################
    
    if info_df["Zeitschrittweite [min]"][i] == tss and info_df["Offset [min]"][i] == ofst:
        idx_strt = (np.where(df[i]["UTC"] == t_list[0]))[0][0]
        idx_end  = (np.where(df[i]["UTC"] == t_list[-1]))[0][0]
        df_final.append(df[i][idx_strt:idx_end+1])
        df_final[i].reset_index(drop = True, inplace = True)
    
    # ANPASSUNG NOTWENDIG #####################################################
    
    else:
                
        # Zähler für den Durchlauf der Rohdaten
        i2 = 0
        
        # Initialisierung der Liste mit den aufbereiteten Messwerten
        value_list = []
        
        #######################################################################
        # METHODE: MITTELWERTBILDUNG ##########################################
        #######################################################################
        
        if mode[i] == "mean":
            
            # Schleife durchläuft alle Zeitschritte des kontinuierlichen Zeitstempels
            for i1 in range(0, len(t_list)):
                
                # UNTERSUCHUNGSRAUM DEFINIEREN ################################
                
                # Zeitgrenzen für die Mittelwertbildung (Untersuchungsraum)
                t_int_min = t_list[i1]-datetime.timedelta(minutes = tss/2)
                t_int_max = t_list[i1]+datetime.timedelta(minutes = tss/2)
                                
                # Index des ersten Messwertes des Untersuchungsraumes
                while i2 < len(df[i]) and df[i]["UTC"][i2] < t_int_min:
                    i2 += 1
                idx_int_strt = i2

                # Index des letzten Messwertes des Untersuchungsraumes
                while i2 < len(df[i]) and df[i]["UTC"][i2] <= t_int_max:
                    i2 += 1
                idx_int_end = i2-1
                
                # AUFLISTUNG DER NUMERISCHEN MESSWERTE IM UNTERSUCHUNGSRAUM ###
                
                # Initialisierung der Liste mit den Messwerten im Untersuchungsraum
                value_int_list = []
                
                # Schleife durchläuft den Untersuchungsraum
                for i2 in range(idx_int_strt, idx_int_end+1):
                    if math.isnan(df[i].iloc[i2, 1]) == False:
                        value_int_list.append(df[i].iloc[i2, 1])

                
                # Mittelwertbildung über die numerischen Messwerte im Untersuchungsraum
                if len(value_int_list) > 0:
                    value_list.append(statistics.mean(value_int_list))
                else:
                    value_list.append("nan")
        
                    
        #######################################################################
        # METHODE: ZEITLICH NÄCHSTLIEGENDER MESSWERT ##########################
        #######################################################################
        
        elif mode[i] == "nearest" or mode[i] == "nearest (mean)":
            
            # Schleife durchläuft alle Zeitschritte des kontinuierlichen Zeitstempels
            for i1 in range(0, len(t_list)):
                
                # UNTERSUCHUNGSRAUM DEFINIEREN ################################
                
                # Untersuchungsraum
                t_int_min = t_list[i1]-datetime.timedelta(minutes = tss/2)
                t_int_max = t_list[i1]+datetime.timedelta(minutes = tss/2)
                                
                # Index des ersten Messwertes des Untersuchungsraumes
                while i2 < len(df[i]) and df[i]["UTC"][i2] < t_int_min:
                    i2 += 1
                idx_int_strt = i2

                # Index des letzten Messwertes des Untersuchungsraumes
                while i2 < len(df[i]) and df[i]["UTC"][i2] <= t_int_max:
                    i2 += 1
                idx_int_end = i2-1
                
                # AUFLISTUNG DER NUMERISCHEN MESSWERTE IM UNTERSUCHUNGSRAUM ###
                
                # Initialisierung der Liste mit den Messwerten im Untersuchungsraum
                value_int_list = []
                date_int_list = []
                t_delta_int_list = []
                
                # Schleife durchläuft den Untersuchungsraum
                for i2 in range(idx_int_strt, idx_int_end+1):
                    if math.isnan(df[i].iloc[i2, 1]) == False:
                        date_int_list.append(df[i].iloc[i2, 0])
                        value_int_list.append(df[i].iloc[i2, 1])
                        t_delta_int_list.append(abs(t_list[i1]-df[i].iloc[i2, 0]).total_seconds())
                
                if len(value_int_list) > 0:
                    
                    if mode[i] == "nearest":
                        t_delta_int_array = np.array(t_delta_int_list)
                        item_index = np.where(t_delta_int_array == min(t_delta_int_list))
                        value_list.append(value_int_list[item_index[0][0]])
                    elif mode[i] == "nearest (mean)":
                        t_delta_int_array = np.array(t_delta_int_list)
                        item_index = np.where(t_delta_int_array == min(t_delta_int_list))
                        value_int_mean_list = []
                        for i3 in range(0, len(item_index[0])):
                            value_int_mean_list.append(value_int_list[item_index[0][i3]])
                        value_list.append(statistics.mean(value_int_mean_list))
                
                else:
                    value_list.append("nan")
        
        #######################################################################
        # METHODE: LINEARE INTERPOLATION ######################################
        #######################################################################
        
        elif mode[i] == "intrpl" or mode[i] == "nearest (max. delta)":    
            
            # Richtung des Schleifendurchlaufs
            direct = 1
            """
                direct = 1...Schleife läuft vorwärts
                direct = -1..Schleife läuft rückwärts
            """ 
            
            # Schleife durchläuft alle Zeitschritte des kontinuierlichen Zeitstempels
            for i1 in range(0, len(t_list)):
                
                # Schleife durchläuft die Rohdaten von vorne bis hinten zur Auffindung des nachfolgenden Wertes
                if direct == 1:
                    
                    loop = True
                    while i2 < len(df[i]) and loop == True:
                        
                        # Der aktuelle Zeitpunkt in den Rohdaten liegt nach dem aktuellen Zeitpunkt
                        # im kontinuierlichen Zeitstempel oder ist mit diesem identisch.
                        if df[i]["UTC"][i2]>= t_list[i1]:
                            
                            # Der aktuelle Zeitpunkt in den Rohdaten liegt nach dem aktuellen Zeitpunkt
                            # im kontinuierlichen Zeitstempel oder ist mit diesem identisch und der 
                            # dazugehörige Messwert ist numerisch
                            if math.isnan(float(df[i].iloc[i2, 1])) == False:
                                
                                # UTC und Messwert vom nachfolgenden Wert übernehmen 
                                time_next = df[i].iloc[i2, 0]
                                value_next = float(df[i].iloc[i2, 1])
                                loop = False
                                
                            else:
                                
                                # Zähler aktuallisieren, wenn Messwert nicht numerisch
                                i2 += 1
                        
                        else:
                            
                            # Zähler aktuallisieren, wenn der aktuelle Zeitpunkt in den Rohdaten vor dem aktuellen
                            # Zeitpunkt im kontinuierlichen Zeitstempel liegt.
                            i2 += 1
                    
                    # Die gesamten Rohdaten wurden durchlaufen und es wurde kein gültiger Messwert gefunden
                    if i2+1 > len(df[i]):
                        value_list.append("nan")
                        
                        # Zähler für die Rohdaten auf Null setzen und Schleifenrichtung festlegen
                        i2 = 0
                        direct = 1
                    else:
                        
                        # Schleifenrichtung umdrehen
                        direct = -1
                
                # Finden des vorangegangenen Wertes
                if direct == -1:
                    
                    loop = True
                    while i2 >= 0 and loop == True:
                        
                        # Der aktuelle Zeitpunkt in den Rohdaten liegt vor dem aktuellen Zeitpunkt
                        # im kontinuierlichen Zeitstempel oder ist mit diesem identisch.
                        if df[i]["UTC"][i2] <= t_list[i1]:
                            
                            # Der aktuelle Zeitpunkt in den Rohdaten liegt vor dem aktuellen Zeitpunkt
                            # im kontinuierlichen Zeitstempel oder ist mit diesem identisch und der 
                            # dazugehörige Messwert ist numerisch
                            if math.isnan(float(df[i].iloc[i2, 1])) == False:
                                
                                # UTC und Messwert vom vorangegangenen Wert übernehmen 
                                time_prior = df[i].iloc[i2, 0]
                                value_prior = float(df[i].iloc[i2, 1])
                                loop = False
                            else:
                                # Zähler aktuallisieren, wenn Messwert nicht numerisch
                                i2 -= 1
                        else:
                            # Zähler aktuallisieren, wenn der aktuelle Zeitpunkt in den Rohdaten nach dem aktuellen
                            # Zeitpunkt im kontinuierlichen Zeitstempel liegt.
                            i2 -= 1
                            
                    # Die gesamten Rohdaten wurden durchlaufen und es wurde kein gültiger Messwert gefunden
                    if i2 < 0:
                        value_list.append("nan")
                        
                        # Zähler für die Rohdaten auf Null setzen und Schleifenrichtung festlegen
                        i2 = 0
                        direct = 1
                        
                    # Es wurde ein gültige Messwerte vor dem aktuellen Zeitpunkt im kontinuierlichen Zeitstempel und nach diesem gefunden
                    else:
                        delta_time = time_next-time_prior
                        
                        #print(str(t_list[i1])+": "+str(time_prior)+" ; "+str(value_prior)+" ; "+str(time_next)+" ; "+str(value_next)+" ; "+str(delta_time))
                  
                        # Zeitabstand zwischen den entsprechenden Messwerten in den Rohdaten [sec]
                        delta_time_sec = delta_time.total_seconds()
                        delta_value = value_prior-value_next
                  
                        if mode[i] == "intrpl":
                  
                            # Zeitpunkte fallen zusammen oder gleichbleibender Messwert - Keine lineare Interpolation notwendig
                            if delta_time_sec == 0 or (delta_value == 0 and delta_time_sec <= intrpl_max[i]*60):
                                value_list.append(value_prior)
                            
                            # Zeitabstand zu groß - Keine lineare Interpolation möglich
                            elif delta_time_sec > intrpl_max[i]*60:
                                value_list.append("nan")
                            
                            # Lineare Interpolation
                            else:
                                delta_time_prior_sec = (t_list[i1]-time_prior).total_seconds()   
                                value_list.append(value_prior-delta_value/delta_time_sec*delta_time_prior_sec)
                                
                        else:
                                                        
                            # Zeitpunkte fallen zusammen oder gleichbleibender Messwert - Keine lineare Interpolation notwendig
                            if delta_time_sec == 0 or (delta_value == 0 and delta_time_sec <= nearest_max[i]*60):
                                value_list.append(value_prior)
                            
                            # Zeitabstand zu groß - Näherst liegender Wert darf nicht verwendet werden
                            elif delta_time_sec > nearest_max[i]*60:
                                value_list.append("nan")
                            
                            # Übernahme des näherst liegenden Wertes
                            else:
                                delta_time_prior_sec = (t_list[i1]-time_prior).total_seconds()
                                delta_time_next_sec = (time_next-t_list[i1]).total_seconds()
                                
                                if delta_time_prior_sec < delta_time_next_sec:
                                    value_list.append(value_prior)
                                else:
                                    value_list.append(value_next)                       
                            
                        direct = 1
                            
        
                
        df_final.append(pd.DataFrame({"UTC": t_list, info[i][1]: value_list}))
        
###############################################################################
# INFORMATIONEN ZU DEN FINALEN DATEN ##########################################
###############################################################################

info_final = []

for i in range (len(files)):
    
    # GRUNDLEGENDE INFORMATIONEN ##############################################
    
    info_final.append([files[i],                                  # Name der Datei
                 df_final[i].columns.values[1],                   # Name der Messreihe
                 min(df_final[i]["UTC"]),                         # Startzeit (UTC)
                 max(df_final[i]["UTC"]),                         # Endzeit (UTC)
                 (max(df_final[i]["UTC"])-min(df_final[i]["UTC"])).\
                     total_seconds()/(60*(len(df_final[i])-1))])  # Zeitschrittweite [min]
                                               
    # OFFSET [MIN] ############################################################

    ofst = ((min(df_final[i]["UTC"]))-
            (min(df_final[i]["UTC"])).replace(minute = 0, 
                                        second = 0, 
                                        microsecond = 0)).total_seconds()/60
    while ofst-info_final[i][-1]>= 0:
       ofst -= info_final[i][-1]
    info_final[i].append(ofst)
    
    # ANZAHL DER DATENPUNKTE ##################################################
    info_final[i].append(len(df_final[i]))
    
    # ANZAHL DER NUMERISCHEN DATENPUNKTE ######################################
    x = 0
    for i1 in range(len(df_final[i])):
        try:
            float(df_final[i].iloc[i1][1])
            if math.isnan(float(df_final[i].iloc[i1][1])) == True:
               x += 1  
        except:
            x += 1  
    info_final[i].append(info_final[i][-1]-x)
    
    # ANTEIL AN NUMERISCHEN DATENPUNKTEN [%] ##################################
    info_final[i].append(info_final[i][-1]/info_final[i][-2]*100)

info_final_df = pd.DataFrame(info_final, columns = ["Name der Datei",
                                        "Name der Messreihe",
                                        "Startzeit (UTC)",
                                        "Endzeit (UTC)",
                                        "Zeitschrittweite [min]",
                                        "Offset [min]",
                                        "Anzahl der Datenpunkte",
                                        "Anzahl der numerischen Datenpunkte",
                                        "Anteil an numerischen Datenpunkten"])

for i in range (len(files)):
    df_final[i].to_csv("historical/data_4/"+str(files[i]), index = False, sep = ";")

