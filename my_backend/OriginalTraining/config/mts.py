"""
MTS (Multivariate Time Series) konfiguracija
Ekstrahirano iz training_original.py linije 619-632

Sadrži osnovne parametre za multivarijatnu analizu vremenskih serija.
"""

###############################################################################
# INFORMATIONEN ZUR MULTIVARIATEN ZEITREIHENANALYSE (MULTIVARIATE TIME ########
# SERIES ANALYSIS (MTS)) ######################################################
class MTS:

    # Anzahl der Zeitschritte der Eingabedaten
    I_N  = 13

    # Anzahl der Zeitschritte der Ausgabedaten
    O_N  = 13

    # Zeitschrittweite für die Bildung der finalen Datensätze [min]
    DELT = 3

    # Offset für die Bildung der finalen Datensätze [min]
    OFST = 0
