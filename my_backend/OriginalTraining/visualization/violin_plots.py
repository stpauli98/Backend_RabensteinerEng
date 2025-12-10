"""
Violin Plots - Vizualizacija distribucije podataka
Ekstrahirano iz training_original.py linije 1876-2027

Sadrži:
- Violin plot za ulazne podatke
- Violin plot za izlazne podatke
- Paleta boja tab20
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config.time_features import T

###############################################################################
###############################################################################
# VIOLINENPLOT ################################################################
###############################################################################
###############################################################################

# Napomena: Ovaj kod zahtijeva prethodno definirane varijable:
# i_combined_array, o_combined_array, i_dat_inf, o_dat_inf

# Farbpalette
palette = sns.color_palette("tab20", i_combined_array.shape[1]+o_combined_array.shape[1])

color_plot = []

###############################################################################
# EINGABEDATEN ################################################################
###############################################################################

# LISTE MIT DEN EINEZELNEN PLOTNAMEN ##########################################

i_list = i_dat_inf.index.tolist()

if T.Y.IMP == True:
    i_list.append("Y_sin")
    i_list.append("Y_cos")
if T.M.IMP == True:
    i_list.append("M_sin")
    i_list.append("M_cos")
if T.W.IMP == True:
    i_list.append("W_sin")
    i_list.append("W_cos")
if T.D.IMP == True:
    i_list.append("D_sin")
    i_list.append("D_cos")
if T.H.IMP == True:
    i_list.append("Holiday")

# DICTIONARY FÜR VIOLINENPLOT #################################################
df = pd.DataFrame(i_combined_array)
data = {}
for i, name in enumerate(i_list):
    values = df.iloc[:,i]
    data[name] = values

# Anzahl der Merkmale der Eingabedaten
n_ft_i = i_combined_array.shape[1]

fig, axes = plt.subplots(1,                         # Eine Zeile an Subplots
                         n_ft_i,                    # Anzahl an Subplots nebeneinander
                         figsize = (2*n_ft_i, 6))   # Größe des gesamten Plots

if len(data) <= 1:

    for i, (name, values) in enumerate(data.items()):

        sns.violinplot(y            = values,
                       ax           = axes,
                       color        = palette[i],
                       inner        = "quartile",
                       linewidth    = 1.5)

        # Titel über jedem Subplot
        axes.set_title(name)

        # Entfernen der Achsenbeschriftungen
        axes.set_xlabel("")
        axes.set_ylabel("")

else:

    # Violinplot in jeden Subplot
    for i, (name, values) in enumerate(data.items()):

        sns.violinplot(y            = values,
                       ax           = axes[i],
                       color        = palette[i],
                       inner        = "quartile",
                       linewidth    = 1.5)

        # Titel über jedem Subplot
        axes[i].set_title(name)

        # Entfernen der Achsenbeschriftungen
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

plt.suptitle("Datenverteilung \nder Eingabedaten",
             fontsize   = 15,
             fontweight = "bold")

plt.tight_layout()
plt.show()

###############################################################################
# AUSGABEDATEN ################################################################
###############################################################################

# LISTE MIT DEN EINEZELNEN PLOTNAMEN ##########################################

o_list = o_dat_inf.index.tolist()

# DICTIONARY FÜR VIOLINENPLOT #################################################
df = pd.DataFrame(o_combined_array)
data = {}
for i, name in enumerate(o_list):
    values = df.iloc[:,i]
    data[name] = values

# Anzahl der Merkmale der Ausgabedaten
n_ft_o = o_combined_array.shape[1]

fig, axes = plt.subplots(1,                         # Eine Zeile an Subplots
                         n_ft_o,                    # Anzahl an Subplots nebeneinander
                         figsize = (2*n_ft_o, 6))   # Größe des gesamten Plots

if len(data) <= 1:

    for i, (name, values) in enumerate(data.items()):

        sns.violinplot(y            = values,
                       ax           = axes,
                       color        = palette[i+n_ft_i],
                       inner        = "quartile",
                       linewidth    = 1.5)

        # Titel über jedem Subplot
        axes.set_title(name)

        # Entfernen der Achsenbeschriftungen
        axes.set_xlabel("")
        axes.set_ylabel("")

else:

    # Violinplot in jeden Subplot
    for i, (name, values) in enumerate(data.items()):

        sns.violinplot(y            = values,
                       ax           = axes[i],
                       color        = palette[i+n_ft_i],
                       inner        = "quartile",
                       linewidth    = 1.5)

        # Titel über jedem Subplot
        axes[i].set_title(name)

        # Entfernen der Achsenbeschriftungen
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

plt.suptitle("Datenverteilung \nder Ausgabedaten",
             fontsize   = 15,
             fontweight = "bold")

plt.tight_layout()
plt.show()
