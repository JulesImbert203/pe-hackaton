import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


def load_data():
    df = pd.read_csv("data/exoplanetes.csv", skiprows = 46)
    return df


df = load_data()
df.head(10)

df.columns

# +
col_pl = ['pl_name', 'sy_snum', 'sy_pnum', 'pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_radj', 'pl_bmasse', 'pl_bmassj', 'pl_bmassprov', 'pl_orbeccen', 'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_metratio', 'st_logg', 'rastr', 'decstr', 'sy_dist', 'sy_vmag',
       'sy_kmag', 'sy_gaiamag']

df_pl = df[col_pl]

# +
df_pl['rastr'] = pd.to_timedelta(df_pl['rastr'])
df_pl['decstr'] = pd.to_timedelta(df_pl['decstr'])


df_pl.dtypes
# -

df_pl.head()

df_selon_mass = df.groupby(by = 'pl_bmassprov')

# +
df_list = []

for mesure,sub_df in df_selon_mass :
    df_list.append(sub_df)

distances = pdist(df_list[0][['pl_bmasse']])

# Effectuez la classification hiérarchique ascendante
Z = linkage(distances, method='ward')

# Affichez le dendrogramme pour visualiser la hiérarchie
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Classification Hiérarchique Ascendante des masses, pour la mesure simple")
plt.xlabel("Masses des planètes")
plt.ylabel("Distance (de Ward)")
plt.show()

##On pourrait classifier selon d'autres critères, c'est un exemple
