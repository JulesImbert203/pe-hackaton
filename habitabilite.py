import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

"""Ce module propose d'évaluer l'habitabilité des exoplanètes à partir de différents critères"""

def densite(df):
    """On calcule la densité des planètes à partir du rayon (exprimé en rayons terrestre) et de la masse en masse terrestre"""
    volume = (df["pl_rade"] * 6.4e6).pow(3) * 4/3*math.pi
    masse = (df["pl_bmasse"] * 5.792e24)

    return masse/volume


def est_rocheuse(df):
    """On utilise ce premier critère de densité simple mais peu précis"""
    return densite(df) > 3000


def critere_1(df):
    """On s'intéresse dans un premier temps aux planètes rocheuses dont la température moyenne se situe entre 0 et 100 °C"""
    return (est_rocheuse(df)) & (df["pl_eqt"] > 273) & (df["pl_eqt"] < 373)


df = pd.read_csv("data/exoplanetes.csv", skiprows = 46)
df.dropna(subset = ["pl_bmasse", "pl_rade", "pl_eqt", "st_teff"], inplace = True)

# +
sns.set_theme()
sns.scatterplot(data = df, x = "pl_orbsmax", y = "pl_eqt")

plt.xscale('log')
plt.yscale('log')
# -

df["density"] = densite(df)

# Quelle est la part de planètes telluriques?
est_rocheuse(df).sum() / len(df)


df["habitable"] = critere_1(df)

# On obtient une première borne supérieure de la proportion de planètes habitable
df["habitable"].sum()/len(df)


def critere_2(df):
    """On s'intéresse maintenant à la témpérature de l'étoile, qui définit sa durée de vie.
    On considère que seule les étoiles entre 4000 et 7000K peuvent abriter la vie.
    (source : https://fr.wikipedia.org/wiki/Habitabilit%C3%A9_d%27une_plan%C3%A8te)"""
    return (df["st_teff"] > 4000) & (df["st_teff"] < 7000)


df["habitable2"] = critere_1(df) & critere_2(df)

# On obtient une seconde estimation de la proportion de planètes habitable
df["habitable2"].sum()/len(df)


