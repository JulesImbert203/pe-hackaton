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
df.dropna(subset = ["pl_bmasse", "pl_rade", "pl_eqt"], inplace = True)

df["density"] = densite(df)

# Quelle est la part de planètes telluriques?
est_rocheuse(df).sum() / len(df)


# +
sns.set_theme()
sns.scatterplot(data = df, x = "pl_insol", y = "pl_eqt")

plt.xscale('log')
plt.yscale('log')
# -

df["habitable"] = critere_1(df)

# On obtient une première borne supérieure de la proportion de planètes habitable
df["habitable"].sum()/len(df)




