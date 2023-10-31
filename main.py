import pandas as pd


def load_data():
    df = pd.read_csv("data/exoplanetes.csv", skiprows = 46)
    return df
