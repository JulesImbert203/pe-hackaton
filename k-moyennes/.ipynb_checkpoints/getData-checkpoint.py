# Obtention des donnees

import pandas as pd


# +

def get_data():
    df = pd.read_csv("../data/exoplanetes.csv", skiprows = 46)

    del df['pl_name']
    del df['hostname']
    del df['default_flag']
    del df['discoverymethod']
    del df['disc_year']
    del df['disc_facility']
    del df['soltype']
    del df['pl_controv_flag']
    del df['pl_refname']
    del df['pl_radj']
    del df['pl_bmassj']
    del df['pl_bmassprov']
    del df['ttv_flag']
    del df['st_refname']
    del df['st_spectype']
    del df['st_met']
    del df['st_metratio']
    del df['sy_refname']
    del df['rastr']
    del df['ra']
    del df['decstr']
    del df['dec']

    df = df.dropna(how='all')
    
    df = df.rename(columns={'sy_snum': 'nb_etoiles', 'sy_pnum': 'nb_planetes', 
                            'pl_orbper': 'periode_orbitale_(jours)', 'pl_orbsmax':'demi_grand_axe_(ua)',
                           'pl_rade': 'rayon', 'pl_bmasse':'masse', 'pl_orbeccen' : 'excentricite', 'pl_insol':'flux_radiatif',
                           'pl_eqt': 'temperature_(K)', 'st_teff' : 'temeprature_etoile_(K)', 'st_rad' : 'rayon_etoile', 
                           'st_mass' : 'masse_etoile', 'st_logg': 'gravite_de_surface', 'sy_dist': 'distance_(pc)'})
    df = df.iloc[:, :14]
    return df
# -




