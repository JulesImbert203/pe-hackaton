""" Ce module propose de regrouper les différentes planètes en utilisant
l'algorithme d'apprentissage non supervisé des k moyennes """

# IMPORTATIONS

import pandas as pd
import numpy as np
import getData

DONNEES = getData.get_data()
NOMS_COLONNES = noms_colonnes_liste = DONNEES.columns.tolist()

# FONCTIONS 

def norme(planete) :
    '''
    [IN] : pandas series d'une ligne dont les colonnes sont celles de donnees 
    [OUT] : reel positif
    '''
    tableau = planete.to_numpy()
    tableau = np.square(tableau)
    somme = np.nansum(tableau)
    return np.sqrt(somme)
    

def distance(planete1, planete2):
    tab1 = planete1.to_numpy()
    tab2 = planete2.to_numpy()
    tab = np.square(tab1 - tab2)
    return np.sqrt(np.nansum(tab))


def plus_proche(nouvelle_planete, planetes) :
    '''
    Renvoie l'indice dans le tableau planetes de la planete la plus proche
    de nouvelle_planete
    '''
    k = len(planetes)
    distance_a_nouv_pl = lambda pl: distance(pl, nouvelle_planete)
    distance_a_nouv_pl = np.vectorize(distance_a_nouv_pl)
    distances = distance_a_nouv_pl(planetes)
    mini = np.inf
    id_mini
    for i in range(0, k) :
        if distances[i] < mini :
            mini = distances[i] 
            id_mini = i
    return id_mini

def moyenne(planetes):
    '''
    Renvoie la "planete moyenne" d'une liste de planete 
    [IN] : tableau de lignes de la df converties en tableau numpy
    [OUT] : un tableau numpy
    '''
    tableau_somme = np.sum(planetes)
    tableau_normes = np.array([norme(planete) for planete in planetes])
    total = np.sum(tableau_normes)
    return tableau_somme / norme


def k_moyennes(df, k, temps_max=1000) :
    '''
    Renvoie une liste de k ensembles de planètes regroupés selon l'algorithme 
    des k-moyennes. 
    '''
    # Initialisation
    planetes_k = [df.iloc[i] for i in range(0, k)]
    flag_cvg = False
    tour = 0
    long = len(df)

    # Mise a jour jusqu'a la convergence 
    while (not flag_cvg) and tour < temps_max : 
        tour += 1
        # Création de la partition de Voronoi
        groupes = [[] for i in range(0, k)]
        for i in range(0, long) :
            id_groupe = plus_proche(df.iloc[i], planetes_k)
            groupes[id_groupe].append(df.iloc[i])
        # Mise a jour de la moyenne de chaque cluster
        for i in range(0, k) : 
            planetes_k[i] = moyenne(groupes[i])
    return groupes

# +
# TESTS
# -




