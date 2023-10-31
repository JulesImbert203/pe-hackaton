import pandas as pd
import numpy as np

""" Ce module propose de regrouper les différentes planètes en utilisant
l'algorithme d'apprentissage non supervisé des k moyennes """

def norme(planete) :
    pass

def distance(planete1, planete2):
    pass


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
    [IN] : tableau de lignes de la df
    [OUT] : un objet de type `pandas.Series`.
    '''
    pass
    

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
