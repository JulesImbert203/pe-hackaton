""" Ce module propose de regrouper les différentes planètes en utilisant
l'algorithme d'apprentissage non supervisé des k moyennes """

# IMPORTATIONS

import pandas as pd
import numpy as np
import getData

DONNEES = getData.get_data().dropna()
NOMS_COLONNES = noms_colonnes_liste = DONNEES.columns.tolist()

# FONCTIONS 

def norme(planete) :
    '''
    [IN] : tableau numpy obtenu avec une pandas series d'une ligne dont les colonnes sont celles de donnees 
    [OUT] : reel positif
    '''
    planete = np.square(planete)
    somme = np.nansum(planete)
    return np.sqrt(somme)
    

def distance(planete1, planete2):
    tab = np.square(planete1 - planete2)
    return np.sqrt(np.nansum(tab))


def plus_proche(nouvelle_planete, planetes) :
    '''
    Renvoie l'indice dans le tableau planetes de la planete la plus proche
    de nouvelle_planete
    [IN] : (tableau_numpy, liste de tableaux numpy)
    [OUT] : int
    '''
    distances = [distance(nouvelle_planete, planete) for planete in planetes]
    id_mini = np.argmin(distances)
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
    return tableau_somme / total


def k_moyennes(df, k, temps_max=1000) :
    '''
    Renvoie une liste de k ensembles de planètes regroupés selon l'algorithme 
    des k-moyennes. 
    '''
    # Initialisation
    planetes_k = [df.iloc[i].to_numpy() for i in range(0, k)]
    flag_cvg = False
    tour = 0
    long = len(df)

    # Mise a jour jusqu'a la convergence 
    while (not flag_cvg) and tour < temps_max : 
        tour += 1
        # Création de la partition de Voronoi
        groupes = [[[],[]] for i in range(0, k)]
        for i in range(0, long) :
            id_groupe = plus_proche(df.iloc[i].to_numpy(), planetes_k)
            groupes[id_groupe][1].append(df.iloc[i].to_numpy())
            groupes[id_groupe][0].append(i)
        # Mise a jour de la moyenne de chaque cluster
        for i in range(0, k) : 
            planetes_k[i] = moyenne(groupes[i][1])
        ids = [groupes[j][0] for j in range(0, k)]
    return ids




