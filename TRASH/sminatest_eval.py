'''
Script for evaluating models according to smina scores.
Computing a comparable average for each model by building the intersection of protein ligand data.
'''

import numpy as np
import pandas as pd

def avg_sminascore(name_list):
    comp_list = []
    protlig_list =[]
    for i in name_list:
        #try:
        csv_file = '../results/' + i + '.csv'
    
        df = pd.read_csv(csv_file, names=['Affinity', 'Protein', 'Ligand'], header=None)
    
        for j in range(len(df['Protein'])):
            df['Protein'][j] = df['Protein'][j] + df['Ligand'][j]
    
        df = df.dropna(subset=['Affinity'])
        
        tuple_list = list(zip(df['Protein'], df['Affinity']))
        protligs = list(df['Protein'])
        comp_list.append(tuple_list)
        protlig_list.append(protligs)
        #except Exception as e:
        #    print(e)
        #    continue

    protlig_list = list(set(protlig_list[0]).intersection(*protlig_list))
    print(protlig_list)

    affinity_list = []
    for i in comp_list:
        affinities = 0
        counter = 0
        for j in i:
            if j[0] in protlig_list:
                affinities = affinities + j[1]
                counter += 1
        avg_affinity = affinities/len(protlig_list)
        affinity_list.append(avg_affinity)

    return(affinity_list, len(protlig_list)) #len(protlig_list) <=> sample size

####EXAMPLE-START#############################################################
name_list = []
x = avg_sminascore(name_list)
print(x)
####EXAMPLE-END###############################################################