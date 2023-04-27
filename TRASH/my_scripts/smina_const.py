import argparse
import pickle
import subprocess
import csv
import pandas as pd
import numpy as np
import re
import gc
from rdkit import Chem
import os
import sys
import random

path = '/projects/mai/users/kkxw544_magdalena/deepfrag_data/%s'

with open('/projects/mai/users/kkxw544_magdalena/deepfrag_data/protein_ligand_complexes', 'rb') as f: 
    pdb_list = pickle.load(f)

the_list = random.sample(pdb_list, 4)

def prep_output(output):

    '''
    Filtering the output from smina.
    '''

    output = output.decode('utf-8').split('\n')
    output = [x for x in output if 'Affinity' in x]
    output = re.findall(r'(\d+\.\d+)', output[0])
    output = [float(x) for x in output]
    output = output[0]

    return output


def apply_workflow(i):

    fin_output = []
    try:
        for j in range(100):
            output = subprocess.check_output(["./smina.static", "--score_only", "-r" + path % i[0] + '.pdb', "-l" + path % i[0] + i[1] + 'lig.sdf'], cwd='/home/kkxw544/')
            fin_output.append(prep_output(output))
        mean = np.mean(fin_output)
        std = np.std(fin_output)
        var = np.var(fin_output)
        fin_output.append(i[0])
        fin_output.append(i[1])
        fin_output.append(mean)
        fin_output.append(std)
        fin_output.append(var)
    except Exception:
        print(Exception, i)

    return fin_output

def main():

    for i in the_list:
        result = apply_workflow(i)
        print(result[90:])
        with open('/home/kkxw544/deepfrag/results/smina_const.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(result)
    

if __name__=='__main__':
    main()