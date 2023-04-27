'''
Script to check top-k fragments for correlation regarding their smina-score.
'''

import argparse
import pickle
import subprocess
import pandas as pd
import numpy as np
import csv
import re
import gc
from rdkit import Chem
from rdkit.Chem import rdFMCS
import os
import sys

from ligand_reconstruction import prep_liglist

path = '/projects/mai/users/kkxw544_magdalena/deepfrag_data/%s'


def tosdf(x):
  '''Rdkit mol to SDF string.'''
  return Chem.MolToMolBlock(x)+'$$$$\n'


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


def apply_workflow(prot, lig, spec_model, k, save_path):

    '''
    Calculate average affinities for a ligand using all top-k reconstructed fragments.
    '''

    outer_list = [prot, lig]    
    with open(save_path, 'a') as f: 
        try:
            for kk in range(int(k)):  
                lig_list = prep_liglist(prot, lig, spec_model, kk)[0]
                frag_scores = []
                for ligand in lig_list:
                    lig_sdf = tosdf(ligand)
                    print(lig_sdf, file=open(path % 'temp1' + '.sdf', 'w+'))
                    output = subprocess.check_output(["./smina.static", "--score_only", "-r" + path % prot + '.pdb', "-l" + path % 'temp1' + '.sdf'], cwd='/projects/mai/users/kkxw544_magdalena/')
                    frag_scores.append(prep_output(output))
                mean = np.mean(frag_scores)
                outer_list.append(mean)
        except Exception:
            print('ERROR', prot, lig)
        finally:
            writer = csv.writer(f)
            writer.writerow(outer_list)                    
            f.flush()
            gc.collect()


#x = apply_workflow('10GS', 'VWW', 'fin1_smin2', 5, '/home/kkxw544/deepfrag/results/sminacor_s2.csv')
#print(x)   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdbs', help='path to list of protein ligand tuples')
    parser.add_argument('--model', help='specify the model to be tested')
    parser.add_argument('--save_path', help='path to save the output file')
    parser.add_argument('--k', help='specify top k number for correlation calculation')
    parser.add_argument('--x', help='specify start number pdblist')


    args = parser.parse_args()
    args_dict = args.__dict__

    # Initialize.
    path_to_data = args_dict['pdbs']
    with open(path_to_data, 'rb') as f: 
        pdb_list = pickle.load(f)

    spec_model = args_dict['model']
    save_path = args_dict['save_path']
    k = args_dict['k']
    x = int(args_dict['x'])

    for i in pdb_list[x:]:
        prot = i[0]
        lig = i[1]
        test = apply_workflow(prot, lig, spec_model, k, save_path)

if __name__=='__main__':
    main()