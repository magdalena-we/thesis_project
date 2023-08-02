'''
Script for evaluating models according to smina scores.
Affinity for protein-reconstructed-ligand-complexes are stored in csv-files.
'''

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

from ligand_reconstruction import prep_liglist

path = './pdb_data/%s'


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


def apply_workflow(prot, lig, spec_model, save_path):

    '''creates sdf-files with parents and their top reconstructed fragment,
    scoring them with smina and saving the result to an csv-file'''
    
    fin_output = []
    with open(save_path, 'a') as f:
        try:
            bestlig_list = prep_liglist(prot, lig, spec_model, k=0)[0]
            for bestlig in bestlig_list:
                bestlig_sdf = tosdf(bestlig)
                print(bestlig_sdf, file=open(path % prot + lig + '_rec.sdf', 'w+'), flush=True)
                output = subprocess.check_output(["./smina.static", "--score_only", "-r" + path % prot + '.pdb', "-l" + path % prot + lig + '_rec.sdf'], cwd='./')
                fin_output.append(prep_output(output))
                print(fin_output)
            mean = np.mean(fin_output)
            fin_output.clear()   
            fin_output.append(mean)
            fin_output.append(prot)
            fin_output.append(lig)
        except Exception:
            print('ERROR', prot, lig)
        finally:
            writer = csv.writer(f)
            writer.writerow(fin_output)
            print('.')
            f.flush()
            gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdbs', help='path to list of protein ligand tuples')
    parser.add_argument('--model', nargs="*", help='specify the model to be tested')
    parser.add_argument('--save_path', nargs="*", help='path to save the output file')
    parser.add_argument('--x', help='specify batch')


    args = parser.parse_args()
    args_dict = args.__dict__

    # Initialize.
    path_to_data = args_dict['pdbs']
    with open(path_to_data, 'rb') as f: 
        pdb_list = pickle.load(f)
    
    spec_model = args_dict['model']
    save_path = args_dict['save_path']
    x = int(args_dict['x'])

    for i in range(x, len(pdb_list), 25):
        prot = pdb_list[i][0]
        lig = pdb_list[i][1]
        for m in range(len(spec_model)):
            test = apply_workflow(prot, lig, spec_model[m], save_path[m])


if __name__=='__main__':
    main()