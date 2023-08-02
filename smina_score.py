'''
Script for scoring protein-ligand-complexes for their binding affinity,
using smina.
'''

import subprocess
import argparse
import pickle
import re
import csv
import sys 
import gc

path = './pdb_data/%s'

def affinity_scores(pdb_list, save_path):

    '''
    Calling on smina and appending desired output to a dataframe.
    '''
    with open(save_path, 'a') as f:
        for i in pdb_list:
            prot = i[0]
            lig = i[1]
            fin_output = []
            try:
                output = subprocess.check_output(["./smina.static", "--score_only", "-r" + path % prot + '.pdb', "-l" + path % prot + lig + 'lig.sdf'])
                fin_output.append(prep_output(output))
                fin_output.append(prot)
                fin_output.append(lig)
            except Exception:
                print(prot, lig, 'ERROR', flush=True)
                continue
            finally:
                writer = csv.writer(f)
                writer.writerow(fin_output)
                f.flush()
                gc.collect()


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdbs', help='path to list of protein ligand tuples')
    parser.add_argument('--save_path', help='path to save the output file')


    args = parser.parse_args()
    args_dict = args.__dict__

    # Initialize.
    path_to_data = args_dict['pdbs']
    with open(path_to_data, 'rb') as f: 
        pdb_list = pickle.load(f)
    save_path = args_dict['save_path']
    scores = affinity_scores(pdb_list, save_path)


if __name__=='__main__':
    main()