import sys
sys.path.append('/home/kkxw544/deepfrag')
sys.path.append('/home/kkxw544/miniconda3/envs/deepfrag_1/lib/python3.7/site-packages/')
import re

import torch
import prody
import py3Dmol
from openbabel import openbabel 
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import Draw, AllChem
import numpy as np
import time
import h5py
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import urllib.request
import pickle

from leadopt.model_conf import LeadoptModel, DIST_FN
from leadopt import grid_util
from leadopt.data_util import REC_TYPER, LIG_TYPER
from leadopt import util

path = '/projects/mai/users/kkxw544_magdalena/deepfrag_data/%s'
filename = '/home/kkxw544/deepfrag/data/moad.h5'

def fragment_information(pdb_id, resname, frag_smile): 

    if resname != None:

        rec = Chem.MolFromPDBFile(path % pdb_id + resname + 'rec.pdb')

        lig = Chem.MolFromMolFile(path % pdb_id + resname + 'lig.sdf')
        ligs = rdmolops.GetMolFrags(lig, asMols = True)
        lig = max(ligs, default=lig, key=lambda m: m.GetNumAtoms())

        frags = util.generate_fragments(lig)
        print('No.Frags: ', len(frags))
    
        data_list = []
        for a in range(len(frags)):
            frag = str(Chem.MolToSmiles(frags[a][1], isomericSmiles=False)) 
            if frag == frag_smile:
                parent = frags[a][0]
                fragment = frags[a][1]
                data_list.append([rec, parent, fragment])
        
    else:
        data_list = [pdb_id, None, None]
        
    return data_list    



def test_ligand(data, pdb_list):
    
    data_list = []
    for i in range(len(data)):
        lig_smi = data[i][2]
        pdb = data[i][0]
        for j in pdb_list:
            if pdb == j[0]:
                try:
                    lig = Chem.MolFromMolFile(path % j[0] + j[1] + 'lig.sdf')
                    lig = str(Chem.MolToSmiles(lig, isomericSmiles=False))
                    if lig == lig_smi:
                        data_list.append([(pdb, j[1]), data[i][1]])
                except Exception:
                    data_list.append([(pdb, None), None])
    
    return data_list



def get_data():         
    
    data = []
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())

        # Get the data
        for i in range(0, len(f['frag_lig_idx']), 70):
            li = int(f['frag_lig_idx'][i])
            ls = f['frag_lig_smi'][li]
            lig_smile = ls.decode('utf-8')
            fs = f['frag_smiles'][i]
            smile = fs.decode('utf-8')
            fl = f['frag_lookup'][i]
            s = fl[0].decode('utf-8').split('_')
            x = s[0].upper()
            data.append([x, smile, lig_smile])
    
    return data


n = 'prot_lig_list'
with open(path % n, 'rb') as f: 
    pdb_list = pickle.load(f)

rec_par_frag = []

data = get_data()
print(len(data))

frag_data = test_ligand(data, pdb_list)
#with open('/home/kkxw544/deepfrag/results/frag_data.txt', 'w') as f:
#    f.write(str(frag_data))
print(len(frag_data))

for i in range(len(frag_data)):
    #try:
    x = fragment_information(frag_data[i][0][0], frag_data[i][0][1], frag_data[i][1])
    rec_par_frag.append(x)
    #except Exception:
    #    rec_par_frag.append([frag_data[i][0][0], None, None])
    #    continue
print(rec_par_frag, len(rec_par_frag))

rec_par_frag = np.asarray(rec_par_frag, dtype=str)

with h5py.File(filename, "r") as f:
    f.create_dataset('mol_info', data=rec_par_frag)
    print("Keys: %s" % f.keys())

m = 'rec_par_frag'
with open(path % m, 'wb') as f:
    pickle.dump(rec_par_frag, f)

