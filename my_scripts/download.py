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

from config import moad_partitions as mp


RCSB_DOWNLOAD = 'https://files.rcsb.org/download/%s.pdb'
path = '/projects/mai/users/kkxw544_magdalena/deepfrag_data/%s'


def prep_data(pdb_id): 

    try:
        urllib.request.urlretrieve(RCSB_DOWNLOAD % pdb_id, path % pdb_id + '.pdb')
    
        file = open(path % pdb_id + '.pdb', 'r')
        data = file.readlines()
        lig_list = []
        for line in data:
            if line.startswith('HET '):
                lig_list.append(line.split()[1])
        lig_list = list(set(lig_list))


    except Exception:
        print(pdb_id, 'ERROR')
        return
    
    return lig_list

def download_data(pdb_list):
    data_list = []
    count = 500
    for i in pdb_list:
        try:
            lig_list = prep_data(i)
            for j in lig_list:
                if len(j) == 3 and j != 'HOH':
                    try:
                        m = prody.parsePDB(path % i + '.pdb')
                        rec = m.select('not (nucleic or hetatm) and not water')
                        lig = m.select('resname ' + j)

                        prody.writePDB(path % i + j + 'rec.pdb', rec)
                        prody.writePDB(path % i + j + 'lig.pdb', lig)

                        conv = openbabel.OBConversion()
                        conv.OpenInAndOutFiles(path % i + j + 'lig.pdb', path % i + j + 'lig.sdf')
                        conv.SetInAndOutFormats('pdb', 'sdf')
                        conv.Convert()

                        data_list.append((i, j))
                    except Exception:
                        print('ERROR')
                        continue
            count += 1
            print(count)
        except Exception:
                        print('ERROR')
                        continue
        
    with open(path + 'protein_ligand_complexes_1', 'wb') as f:
        pickle.dump(data_list, f)                

pdb_list = mp.TRAIN + mp.VAL
x = download_data(pdb_list[500:])
