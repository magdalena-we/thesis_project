'''
Script to download and prepare data for applying evaluation and reconstruction methods
'''


import prody
from openbabel import openbabel
import urllib.request
import pickle

from config import moad_partitions as mp


RCSB_DOWNLOAD = 'https://files.rcsb.org/download/%s.pdb'
path = '/projects/mai/users/kkxw544_magdalena/deepfrag_data/%s'


def download_data(pdb_id): 

    '''
    Method to download specified pdb-files and extract ligand-ids
    '''

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

def prep_data(pdb_list):

    '''
    Method to prepare data for evaluation, seperate receptor and filter out water molecules,
    save receptor data as pdb file and ligand data as sdf file.
    Also creates a list of protein ligand tuples that can be used as an input for various methods.
    '''

    #count allows you to continue script from certain position if it aborts
    count = 0
    data_list = []
    for i in pdb_list:
        try:
            lig_list = download_data(i)
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
        
    with open(path + 'protein_ligand_complexes', 'wb') as f:
        pickle.dump(data_list, f)                

pdb_list = mp.TRAIN + mp.VAL
x = prep_data(pdb_list)