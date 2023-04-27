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


from leadopt.model_conf import LeadoptModel, DIST_FN
from leadopt import grid_util
from leadopt.data_util import REC_TYPER, LIG_TYPER
from leadopt import util

def to2d(x):
  '''Remove 3d coordinate info from a rdkit mol for display purposes.'''
  return Chem.MolFromSmiles(Chem.MolToSmiles(x, isomericSmiles=False))

def fragment_reconstruction(pdb_id, resname, spec_model, top_k): 
    path = '/projects/mai/users/kkxw544_magdalena/deepfrag_data/%s'

    rec_coords, rec_types = util.load_receptor_ob(path % pdb_id + resname + 'rec.pdb')

    lig = Chem.MolFromMolFile(path % pdb_id + resname + 'lig.sdf')
    ligs = rdmolops.GetMolFrags(lig, asMols = True)
    lig = max(ligs, default=lig, key=lambda m: m.GetNumAtoms())

    frags = util.generate_fragments(lig)
    print('No.Frags: ', len(frags))
    
    inner_counter = 0
    for a in range(len(frags)):
        parent = frags[a][0]

        parent_coords = util.get_coords(parent)
        parent_types = np.array(util.get_types(parent)).reshape((-1,1))
        conn = util.get_connection_point(frags[a][1])

        USE_CPU = False

        device = torch.device('cpu') if USE_CPU else torch.device('cuda')

        model = LeadoptModel.load('/home/kkxw544/deepfrag/models/' + spec_model, device=device)

        with h5py.File('/home/kkxw544/deepfrag/fingerprints.h5', 'r') as f:
            f_smiles = f['smiles'][()]
            f_fingerprints = f['fingerprints'][()].astype(np.float)

        batch = grid_util.get_raw_batch(
            rec_coords, rec_types, parent_coords, parent_types,
            rec_typer=REC_TYPER[model._args['rec_typer']],
            lig_typer=LIG_TYPER[model._args['lig_typer']],
            conn=conn,
            num_samples=32,
            width=model._args['grid_width'],
            res=model._args['grid_res'],
            point_radius=model._args['point_radius'],
            point_type=model._args['point_type'],
            acc_type=model._args['acc_type'],
            cpu=USE_CPU
        )
        batch = torch.as_tensor(batch)
        
        pred = model.predict(batch.float()).cpu().numpy()
        avg_fp = np.mean(pred, axis=0)
        dist_fn = DIST_FN[model._args['dist_fn']]
        dist = dist_fn(
            torch.Tensor(avg_fp).unsqueeze(0),
            torch.Tensor(f_fingerprints))

        dist = list(dist.numpy())
        scores = list(zip(f_smiles, dist))
        scores = sorted(scores, key=lambda x:x[1])

        mols = [Chem.MolFromSmiles(x[0]) for x in scores[:top_k]]
        leg = ['Dist: %0.3f' % x[1] for x in scores[:top_k]]
        for x in range(len(mols)):  
            solution = str(Chem.MolToSmiles(mols[x], isomericSmiles=False)) 
            original = str(Chem.MolToSmiles(frags[a][1], isomericSmiles=False))
            #print(original, solution)
            #original = re.sub("[\[].*?[\]]", "", original)
            if original == solution:
                inner_counter += 1
                print(pdb_id, a, original, solution, x, leg[x])
    return [len(frags), inner_counter]

#model = 'final_model_retrained'
#pdb_list = [('11AS', 'ASN')]
#for a in range(len(pdb_list)):
#    x = fragment_reconstruction(pdb_list[a][0], pdb_list[a][1], model)