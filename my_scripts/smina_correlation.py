import sys
sys.path.append('/home/kkxw544/deepfrag')
sys.path.append('/home/kkxw544/anaconda3/envs/deepfrag/lib/python3.7/site-packages/')

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
from tqdm import tqdm
import matplotlib.pyplot as plt
import urllib.request
import subprocess
import pandas as pd


from leadopt.model_conf import LeadoptModel, DIST_FN
from leadopt import grid_util
from leadopt.data_util import REC_TYPER, LIG_TYPER
from leadopt import util

path = '/projects/mai/users/kkxw544_magdalena/deepfrag_data/%s'

USE_CPU = False
device = torch.device('cpu') if USE_CPU else torch.device('cuda')

with h5py.File('/home/kkxw544/deepfrag/fingerprints.h5', 'r') as f:
    f_smiles = f['smiles'][()]
    f_fingerprints = f['fingerprints'][()].astype(np.float)


def to2d(x):
  '''Remove 3d coordinate info from a rdkit mol for display purposes.'''
  return Chem.MolFromSmiles(Chem.MolToSmiles(x, isomericSmiles=False))

def tosdf(x):
  '''Rdkit mol to SDF string.'''
  return Chem.MolToMolBlock(x)+'$$$$\n'

def fragment_reconstruction(pdb_id, resname, spec_model, top_k): 

    rec_coords, rec_types = util.load_receptor_ob(path % pdb_id + resname + 'rec.pdb')
    rec = Chem.MolFromPDBFile(path % pdb_id + resname + 'rec.pdb')

    lig = Chem.MolFromMolFile(path % pdb_id + resname + 'lig.sdf')
    ligs = rdmolops.GetMolFrags(lig, asMols = True)
    lig = max(ligs, default=lig, key=lambda m: m.GetNumAtoms())

    frags = util.generate_fragments(lig)
    
    bestlig_list = []
    for a in range(len(frags)):
        parent = frags[a][0]

        parent_coords = util.get_coords(parent)
        parent_types = np.array(util.get_types(parent)).reshape((-1,1))
        conn = util.get_connection_point(frags[a][1])

        model = LeadoptModel.load('/home/kkxw544/deepfrag/models/' + spec_model, device=device)

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

        mols = [Chem.MolFromSmiles(x[0]) for x in scores[:20]]
        bestlig_list_u = []
        for i in range(top_k)
            fragment = mols[i]
            bestlig = embed_fragment(rec, parent, fragment)
            bestlig_list_u.append(bestlig)
        bestlig_list.append(bestlig_list_u)    
        
    return bestlig_list


def geometric_embedding(fragment):
 
    frag = to2d(fragment)

    # Temporarily replace dummy atoms with hydrogen so we get reasonable geometry.
    dummy_idx = [x.GetIdx() for x in fragment.GetAtoms() if x.GetAtomicNum() == 0]
    for idx in dummy_idx:
        frag.GetAtomWithIdx(idx).SetAtomicNum(1)

    # Minimize engergy.
    frag = Chem.AddHs(frag)
    cids = AllChem.EmbedMultipleConfs(frag, 50, pruneRmsThresh=1)
    for conf in cids:
        AllChem.UFFOptimizeMolecule(frag, confId=conf, maxIters=200)

    # Replace any dummy atoms.
    for idx in dummy_idx:
        frag.GetAtomWithIdx(idx).SetAtomicNum(0)

    return frag, cids

def get_connecting_atoms(mol):
    connectidx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0][0]
    atm = mol.GetAtomWithIdx(connectidx)
    nextatm = atm.GetNeighbors()[0]
    nextidx = nextatm.GetIdx()
    nextnextidx = [a for a in nextatm.GetNeighbors() if a.GetIdx() != connectidx][0].GetIdx()
    return connectidx, nextidx, nextnextidx

def embed_fragment(rec, parent, fragment):

    energies = []
    
    Chem.SanitizeMol(rec)
    fragment, cids = geometric_embedding(fragment)

    # Find the dihedral
    paridx, parnext, parnextnext = get_connecting_atoms(parent)
  
    best_energy = np.inf
    best_mol = None

    # For each conformer...
    for cid in tqdm(cids, desc='Sampling conformations'):
        mol = Chem.RWMol(fragment, False, cid)

        # Align the connection point.
        fragidx, fragnext, fragnextnext = get_connecting_atoms(mol)
        Chem.rdMolAlign.AlignMol(
            mol, parent, atomMap=[(fragidx,parnext),(fragnext,paridx)])

        # Merge into new molecule.
        merged = Chem.RWMol(Chem.CombineMols(parent,mol))

        # Update fragment indices.
        fragidx += parent.GetNumAtoms()
        fragnext += parent.GetNumAtoms()
        fragnextnext += parent.GetNumAtoms()
        bond = merged.AddBond(parnext,fragnext,Chem.rdchem.BondType.SINGLE)
        merged.RemoveAtom(fragidx)
        merged.RemoveAtom(paridx)
        Chem.SanitizeMol(merged)

        # Update indices to account for deleted atoms.
        if fragnext > fragidx: fragnext -= 1
        if fragnextnext > fragidx: fragnextnext -= 1
        fragnext -= 1
        fragnextnext -= 1
        if parnext > paridx: parnext -= 1
        if parnextnext > paridx: parnextnext -= 1

        # Optimize the connection of the fragment (bond is wrong length).
        ff = AllChem.UFFGetMoleculeForceField(merged)
        for p in range(parent.GetNumAtoms()-1): # Don't include dummy atom.
            ff.AddFixedPoint(p) # Don't move parent.
        ff.Minimize()

        # Create a complex with the receptor.
        reclig = Chem.CombineMols(rec, merged)
        Chem.SanitizeMol(reclig)

        # Determine dihedral indices.
        l = fragnextnext+rec.GetNumAtoms()
        k = fragnext+rec.GetNumAtoms()
        j = parnext+rec.GetNumAtoms()
        i = parnextnext+rec.GetNumAtoms()

        # Sample the dihedral.
        for deg in tqdm(range(0,360,5), desc='Sampling dihedral angle'):
            Chem.rdMolTransforms.SetDihedralDeg(reclig.GetConformer(),i,j,k,l,deg)

            # Create forcefield for the whole complex.
            ff = AllChem.UFFGetMoleculeForceField(reclig,ignoreInterfragInteractions=False)

            # Fix everything but the fragment.
            for p in range(rec.GetNumAtoms()+parent.GetNumAtoms()-1):
                ff.AddFixedPoint(p)    
            energy = ff.CalcEnergy()

            energies.append(energy)
            if energy < best_energy:
                best_energy = energy
                best_mol = Chem.RWMol(reclig)

    # Extract the best ligand.
    ligatoms = set(range(rec.GetNumAtoms(), best_mol.GetNumAtoms()))
    ligbonds = [b.GetIdx() for b in best_mol.GetBonds() if b.GetBeginAtomIdx() in ligatoms and b.GetEndAtomIdx() in ligatoms]
    bestlig = Chem.PathToSubmol(best_mol, ligbonds)
    
    return bestlig

def apply_workflow(pdb_list, spec_model, res_name, top_k, x):
    x = int(x)
    pdb_list = pdb_list[x*100:x*100+500]
    fin_out_list = []
    for i in pdb_list:
        prot = i[0]
        lig = i[1]
        try:
            bestlig_list = fragment_reconstruction(prot, lig, spec_model, top_k)
            k_fin_out_list = []
            for k in range(top_k):
                fin_output = []
                for j in range(len(bestlig_list)):
                    bestlig_sdf = tosdf(bestlig_list[j][k])
                    print(bestlig_sdf, file=open(path % prot + lig + '_rec.sdf', 'w+'))
                    full_output = str(subprocess.check_output(["./smina.static", "--score_only", "-r" + path % prot + '.pdb', "-l" + path % prot + lig + '_rec.sdf'], cwd='/home/kkxw544/'))
                    final_output = prep_output(full_output)
                    fin_output.append(final_output)
                mean = np.mean(fin_output)
                fin_output.clear()   
                append = fin_output.append
                append(mean)
                append(prot)
                append(lig) 
                k_fin_out_list.append(fin_output)
            fin_out_list.append(fin_output)
        except Exception:
            print('ERROR')
            continue
    

def prep_output(output):
    output_1 = output.decode('utf-8').split('\\n')
    output_2 = [x for x in output_1 if 'Affinity' in x]
    output_3 = output_2.extract(r'(\d+.\d+)').astype('float')

    return output_3

    df = pd.DataFrame(fin_out_list)
    df.to_csv('/home/kkxw544/deepfrag/results/' + res_name + '.csv', sep = ',')