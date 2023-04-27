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


from leadopt.model_conf import LeadoptModel, DIST_FN
from leadopt import grid_util
from leadopt.data_util import REC_TYPER, LIG_TYPER
from leadopt import util

path = '/projects/mai/users/kkxw544_magdalena/deepfrag_data/%s


def tosdf(x):
  '''Rdkit mol to SDF string.'''
  return Chem.MolToMolBlock(x)+'$$$$\n'

def pred_affinity(rec, parent, predicted_fp):

    with h5py.File('/home/kkxw544/deepfrag/fingerprints.h5', 'r') as f:
        f_smiles = f['smiles'][()]
        f_fingerprints = f['fingerprints'][()].astype(np.float)
    
    affinities = []
    for i in range(len(pred_fp)):
        p_fingerprint = pred_fp[i]
        dist_fn = DIST_FN[model._args['dist_fn']]
        dist = dist_fn(
            torch.Tensor(p_fingerprint).unsqueeze(0),
            torch.Tensor(f_fingerprints))

        dist = list(dist.numpy())
        scores = list(zip(f_smiles, dist))
        scores = sorted(scores, key=lambda x:x[1])
    
        fragment = Chem.MolFromSmiles(scores[0][0])

        a = affinity(mols[i][0], mols[i][1], fragment)

        affinities.append(a)

    return affinities


def affinity(rec, parent, fragment):
        
    bestlig = embed_fragment(rec, parent, fragment)
    bestlig_sdf = tosdf(bestlig)
    print(bestlig_sdf, file=open(path % prot + lig + '_rec.sdf', 'w+'))
    full_output = str(subprocess.check_output(["./smina.static", "--score_only", "-r" + path % prot + '.pdb', "-l" + path % prot + lig + '_rec.sdf'], cwd='/home/kkxw544/'))
    output = full_output.strip("b'").split("\\n")
    output = [x for x in output if 'Affinity' in x]
    affinity = [float(x) for x == output[0].split()[1]]

    return affinity


        
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

#x = affinity()