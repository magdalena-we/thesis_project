import torch
import prody
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import Draw, AllChem
import numpy as np
import h5py
import urllib.request
import gc
import subprocess
import re
import csv
import pandas as pd
import argparse
import pickle


from leadopt.model_conf import LeadoptModel, DIST_FN
from leadopt import grid_util
from leadopt.data_util import REC_TYPER, LIG_TYPER
from leadopt import util

path = './pdb_data/%s'

USE_CPU = True
device = torch.device('cpu') if USE_CPU else torch.device('cuda')


def to2d(x):
  '''Remove 3d coordinate info from a rdkit mol for display purposes.'''
  return Chem.MolFromSmiles(Chem.MolToSmiles(x, isomericSmiles=False))


def tosdf(x):
  '''Rdkit mol to SDF string.'''
  return Chem.MolToMolBlock(x)+'$$$$\n'


def iterative_ligand_reconstruction(pdb_id, resname, spec_model): 

    '''
    Ligand reconstruction by applying the predicted top fragment to the parent.
    This process is repeated until the predicted top fragments do not differ
    from the 'original' fragment.
    '''

    model = LeadoptModel.load(spec_model, device=device)

    rec_coords, rec_types = util.load_receptor_ob(path % pdb_id + resname + 'rec.pdb')
    rec = Chem.MolFromPDBFile(path % pdb_id + resname + 'rec.pdb')

    lig = Chem.MolFromMolFile(path % pdb_id + resname + 'lig.sdf')

    with h5py.File('./fingerprints.h5', 'r') as f:
        f_smiles = f['smiles'][()]
        f_fingerprints = f['fingerprints'][()].astype(np.float)
    
    k = True
    while k == True:
        ligs = rdmolops.GetMolFrags(lig, asMols = True)
        lig = max(ligs, default=lig, key=lambda m: m.GetNumAtoms())

        frags = util.generate_fragments(lig)

        for a in range(len(frags)):
            parent = frags[a][0]

            parent_coords = util.get_coords(parent)
            parent_types = np.array(util.get_types(parent)).reshape((-1,1))
            conn = util.get_connection_point(frags[a][1])

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
                torch.Tensor(f_fingerprints)).cpu()

            dist = list(dist.cpu().numpy())
            scores = list(zip(f_smiles, dist))
            scores = sorted(scores, key=lambda x:x[1])

            mols = [Chem.MolFromSmiles(x[0]) for x in scores[:20]]
            fragment = mols[0] 
            solution = str(Chem.MolToSmiles(mols[0], isomericSmiles=False)) 
            orig = Chem.RemoveHs(frags[a][1])
            original = str(Chem.MolToSmiles(orig, isomericSmiles=False))

            kk = True
            if solution != original:
                kk = True
                print(original, solution)
                lig = embed_fragment(rec, parent, fragment)
                break
            else:
                kk = False

            print(original, solution)
        
        if kk == False:
            bestlig = lig
            k = False

    return bestlig


def interactive_ligand_reconstruction(pdb_id, lig_id, spec_model, save_path): 

    '''
    Ligand reconstruction by adding the top predicted fragment to the parent
    and comparing wether it scores higher than the original ligand on 
    application of smina. If it does the predicted fragment is kept, otherwise
    the original ligand will be processed further. In the second step another 
    fragment is contemplated. This step will be repeated untli all of the 
    ligands fragments have been checked.
    '''

    model = LeadoptModel.load(spec_model, device=device)

    rec_coords, rec_types = util.load_receptor_ob(path % pdb_id + lig_id + 'rec.pdb')
    rec = Chem.MolFromPDBFile(path % pdb_id + lig_id + 'rec.pdb')

    lig = Chem.MolFromMolFile(path % pdb_id + lig_id + 'lig.sdf')

    img = Draw.MolsToGridImage([to2d(lig)])
    img = img.save(save_path + pdb_id + lig_id + '.jpeg')

    output = subprocess.check_output(["./smina.static", "--score_only", "-r" + path % pdb_id + '.pdb', "-l" + path % pdb_id + lig_id + 'lig.sdf'])
    smina_old = prep_output(output)

    with h5py.File('./fingerprints.h5', 'r') as f:
        f_smiles = f['smiles'][()]
        f_fingerprints = f['fingerprints'][()].astype(np.float)
    
    k = True
    while k == True:
        ligs = rdmolops.GetMolFrags(lig, asMols = True)
        lig = max(ligs, default=lig, key=lambda m: m.GetNumAtoms())

        frags = util.generate_fragments(lig)

        for a in range(len(frags)):
            parent = frags[a][0]

            parent_coords = util.get_coords(parent)
            parent_types = np.array(util.get_types(parent)).reshape((-1,1))
            conn = util.get_connection_point(frags[a][1])

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
                torch.Tensor(f_fingerprints)).cpu()

            dist = list(dist.cpu().numpy())
            scores = list(zip(f_smiles, dist))
            scores = sorted(scores, key=lambda x:x[1])

            mols = [Chem.MolFromSmiles(x[0]) for x in scores[:20]]
            fragment = mols[0]
            lig = embed_fragment(rec, parent, fragment)
            lig_sdf = tosdf(lig)
            print(lig_sdf, file=open(path % pdb_id + lig_id + '_temp.sdf', 'w+'))
            output = subprocess.check_output(["./smina.static", "--score_only", "-r" + path % pdb_id + '.pdb', "-l" + path % pdb_id + lig_id + '_temp.sdf'])
            smina_new = prep_output(output)

            kk = True
            if smina_new > smina_old:
                smina_old = smina_new
                kk = True
                break
            else:
                kk = False
        
        if kk == False:
            bestlig = lig
            k = False
        
    #bestlig_sdf = tosdf(bestlig)

    return bestlig


def prep_liglist(pdb_id, lig_id, spec_model, k): 

    '''Helper function, prepares a list containing data about the ligand and 
    its top predicted fragments. This function is used to calculate the 
    correlation of the smina scores of the top predicted fragments.'''

    model = LeadoptModel.load(spec_model, device=device)

    with h5py.File('./fingerprints.h5', 'r') as f:
        f_smiles = f['smiles'][()]
        f_fingerprints = f['fingerprints'][()].astype(np.float)

    rec_coords, rec_types = util.load_receptor_ob(path % pdb_id + lig_id + 'rec.pdb')
    rec = Chem.MolFromPDBFile(path % pdb_id + lig_id + 'rec.pdb')

    lig = Chem.MolFromMolFile(path % pdb_id + lig_id + 'lig.sdf')
    
    ligs = rdmolops.GetMolFrags(lig, asMols = True)
    lig = max(ligs, default=lig, key=lambda m: m.GetNumAtoms())

    frags = util.generate_fragments(lig)
    
    lig_list =[[], []]
    for a in range(len(frags)):
        parent = frags[a][0]

        parent_coords = util.get_coords(parent)
        parent_types = np.array(util.get_types(parent)).reshape((-1,1))
        conn = util.get_connection_point(frags[a][1])

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
            torch.Tensor(f_fingerprints))#.cuda()

        dist = list(dist.cpu().numpy())
        scores = list(zip(f_smiles, dist))
        scores = sorted(scores, key=lambda x:x[1])

        mols = [Chem.MolFromSmiles(x[0]) for x in scores[:20]]
        fragment = mols[k] 

        lig = embed_fragment(rec, parent, fragment)
        lig_list[0].append(lig)
        lig_list[1].append([rec, parent, fragment])
        gc.collect()

    return lig_list


def topk_ligand_reconstruction(prot, lig, spec_model, k): 

    '''
    Not  yet functional.
    This method was meant to choose the top scoring fragment out of a given
    range of top-k fragments and build a new ligand with these.
    '''
    
    lig_list = []
    data_list = [] 
    for kk in range(int(k)):
        temp_list = prep_liglist(prot, lig, spec_model, kk)  
        ligs = temp_list[0]
        data = temp_list[1]
        lig_list.append(ligs)
        data_list.append(data)
        
    scores = []
    for i in range(len(lig_list[0])):
        frag_scores = []
        for j in range(len(lig_list)):
            lig_sdf = tosdf(lig_list[j][i])
            print(lig_sdf, file=open(path % 'temp2' + '.sdf', 'w+'))
            output = subprocess.check_output(["./smina.static", "--score_only", "-r" + path % prot + '.pdb', "-l" + path % 'temp2' + '.sdf'], cwd='/projects/mai/users/kkxw544_magdalena/')
            frag_scores.append(prep_output(output))
        scores.append(frag_scores)
    
    index_list = []
    for i in scores:
        index_max = np.argmax(i)
        index_list.append(index_max)

    mols = []
    for i in data_list[0]:
        mols.append(i[1])
        res = rdFMCS.FindMCS(mols,timeout=10,bondCompare=rdkit.Chem.rdFMCS.BondCompare.CompareAny,atomCompare = rdkit.Chem.rdFMCS.AtomCompare.CompareIsotopes )
        parent = Chem.MolFromSmarts( res.smartsString)
    
    rec = data_list[0][0]
    i = 0
    while i < len(lig_list):
        for j in index_list:
            try: 
                new_lig = embed_fragment(rec, parent, lig_list[i][j])
                parent = new_lig
            except Exception:
                continue
            i += 1
            
        
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
    for cid in cids:

        print(' ', flush=True)

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
        for deg in range(0,360,5):
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


def draw_iterative(pdb_id, lig_id, spec_model, save_path):

    '''
    Using the reconstruction function to save the new ligand as sdf file
    and a drawing of its structure.
    '''

    x = iterative_ligand_reconstruction(pdb_id, lig_id, spec_model)
    out = tosdf(x)
    print(out, file=open(path % pdb_id + lig_id + '_mult.sdf', 'w+'))
    output = subprocess.check_output(["./smina.static", "--score_only", "-r" + path % pdb_id + '.pdb', "-l" + path % pdb_id + lig_id + '_mult.sdf'])
    smina_score = prep_output(output)
    img = Draw.MolsToGridImage([to2d(x)])
    img = img.save(save_path + pdb_id + lig_id + '_multilig.jpeg')

    return smina_score

def draw_interactive(pdb_id, lig_id, spec_model, save_path):

    '''
    Using the reconstruction function to save the new ligand as sdf file
    and a drawing of its structure.
    '''

    x = interactive_ligand_reconstruction(pdb_id, lig_id, spec_model, save_path)
    out = tosdf(x)
    print(out, file=open(path % pdb_id + lig_id + '_intact.sdf', 'w+'))
    output = subprocess.check_output(["./smina.static", "--score_only", "-r" + path % pdb_id + '.pdb', "-l" + path % pdb_id + lig_id + '_intact.sdf'])
    smina_score = prep_output(output)
    img = Draw.MolsToGridImage([to2d(x)])
    img = img.save(save_path + pdb_id + lig_id + '_intact.jpeg')

    return smina_score


def draw_topk(pdb_id, lig_id, spec_model, save_path):

    '''
    Using the reconstruction function to save the new ligand as sdf file
    and a drawing of its structure.
    '''
    
    x = topk_ligand_reconstruction(pdb_id, lig_id, spec_model, k=5)
    out = tosdf(x)
    print(out, file=open(path % pdb_id + lig_id + '_new.sdf', 'w+'))
    output = subprocess.check_output(["./smina.static", "--score_only", "-r" + path % pdb_id + '.pdb', "-l" + path % pdb_id + lig_id + '_new.sdf'])
    output = prep_output(output)
    smina_score = prep_output(output)
    img = Draw.MolsToGridImage([to2d(x)])
    img = img.save(save_path + pdb_id + lig_id + '_newtopk.jpeg')

    return smina_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdbs', help='path to list of protein ligand tuples')
    parser.add_argument('--save_path', help='path to save the output folder')
    parser.add_argument('--model', help='specify the model to be tested')

    args = parser.parse_args()
    args_dict = args.__dict__

    save_path = args_dict['save_path']
    spec_model = args_dict['model']

    # Initialize.
    path_to_data = args_dict['pdbs']
    with open(path_to_data, 'rb') as f: 
        pdb_list = pickle.load(f)
    
    for i in pdb_list[1:]:
        
        #try:
        x = draw_iterative(i[0], i[1], spec_model, save_path)
        print(x)
        y = draw_interactive(i[0], i[1], spec_model, save_path)
        print(y)
            #z = draw_topk(i[0], i[1], spec_model, save_path)
            #print(z)
        #except Exception:
        #   continue

        df = pd.read_csv(save_path + 'smina_scores.csv', names=['Affinity', 'Protein', 'Ligand'], header=None)

        a=0 
        for j in range(len(df['Protein'])):
            df['Protein'][j] = df['Protein'][j] + df['Ligand'][j]
            if df['Protein'][j] == i:
                a = df['Affinity'][j]
        
        with open(save_path + '.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([i, a, x, y])


if __name__=='__main__':
    main()