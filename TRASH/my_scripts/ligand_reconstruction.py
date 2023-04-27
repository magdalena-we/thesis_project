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


from leadopt.model_conf import LeadoptModel, DIST_FN
from leadopt import grid_util
from leadopt.data_util import REC_TYPER, LIG_TYPER
from leadopt import util

path = '/projects/mai/users/kkxw544_magdalena/deepfrag_data/%s'

USE_CPU = False
device = torch.device('cpu') if USE_CPU else torch.device('cuda')


def to2d(x):
  '''Remove 3d coordinate info from a rdkit mol for display purposes.'''
  return Chem.MolFromSmiles(Chem.MolToSmiles(x, isomericSmiles=False))


def tosdf(x):
  '''Rdkit mol to SDF string.'''
  return Chem.MolToMolBlock(x)+'$$$$\n'


def iterative_ligand_reconstruction(pdb_id, resname, spec_model): 

    model = LeadoptModel.load('/home/kkxw544/deepfrag/models/' + spec_model, device=device)

    rec_coords, rec_types = util.load_receptor_ob(path % pdb_id + resname + 'rec.pdb')
    rec = Chem.MolFromPDBFile(path % pdb_id + resname + 'rec.pdb')

    lig = Chem.MolFromMolFile(path % pdb_id + resname + 'lig.sdf')

    with h5py.File('/home/kkxw544/deepfrag/fingerprints.h5', 'r') as f:
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
                torch.Tensor(f_fingerprints))

            dist = list(dist.numpy())
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


def interactive_ligand_reconstruction(pdb_id, lig_id, spec_model): 

    model = LeadoptModel.load('/home/kkxw544/deepfrag/models/' + spec_model, device=device)

    rec_coords, rec_types = util.load_receptor_ob(path % pdb_id + lig_id + 'rec.pdb')
    rec = Chem.MolFromPDBFile(path % pdb_id + lig_id + 'rec.pdb')

    lig = Chem.MolFromMolFile(path % pdb_id + lig_id + 'lig.sdf')

    img = Draw.MolsToGridImage([to2d(lig)])
    img = img.save('/home/kkxw544/deepfrag/results/' + pdb_id + lig_id + '.jpeg')

    output = subprocess.check_output(["./smina.static", "--score_only", "-r" + path % pdb_id + '.pdb', "-l" + path % pdb_id + lig_id + 'lig.sdf'], cwd='/home/kkxw544/')
    smina_old = prep_output(output)

    with h5py.File('/home/kkxw544/deepfrag/fingerprints.h5', 'r') as f:
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
                torch.Tensor(f_fingerprints))

            dist = list(dist.numpy())
            scores = list(zip(f_smiles, dist))
            scores = sorted(scores, key=lambda x:x[1])

            mols = [Chem.MolFromSmiles(x[0]) for x in scores[:20]]
            fragment = mols[0]
            lig = embed_fragment(rec, parent, fragment)
            lig_sdf = tosdf(lig)
            print(lig_sdf, file=open(path % pdb_id + lig_id + '_temp.sdf', 'w+'))
            output = subprocess.check_output(["./smina.static", "--score_only", "-r" + path % pdb_id + '.pdb', "-l" + path % pdb_id + lig_id + '_temp.sdf'], cwd='/home/kkxw544/')
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

    model = LeadoptModel.load('/home/kkxw544/deepfrag/models/' + spec_model, device=device)

    with h5py.File('/home/kkxw544/deepfrag/fingerprints.h5', 'r') as f:
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
            torch.Tensor(f_fingerprints))

        dist = list(dist.numpy())
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
            output = subprocess.check_output(["./smina.static", "--score_only", "-r" + path % prot + '.pdb', "-l" + path % 'temp2' + '.sdf'], cwd='/home/kkxw544/')
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


def draw_iterative(pdb_id, lig_id, spec_model):
    x = iterative_ligand_reconstruction(pdb_id, lig_id, spec_model)
    out = tosdf(x)
    print(out, file=open(path % pdb_id + lig_id + '_mult.sdf', 'w+'))
    output = subprocess.check_output(["./smina.static", "--score_only", "-r" + path % pdb_id + '.pdb', "-l" + path % pdb_id + lig_id + '_mult.sdf'], cwd='/home/kkxw544/')
    smina_score = prep_output(output)
    img = Draw.MolsToGridImage([to2d(x)])
    img = img.save('/home/kkxw544/deepfrag/results/' + pdb_id + lig_id + '_multilig.jpeg')

    return smina_score

def draw_interactive(pdb_id, lig_id, spec_model, savepath):
    x = interactive_ligand_reconstruction(pdb_id, lig_id, spec_model)
    out = tosdf(x)
    print(out, file=open(path % pdb_id + lig_id + '_intact.sdf', 'w+'))
    output = subprocess.check_output(["./smina.static", "--score_only", "-r" + path % pdb_id + '.pdb', "-l" + path % pdb_id + lig_id + '_intact.sdf'], cwd='/home/kkxw544/')
    smina_score = prep_output(output)
    img = Draw.MolsToGridImage([to2d(x)])
    img = img.save(savepath)

    return smina_score


def draw_topk(pdb_id, lig_id, spec_model):
    x = topk_ligand_reconstruction(pdb_id, lig_id, spec_model, k=5)
    out = tosdf(x)
    print(out, file=open(path % pdb_id + lig_id + '_new.sdf', 'w+'))
    output = subprocess.check_output(["./smina.static", "--score_only", "-r" + path % pdb_id + '.pdb', "-l" + path % pdb_id + lig_id + '_new.sdf'], cwd='/home/kkxw544/')
    output = prep_output(output)
    smina_score = prep_output(output)
    img = Draw.MolsToGridImage([to2d(x)])
    img = img.save('/home/kkxw544/deepfrag/results/' + pdb_id + lig_id + '_newtopk.jpeg')

    return smina_score


'''def main():

    protlig_list = ['1A1CDIX', '1BDUFMT', '1AMQPMP', '1AL6OAA', '1AKUSO4', '1B9TRAI', '1C7EFMN', '1C1XIPA', '1AX1BGC', '13GSSAS', '1AX1XYP', '1AX2GAL', '1BU5SO4', '1AKTFMN', '1AXZBMA', '1AXZMAN', '1BD0IN5', '1BWKFMN', '1AX0MAN', '1AX1FUC', '1AX2FUC', '1BAGGLC', '1AKVFMN', '1BNUAL3', '1BG0NO3', '1B9SFDI', '1AIQCB3', '1B4ZACT', '1BWCFAD', '1B32ACT', '1AKWFMN', '1C3EGAR', '1AVNHSM', '1AX2MAN', '1B4BARG', '1AX0A2G', '1BCDFMS', '1AIQCXM', '1AM6HAE', '1AWFGR4', '1C3ENHR', '1AZLFMN', '1BR5NEO', '1B3HALC', '1C21MET', '154LNAG', '1B11SO4', '1A1BPTR', '1BNWTPD', '1AXZGLA', '1C1XHFA', '13GSGSH', '1AXZFUC', '1B51IUM', '1AKVSO4', '1BU42GP', '10GSVWW', '1C1XPO4', '1AX0FUC', '1C1EMLT', '1AL8FMN', '1AKQFMN', '12GS0HH', '1AX2BMA', '1A2KSO4', '1BG0DAR', '1BU5RBF', '1C7FFMN', '1A1AACE', '1AX1MAN', '1AX1GAL', '1AX1BMA', '1BWCAJ3', '1B2MU34', '1C27NLP', '1AX0BMA', '1AX0XYP']

    for i in protlig_list:

        x = draw_iterative(i[:4], i[4:], 'fin1_smin5')
        print(x))
        y = draw_interactive(i[:4], i[4:], 'fin1_smin5')
        print(y)

        df = pd.read_csv('/home/kkxw544/deepfrag/results/sminascores_MOAD.csv', names=['Affinity', 'Protein', 'Ligand'], header=None)
    
        for j in range(len(df['Protein'])):
            df['Protein'][j] = df['Protein'][j] + df['Ligand'][j]
            if df['Protein'][j] == i:
                a = df['Affinity'][j]
        
        with open('/home/kkxw544/deepfrag/results/ligrec_s5.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([i, a, x, y])

    #z = draw_topk('1A0J', 'BEN', 'fin1_smin5')
    #print(z)
    #z2 = draw_topk('1A0J', 'BEN', 'fin1_smin5')
    #print(z2)
    #z1 = draw_topk('10GS', 'VWW', 'fin1_smin5')
    #print(z1)
    
    print(x, y)
    

if __name__=='__main__':
    main()'''

def main():

    #protlig_list = ['1A1CDIX', '1BDUFMT', '1AMQPMP', '1AL6OAA', '1AKUSO4', '1B9TRAI', '1C7EFMN', '1C1XIPA', '1AX1BGC', '13GSSAS', '1AX1XYP', '1AX2GAL', '1BU5SO4', '1AKTFMN', '1AXZBMA', '1AXZMAN', '1BD0IN5', '1BWKFMN', '1AX0MAN', '1AX1FUC', '1AX2FUC', '1BAGGLC', '1AKVFMN', '1BNUAL3', '1BG0NO3', '1B9SFDI', '1AIQCB3', '1B4ZACT', '1BWCFAD', '1B32ACT', '1AKWFMN', '1C3EGAR', '1AVNHSM', '1AX2MAN', '1B4BARG', '1AX0A2G', '1BCDFMS', '1AIQCXM', '1AM6HAE', '1AWFGR4', '1C3ENHR', '1AZLFMN', '1BR5NEO', '1B3HALC', '1C21MET', '154LNAG', '1B11SO4', '1A1BPTR', '1BNWTPD', '1AXZGLA', '1C1XHFA', '13GSGSH', '1AXZFUC', '1B51IUM', '1AKVSO4', '1BU42GP', '10GSVWW', '1C1XPO4', '1AX0FUC', '1C1EMLT', '1AL8FMN', '1AKQFMN', '12GS0HH', '1AX2BMA', '1A2KSO4', '1BG0DAR', '1BU5RBF', '1C7FFMN', '1A1AACE', '1AX1MAN', '1AX1GAL', '1AX1BMA', '1BWCAJ3', '1B2MU34', '1C27NLP', '1AX0BMA', '1AX0XYP']
    protlig_list = ['17GSGTX', '1BVI2GP', '1B9VRA2', '1AX2NDG', '1B4XPLP', '1B4XMAE', '1AXZMAN', '1BD0IN5']

    for i in protlig_list:

        result = [i]
        for j in range(10):
            y = draw_interactive(i[:4], i[4:], 'fin1_smin5', '/home/kkxw544/deepfrag/results/const_intact/' + str(i) + str(j) + '_intact.jpeg')
            result.append(y)
            print(y)
        
        with open('/home/kkxw544/deepfrag/results/const_intact.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(result)
    

if __name__=='__main__':
    main()


#mightwannalookatitagain = ['1B6HNVA', '1C5NTYS', '1AL6HAX', '1AIAPMP']