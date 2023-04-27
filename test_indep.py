'''
Independent script for evaluation of deepfrag-models.
Can be used to test models on unknown/new protein data.
'''

import pandas as pd
import torch 
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np
import h5py
import argparse
import gc
import pickle

from leadopt.model_conf import LeadoptModel, DIST_FN
from leadopt import grid_util
from leadopt.data_util import REC_TYPER, LIG_TYPER
from leadopt import util


#path to input data
path = './pdb_data/%s'

#Choose device
USE_CPU = True
device = torch.device('cpu') if USE_CPU else torch.device('cuda')

#open the fragment-fingerprint file and load fragment-smiles and -fingerprints for comparison
with h5py.File('./fingerprints.h5', 'r') as f:
    f_smiles = f['smiles'][()]
    f_fingerprints = f['fingerprints'][()].astype(np.float)


def apply_workflow(pdb_list, spec_model, top_k):

    '''
    Initialization of the model, iteration over the input and computation of overall accuracy
    '''

    model = LeadoptModel.load(spec_model, device=device)

    data_list = []
    outer_counter = 0
    total_frags = 0
    for i in pdb_list:
        prot = i[0]
        lig = i[1]
        try:
            x = fragment_reconstruction(prot, lig, model, top_k)
            frags = x[0]
            total_frags = total_frags + frags
            inner_counter = x[1]
            outer_counter = outer_counter + inner_counter
            accuracy = inner_counter/frags
            total_accuracy = outer_counter/total_frags
            data_list.append([frags, inner_counter, accuracy, outer_counter, total_frags, total_accuracy])
            print('Sim.Count: ', inner_counter)
            print('TotalFrags: ', total_frags, 'TotalSims: ', outer_counter, 'Accuracy: ', 100*total_accuracy)
            gc.collect()
        except Exception:
            print('ERROR')
            continue    

    return data_list


def evaluate_data(pdb_list, spec_model, top_k, output_name):

    '''
    Saving the output as a csv-file
    '''

    data_list = apply_workflow(pdb_list, spec_model, top_k)
    df = pd.DataFrame(data_list, columns = ['generated_fragments', 'reconstructed_fragments', 'accuracy', 'total_recfrags', 'total_genfrags', 'total_accuracy'])
    df.to_csv(output_name, sep  = ',')


def fragment_reconstruction(pdb_id, lig_id, model, top_k): 

    '''
    Fragmentation, reconstruction and comparison
    '''

    rec_coords, rec_types = util.load_receptor_ob(path % pdb_id + lig_id + 'rec.pdb')

    lig = Chem.MolFromMolFile(path % pdb_id + lig_id + 'lig.sdf')
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
        
        #sorting all fragments according to their similarity to the predicted fingerprint
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
        
        #comparison of the original fragment with the top k results for the reconstructed fragment
        for x in range(len(mols)):  
            solution = str(Chem.MolToSmiles(mols[x], isomericSmiles=False)) 
            original = str(Chem.MolToSmiles(frags[a][1], isomericSmiles=False))

            if original == solution:
                inner_counter += 1
            #comment to reduce command line output
            print(pdb_id, a, original, solution, x, leg[x])
    
    return [len(frags), inner_counter]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdbs', help='path to list of protein ligand tuples')
    parser.add_argument('--model', help='specify the model to be tested')
    parser.add_argument('--top_k', help='set top k to be checked')
    parser.add_argument('--save_path', help='name of the output file')


    args = parser.parse_args()
    args_dict = args.__dict__

    # Initialize.
    path_to_data = args_dict['pdbs']
    with open(path_to_data, 'rb') as f: 
        pdb_list = pickle.load(f)
    spec_model = args_dict['model']
    top_k = int(args_dict['top_k'])
    output_name = args_dict['save_path']
    test = evaluate_data(pdb_list, spec_model, top_k, output_name)


if __name__=='__main__':
    main()