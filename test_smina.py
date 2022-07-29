import argparse
import pickle

from my_scripts.dfeval_smina import apply_workflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdbs', help='path to list of protein ligand tuples')
    parser.add_argument('--spec_ligfn', help='specify evaluation for multi or single fragment replacement')
    parser.add_argument('--spec_model', help='specify the model to be tested')
    parser.add_argument('--res_name', help='name of the output file')
    parser.add_argument('--x', help='specify batch')


    args = parser.parse_args()
    args_dict = args.__dict__

    # Initialize.
    path_to_data = args_dict['pdbs']
    with open(path_to_data, 'rb') as f: 
        pdb_list = pickle.load(f)
    
    ligfn = args_dict['spec_ligfn']
    spec_model = args_dict['spec_model']
    res_name = args_dict['res_name']
    x = args_dict['x']
    test = apply_workflow(pdb_list, ligfn, spec_model, res_name, x)


if __name__=='__main__':
    main()