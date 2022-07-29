import argparse
import pickle

from my_scripts.dfeval import evaluate_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdbs', help='path to list of protein ligand tuples')
    parser.add_argument('--spec_model', help='specify the model to be tested')
    parser.add_argument('--top_k', help='set top k to be checked')
    parser.add_argument('--res_name', help='name of the output file')


    args = parser.parse_args()
    args_dict = args.__dict__

    # Initialize.
    path_to_data = args_dict['pdbs']
    with open(path_to_data, 'rb') as f: 
        pdb_list = pickle.load(f)
    spec_model = args_dict['spec_model']
    top_k = int(args_dict['top_k'])
    res_name = args_dict['res_name']
    test = evaluate_data(pdb_list, spec_model, top_k, res_name)


if __name__=='__main__':
    main()