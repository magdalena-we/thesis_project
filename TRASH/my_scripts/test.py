'''
Script for evaluation of DeepFrag models.
'''

import argparse
import json
import torch
import numpy as np
import os

from leadopt.metrics import cos, top_k_acc
from leadopt.model_conf import LeadoptModel, DIST_FN, MODELS
from leadopt.data_util import FingerprintDataset


def load_all_fingerprints(args_dict, model):

    print("Loading full dataset")
    train_dat, val_dat = model.load_data()
    fingerprints = FingerprintDataset('/home/kkxw544/deepfrag/data/rdk10_moad.h5')

    train_smiles = train_dat.get_valid_smiles()
    val_smiles = val_dat.get_valid_smiles()

    all_smiles = list(set(train_smiles) | set(val_smiles))

    return fingerprints.for_smiles(all_smiles).cuda()


def test(predicted_fp, correct_fp, all_fp):

    #load the fingerprint arrays from npy-files computed by the run_test method 
    correct_fp = torch.tensor(np.load(correct_fp)).cuda()
    pred_fp = torch.tensor(np.load(predicted_fp)).cuda()
    
    #call on the top-k accuracy function from leadopt-metrics
    accuracy = top_k_acc(all_fp, cos, [1, 8, 64], pre="")(pred_fp, correct_fp)
    
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_test', default=False, action="store_true", help='')
    parser.add_argument('--save_path', help='location to save the model')
    parser.add_argument('--configuration', help='specify the model to be used for testing')
    parser.add_argument('--samples_per_example', type=int)

    subparsers = parser.add_subparsers(dest='version')

    for m in MODELS:
        sub = subparsers.add_parser(m)
        MODELS[m].setup_parser(sub)

    args = parser.parse_args()
    args_dict = args.__dict__

 
    # Initialize model.
    model_type = args_dict['version']
    model = LeadoptModel.load(args.configuration, device=torch.device('cuda'))
    
    #call on run_test method from model conf, saves file of predicted and correct fingerprint arrays
    if args.run_test:
        model.run_test(args.save_path, args.samples_per_example)
    
    all_fp = all_fps = load_all_fingerprints(args_dict, model)
    correct_fp = os.path.join(args.save_path, 'correct_fp.npy')
    predicted_fp = os.path.join(args.save_path, 'predicted_fp.npy')
    
    acc = test(predicted_fp, correct_fp, all_fp)

    print(acc)

if __name__=='__main__':
    main()