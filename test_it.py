import torch
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np
import h5py
import argparse

from leadopt.metrics import cos, top_k_acc


def test(predicted_fp, correct_fp):

    correct_fp = torch.tensor(np.load(correct_fp))
    
    pred_fp = torch.tensor(np.load(predicted_fp))
    
    accuracy = top_k_acc(correct_fp, cos, [1, 8, 64], pre="")(pred_fp, correct_fp)
    print(accuracy)
    return

#x = test('/home/kkxw544/deepfrag/models/final_model_retrained_05/predicted_fp.npy', '/home/kkxw544/deepfrag/models/final_model_retrained_05/correct_fp.npy')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--correct_fp_file')
    parser.add_argument('--predicted_fp_file')

    args = parser.parse_args()
    args_dict = args.__dict__
    
    correct_fp = args.correct_fp_file
    predicted_fp = args.predicted_fp_file
    x = test(predicted_fp, correct_fp)
    print(x)


if __name__=='__main__':
    main()