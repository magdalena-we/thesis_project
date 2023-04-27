import torch
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np
import h5py

from leadopt.metrics import cos


def test(test_file, smiles_file):

    correct_smi = np.load(smiles_file)
    
    pred_fp = np.load(test_file)
    pred_fp = np.mean(pred_fp, axis=1)

    with h5py.File('/home/kkxw544/deepfrag/fingerprints.h5', 'r') as f:
            f_smiles = f['smiles'][()]
            f_fingerprints = f['fingerprints'][()].astype(np.float)
    
    all_counter = 0
    c_counter = 0
    for i in range(len(pred_fp)):
        p_fingerprint = pred_fp[i]
        dist = cos(
            torch.Tensor(p_fingerprint).unsqueeze(0),
            torch.Tensor(f_fingerprints))

        dist = list(dist.numpy())
        scores = list(zip(f_smiles, dist))
        scores = sorted(scores, key=lambda x:x[1])
            
        for j in range(8):
            if correct_smi[i] == scores[j][0]:
                c_counter += 1
            
        all_counter += 1

    print(all_counter, c_counter)

x = test('/home/kkxw544/deepfrag/models/final_model_retrained/predicted_fp.npy', '/home/kkxw544/deepfrag/models/final_model_retrained_03/correct_smi.npy')