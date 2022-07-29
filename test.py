import argparse
import json
import torch
import numpy as np

from leadopt.metrics import cos, top_k_acc
from leadopt.model_conf import LeadoptModel, DIST_FN, MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', help='location to save the model')
    parser.add_argument('--configuration', help='specify the model to be used for testing')
    parser.add_argument('--samples_per_example', type=int)
    parser.add_argument('--output_size', type=int)
    parser.add_argument('--moad_partition', help='choose a file with the datasets')

    subparsers = parser.add_subparsers(dest='version')

    for m in MODELS:
        sub = subparsers.add_parser(m)
        MODELS[m].setup_parser(sub)

    args = parser.parse_args()
    args_dict = args.__dict__

 
    # Initialize model.
    model_type = args_dict['version']
    model = LeadoptModel.load(args.configuration, device=torch.device('cuda'))

    model.run_test(args.save_path, args.samples_per_example)


if __name__=='__main__':
    main()




def test(predicted_fp, correct_fp):

    correct_fp = torch.tensor(np.load(correct_fp))
    
    pred_fp = torch.tensor(np.load(predicted_fp))
    
    accuracy = top_k_acc(correct_fp, cos, [1, 8, 64], pre="")(pred_fp, correct_fp)
    print(accuracy)
    return

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