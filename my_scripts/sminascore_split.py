'''
Script to split protein-ligand-data into sets according to their binding affinity.
'''

import numpy as np
import pandas as pd
import argparse 

import moad_partitions as mp


def build_subset(tuple_list, x):

    '''
    Method to sort protein-ligand-complexes according to their affinity scores and the set they belong to.
    '''

    train_list = []
    val_list = []
    test_list = []
    for n in range(len(tuple_list)):
        if tuple_list[n][1] > x and tuple_list[n][0] in mp.TRAIN:
            train_list.append(tuple_list[n][0])
        if tuple_list[n][1] > x and tuple_list[n][0] in mp.VAL:
            val_list.append(tuple_list[n][0])
        else:
            test_list.append(tuple_list[n][0])

    return train_list, val_list, test_list


def build_new_set(tuple_list, x):

    '''
    Method creates two list of protein ligand complexes; below and above threshold,
    the list containing complexes above threshold value is distributed to VAL and TRAIN set.
    ''' 

    tv_list = []
    test_list = []
    for n in range(len(tuple_list)):
        if tuple_list[n][1] > x:
            tv_list.append(tuple_list[n][0])
        else:
            test_list.append(tuple_list[n][0])

    val_list = tv_list[0:(len(tv_list)):5]
    train_list = list(set(tv_list)-set(val_list))

    return train_list, val_list, test_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file')
    parser.add_argument('--split_fn')
    parser.add_argument('--threshold')
    parser.add_argument('--save_path', help='path to txt-file')

    args = parser.parse_args()
    args_dict = args.__dict__
    
    csv_file = args_dict['csv_file']
    df = pd.read_csv(csv_file)
    df. rename(columns = {'2':'Protein', '3':'Ligand'}, inplace = True)
    df['Affinity'] = df['0'].str.extract(r'(\d+.\d+)').astype('float')

    tuple_list = list(zip(df['Protein'], df['Affinity']))
    tuple_list = list(dict(sorted(tuple_list, key=lambda x: int(x[1]))).items())
    
    threshold = args_dict['threshold']
    fn = args_dict['split_fn']
    save_path = args_dict['save_path']

    if fn == 'subset':
        train_list, val_list, test_list = build_subset(tuple_list, threshold)
    else:
        train_list, val_list, test_list = build_new_set(tuple_list, threshold)

    with open(save_path, 'w') as f:
        f.write(str('Train = ' + str(train_list) + 'VAL = ' + str(val_list) + 'TEST = ' + str(test_list)))

if __name__=='__main__':
    main()