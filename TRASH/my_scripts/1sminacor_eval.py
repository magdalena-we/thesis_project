import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#['smina_scores_MOAD', 'sminatest_single_fmr03', 'sminatest_single_fmr04', 'sminatest_single_fmr06', 'sminatest_single_smin2', 'sminatest_single_smin3', 'sminatest_single_smin4', 'sminatest_single_smin5', 'sminatest_single_smin6', 'sminatest_single_smin7', 'sminatest_single_smin8', 'sminatest_single_smin9']
name_list = ['sminacor_03', 'sminacor_06', 'sminacor_s2', 'sminacor_s3', 'sminacor_s5', 'sminacor_s9']
avgs_list = []
for i in name_list:
    #try:
    csv_file = '../results/' + i + '.csv'
    
    df = pd.read_csv(csv_file, names=['1', '2', '3', '4', '5'], header=None)

    df = df.dropna()

    temp_list = [] 
    for column in df:
        i = list(df[column])
        x = np.mean(i)
        temp_list.append((x, len(i)))

    avgs_list.append(temp_list)

print(avgs_list)