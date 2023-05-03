'''
script containing functions for creating miscellaneous plots 
'''

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd 
#from sminatest_eval import avg_sminascore


def hist_sminascores(csv_file, binsize, filename):
    
    df = pd.read_csv(csv_file, names = ['Affinity', 'Protein', 'Ligand'])
    aff_list = sorted(list(df['Affinity']))
    
    value = binsize*math.ceil(aff_list[0]/binsize)
    count_list = []
    counter = 0
    i = 0
    while i < len(aff_list):
        if aff_list[i] < value:
            counter += 1
            i+=1
        else:
            count_list.append(counter)
            counter = 0
            value += binsize

    labels = list(range(binsize*math.ceil(aff_list[0]/binsize), value, binsize))
    x = list(range(len(count_list)))
    y = count_list
    fig, ax = plt.subplots()
    ax.bar(x, y) #, width=1, edgecolor='white', linewidth=1
    ax.set_ylabel('Count')
    ax.set_xlabel('Affinity')
    ax.set_xticks(x[::4], labels[::4], rotation=45)
    fig.tight_layout()
    plt.savefig('./results/' + filename + '.png')
    

def accuracy_plot(data_array, filename):
    data = data_array # list containing label list, top1 accuracy list, top8 accuracy list and top64 accuracy list
    x = np.arange(len(data[0]))
    width = 0.25

    fig = plt.figure() #figsize=(16, 8)
    ax = fig.add_subplot()
    top1 = ax.bar(x - width, data[1], width, label='Top 1', color='#daa719', linewidth=1, edgecolor='white')
    top8 = ax.bar(x, data[2], width, label='Top 8', color='#bd217a', linewidth=1, edgecolor='white')
    top64 = ax.bar(x + width, data[3], width, label='Top 64', color='#55063a', linewidth=1, edgecolor='white')

    ax.set_ylabel('Accuracy')
    ax.set_ylim(top=1)
    #ax.set_title('')
    ax.set_xticks(x, labels)
    ax.legend(loc='upper left', frameon=False)

    ax.bar_label(top1, labels=[f'{x:.1%}' for x in top1.datavalues], padding=3, fontsize=10, rotation=90)
    ax.bar_label(top8, labels=[f'{x:.1%}' for x in top8.datavalues], padding=3, fontsize=10, rotation=90)
    ax.bar_label(top64, labels=[f'{x:.1%}' for x in top64.datavalues], padding=3, fontsize=10, rotation=90)

    fig.tight_layout()
    plt.savefig('./results/' + filename + '.png')