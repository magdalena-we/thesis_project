import numpy as np
import pandas as pd
import matplotlib as plt


def eval_sminatest(name_list):
    comp_list = []
    protlig_list =[]
    for i in name_list:
        #try:
        csv_file = '../results/' + i + '.csv'
    
        df = pd.read_csv(csv_file, names=['Affinity', 'Protein', 'Ligand'], header=None)
    
        for j in range(len(df['Protein'])):
            df['Protein'][j] = df['Protein'][j] + df['Ligand'][j]
    
        df = df.dropna(subset=['Affinity'])
        
        tuple_list = list(zip(df['Protein'], df['Affinity']))
        protligs = list(df['Protein'])
        comp_list.append(tuple_list)
        protlig_list.append(protligs)
        #except Exception as e:
        #    print(e)
        #    continue

    protlig_list = list(set(protlig_list[0]).intersection(*protlig_list))
    print(protlig_list)

    affinity_list = []
    for i in comp_list:
        affinities = 0
        counter = 0
        for j in i:
            if j[0] in protlig_list:
                affinities = affinities + j[1]
                counter += 1
        avg_affinity = affinities/len(protlig_list)
        affinity_list.append(avg_affinity)

    return(affinity_list, len(protlig_list)) #len(protlig_list) <=> sample size

def vis_avgaff_sminatest(name_list, title, filename):
    
    models = name_list
    y, sample_size = eval_sminatest(name_list)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots()
    bap = ax.bar(x, y, width, color='#55063a', linewidth=1, edgecolor='white')

    ax.set_ylabel('Affinity')
    ax.set_title(title + 'Smple Size: ' + sample_size)
    ax.set_xticks(x, models)

    ax.bar_label(bap, padding=3, fontsize=7)

    fig.tight_layout()

    plt.savefig('./results/' + filename + '.png')


def vis_affdev_sminatest(standard, name_list, title, filename):
    
    aff_list, sample_size = eval_sminatest(name_list)
    y = []
    for i in aff_list:
        j = i - standard
        y.append(j)

    models = name_list

    x = np.arange(len(models)) 
    width = 0.5

    fig, ax = plt.subplots()
    bap = ax.bar(x, y, width, color='#630c3c', linewidth=1, edgecolor='white')

    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_ylabel('Deviation')
    ax.set_title(title + 'Sample Size: ' + sample_size)
    ax.set_xticks(x, models)

    ax.bar_label(bap, models=[f'{x:.2f}' for x in aff_list], padding=3, fontsize=7)

    fig.tight_layout()

    plt.savefig('./results/' + filename + '.png')


def eval_sminacor(name_list):

    avgs_list = []
    sample_size = []
    for i in name_list:
        #try:
        csv_file = '../results/' + i + '.csv'
    
        df = pd.read_csv(csv_file, names=['1', '2', '3', '4', '5'], header=None)

        df = df.dropna()

        tmp_1 = [] 
        tmp_2 = []
        for column in df:
            i = list(df[column])
            x = np.mean(i)
            tmp_1.append(x)
            tmp_2.append(len(i))

        avgs_list.append(tmp_1)
        sample_size.append(np.mean(tmp_2))

    return(avgs_list, sample_size)


def vis_sminacor(name_list, filename):

    data, sample_size = eval_sminacor(name_list)
    colors = ['#51c9d3', '#afca4f', '#daa719', '#bd217a', '#55063a']
    labels = ['Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5']

    x = np.arange(len(labels))
    width = 0.95

    fig, ax = plt.subplots(3, 2)
    for m in len(data):
        ax.bar(x, data[m], width, color=colors[m], linewidth=1, edgecolor='white')
        ax.set_title(name_list[m] + "Sample Size: " + sample_size[m])
        ax.set_ylabel('Binding Affinity')
        ax.set_xticks(x, labels)
        ax.set_ylim(bottom=200)

    fig.tight_layout()

    plt.savefig('./results/' + filename + '.png')