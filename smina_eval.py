import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def eval_sminatest(path, name_list):

    '''This function processes the results from the sminatest,
    it retrieves comparable sets of protein/ligand complexes for different 
    models and computes the the average affinity per model from these 
    datasets.'''

    comp_list = []
    protlig_list =[]
    for i in name_list:
        #try:
        csv_file = path + i + '.csv'
    
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

    return (affinity_list, len(protlig_list)) #len(protlig_list) <=> sample size

####EXAMPLE###################################################################
#x,y = eval_sminatest('./thesis/mydata/results/sminatest/', ['omp_ep50', 'omp_ep500', '200_ep50', '200_ep500', '400_ep50', '400_ep500'])
##############################################################################

def vis_avgaff_sminatest(path, name_list, title, filename):

    '''This function takes the output of the sminatest_eval() function as 
    input and visualizes the results as a barplot'''
    
    models = name_list
    y, sample_size = eval_sminatest(path, name_list)
    
    x = list(np.arange(len(models)))
    width = 0.35

    fig, ax = plt.subplots()
    bap = ax.bar(x, y, width, color='#55063a', linewidth=1, edgecolor='white')

    ax.set_ylabel('Affinity')
    #ax.set_title(title + '; Sample Size: ' + str(sample_size))
    ax.set_xticks(x, models)

    ax.bar_label(bap, padding=3, fontsize=7)

    fig.tight_layout()

    plt.savefig('./results/' + filename + '.png')

####EXAMPLE###################################################################
#x = vis_avgaff_sminatest('./thesis/mydata/results/sminatest/', ['omp_ep50', 'omp_ep500', '200_ep50', '200_ep500', '400_ep50', '400_ep500'], 'Average Binding Affinity of Ligands', 'avgaff_plot')
##############################################################################


def vis_affdev_sminatest(standard, path, name_list, title, filename):

    '''This function takes the output of the sminatest_eval() function as 
    input and visualizes the different models' deviations from the 
    original model in a plot'''
    
    aff_list, sample_size = eval_sminatest(path, name_list)
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
    #ax.set_title(title + '; Sample Size: ' + str(sample_size))
    ax.set_xticks(x, models)

    ax.bar_label(bap, labels=[f'{x:.2f}' for x in y], padding=3, fontsize=7)

    fig.tight_layout()

    plt.savefig('./results/' + filename + '.png')

####EXAMPLE###################################################################
#x = vis_affdev_sminatest(300.272, './thesis/mydata/results/sminatest/', ['omp_ep50', 'omp_ep500', '200_ep50', '200_ep500', '400_ep50', '400_ep500'], 'Deviation of Average Binding Affinity', 'avdev_plot')
##############################################################################


def eval_sminacor(name_list):

    '''This function processes the results from the sminacor,
    it computes the mean affinity for different top-k positions from different
    models.'''

    avgs_list = []
    sample_size = []
    for i in name_list:
        #try:
        csv_file = './results/' + i + '.csv'
    
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

    return (avgs_list, sample_size)


def vis_sminacor(name_list, filename):

    ''' This function uses the output of the eval_sminacor function as input 
    visualizes the results as bar plots'''

    data, sample_size = eval_sminacor(name_list)
    colors = ['#51c9d3', '#afca4f', '#daa719', '#bd217a', '#55063a']
    labels = ['Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5']

    x = np.arange(len(labels))
    width = 0.95

    fig, ax = plt.subplots(2, len(name_list)//2, figsize=(10, 8))
    ax = ax.flatten()
    for m in range(len(data)):
        ax[m].bar(x, data[m], width, color=colors[m], linewidth=1, edgecolor='white')
        ax[m].set_title(name_list[m]) # + "; Sample Size: " + str(sample_size[m]))
        ax[m].set_ylabel('Binding Affinity')
        ax[m].set_xticks(x, labels)
        ax[m].set_ylim(bottom=200)

    fig.tight_layout()

    plt.savefig('./results/' + filename + '.png')

####EXAMPLE###################################################################
x = vis_sminacor(['sminacor_omplr0001ep50', 'sminacor_omplr0001ep500', 'sminacor_400lr0001ep50', 'sminacor_400lr0001ep500'], 'sminacor')
##############################################################################