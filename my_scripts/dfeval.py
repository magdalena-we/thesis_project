import urllib.request
import pandas as pd

from my_scripts.dfwf1 import fragment_reconstruction

def apply_workflow(pdb_list, spec_model, top_k):
    data_list = []
    outer_counter = 0
    total_frags = 0
    for i in pdb_list:
        prot = i[0]
        lig = i[1]
        try:
            x = fragment_reconstruction(prot, lig, spec_model, top_k)
            frags = x[0]
            total_frags = total_frags + frags
            inner_counter = x[1]
            outer_counter = outer_counter + inner_counter
            accuracy = inner_counter/frags
            total_accuracy = outer_counter/total_frags
            data_list.append([frags, inner_counter, accuracy, outer_counter, total_frags, total_accuracy])
            print('Sim.Count: ', inner_counter)
            print('TotalFrags: ', total_frags, 'TotalSims: ', outer_counter, 'Accuracy: ', 100*total_accuracy)
        except Exception:
            print('ERROR')
            continue    

    return data_list

def evaluate_data(pdb_list, spec_model, top_k, res_name):
    data_list = apply_workflow(pdb_list, spec_model, top_k)
    df = pd.DataFrame(data_list, columns = ['generated_fragments', 'reconstructed_fragments', 'accuracy', 'total_recfrags', 'total_genfrags', 'total_accuracy'])
    df.to_csv('/home/kkxw544/deepfrag/results/' + res_name + '.csv', sep  = ',')


#pdb_list = [('2XP9', '4G8')]
#x = evaluate_data(pdb_list, 'final_model_retrained_03', 8, 'del_out')