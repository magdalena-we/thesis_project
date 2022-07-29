import urllib.request
import subprocess
import pandas as pd
import numpy as np
import os

from my_scripts.dfwf_smina import fragment_reconstruction
from my_scripts.dfwf_multilig import ligand_reconstruction

path = '/projects/mai/users/kkxw544_magdalena/deepfrag_data/%s'

def prep_output(output):
    output_1 = output.decode('utf-8').split('\\n')
    output_2 = [x for x in output_1 if 'Affinity' in x]
    output_3 = output_2.extract(r'(\d+.\d+)').astype('float')

    return output_3

def apply_workflow(pdb_list, ligfn, spec_model, res_name, x):

    x = int(x)
    pdb_list = pdb_list
    if ligfn == 'multi':
        for i in pdb_list:
            prot = i[0]
            lig = i[1]
            try:
                with open('/home/kkxw544/deepfrag/results/' + res_name + '.csv', 'a') as fd:    
                    bestlig_sdf = ligand_reconstruction(prot, lig, spec_model)
                    print(bestlig_sdf, file=open(path % prot + lig + '_rec.sdf', 'w+'))
                    output = str(subprocess.check_output(["./smina.static", "--score_only", "-r" + path % prot + '.pdb', "-l" + path % prot + lig + '_rec.sdf'], cwd='/home/kkxw544/'))
                    final_output = prep_output(output)
                    append = fin_output.append
                    append(prot)
                    append(lig)
                    writer = csv.writer(fd)
                    writer.writerow(final_output)
                    fd.flush()
                    os.fdatasync(fd)
            except Exception:
                print('ERROR')
                continue
    else:
        for i in pdb_list:
            prot = i[0]
            lig = i[1]
            try:
                with open('/home/kkxw544/deepfrag/results/' + res_name + '.csv', 'a') as fd:    
                    bestlig_list = fragment_reconstruction(prot, lig, spec_model)
                    fin_output = []
                    for bestlig in bestlig_list:
                        bestlig_sdf = tosdf(bestlig)
                        print(bestlig_sdf, file=open(path % prot + lig + '_rec.sdf', 'w+'))
                        output = str(subprocess.check_output(["./smina.static", "--score_only", "-r" + path % prot + '.pdb', "-l" + path % prot + lig + '_rec.sdf'], cwd='/home/kkxw544/'))
                        final_output = prep_output(output)
                        fin_output.append(final_output)
                    mean = np.mean(fin_output)
                    fin_output.clear()   
                    append = fin_output.append
                    append(mean)
                    append(prot)
                    append(lig) 
                    writer = csv.writer(fd)
                    writer.writerow(final_output)
                    fd.flush()
                    os.fdatasync(fd)
            except Exception:
                print('ERROR')
                continue  


#pdb_list = [('2XP9', '4G8')]
#x = apply_workflow(pdb_list, 'multi', 'final_model_retrained_03', 'del_out')
