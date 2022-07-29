import subprocess
import urllib.request
import pandas as pd
from config import moad_partitions as md

RCSB_DOWNLOAD = 'https://files.rcsb.org/download/%s.pdb'
LIG_DOWNLOAD = 'https://files.rcsb.org/ligands/download/%s_model.sdf'
path = '/home/kkxw544/deepfrag/temp/%s'

def prep_data(pdb_id): 

    try:
        urllib.request.urlretrieve(RCSB_DOWNLOAD % pdb_id, path % pdb_id + '.pdb')
    
        file = open(path % pdb_id + '.pdb', 'r')
        data = file.readlines()
        lig_list = []
        for line in data:
            if line.startswith('HET '):
                lig_list.append(line.split()[1])
        lig_list = list(set(lig_list))

    except Exception:
        print(pdb_id, 'ERROR')
        return
    
    return lig_list

def apply_workflow():
    fin_out_list = []
    moad = [md.TEST, md.VAL, md.TRAIN]
    for m in moad:
        for i in m:
            lig_list = prep_data(i)
            try: 
                for j in lig_list:
                    if len(j) == 3 and j != 'HOH':
                        try:
                            urllib.request.urlretrieve(LIG_DOWNLOAD % j, path % i + j + '.sdf')
                            output = str(subprocess.check_output(["./smina.static", "--score_only", "-r" + path % i + '.pdb', "-l" + path % i + j + '.sdf'], cwd='/home/kkxw544/smina-code/'))
                            final_output = list(prep_output(output))
                            append = final_output.append
                            append(i)
                            append(j)
                            print(final_output)
                            fin_out_list.append(final_output)
                        except Exception:
                            print(j, 'ERROR')
                            continue
            except Exception:
                print('ERROR')
                continue
    df = pd.DataFrame(fin_out_list)
    df.to_csv('/home/kkxw544/deepfrag/smina_results.csv', sep = ',')


def prep_output(output):
    output_1 = output.decode('utf-8').split('\\n')
    output_2 = [x for x in output_1 if 'Affinity' in x]
    output_3 = output_2.extract(r'(\d+.\d+)').astype('float')

    return output_3

x = apply_workflow()