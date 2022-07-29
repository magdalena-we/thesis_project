import urllib.request
import config.moad_partitions as mp
import pandas as pd


RCSB_DOWNLOAD = 'https://www.rcsb.org/fasta/entry/%s'
path = '/home/kkxw544/deepfrag/temp/%s'


class Tanimoto:

    
    def __init__(self, list1, list2, filename):
        self.list1 = list1
        self.list2 = list2
        self.filename = filename

    def download_pdb(self, pdb_id):
        try:
            urllib.request.urlretrieve(RCSB_DOWNLOAD % pdb_id, path % pdb_id)
        except Exception:
            print(pdb_id, "ERROR")
            return
    
    def prep_data(self, pdb_id):
        file = open(path % pdb_id, "r")
        data = file.read()
        data = data.split('\n')
        data.pop(0)
        while len(data) > 1:    
            data.pop(len(data)-1)
        seq = ''
        for i in range(0, len(data)) :
            seq = seq + data[i]
        if len(seq) == 0:
            print(pdb_id, "ERROR")
            seq += 'x'
        return seq

    
    def compare_seq(self, seq1, seq2):
        counter = 0
        sequence = None
        pattern = None
        if len(seq1) >= len(seq2):
            sequence = seq1
            pattern = seq2
        else: 
            sequence = seq2
            pattern = seq1
        n = len(sequence)
        m = len(pattern)
        match_list = []
    
        for s in range(0, n-m+1) :
            i = 0
            counter = 0
            while (i < m) :
                if pattern[i] == sequence[s + i] :
                    counter = counter + 1
                    i += 1
                else:
                    i += 1
            if i == m :
                s = s + 1
                match_list.append(counter)
            
        result = max(match_list)/m
        return result

    def out(self):
        full_list = []

        for i in self.list1:
            pdb1 = self.download_pdb(i)
            try:
                seq1 = self.prep_data(i)
            except Exception:
                print('ERROR', pdb1)
                
            for j in self.list2:
                pdb2 = self.download_pdb(j)
                try:
                    seq2 = self.prep_data(j)
                    tan_sim = self.compare_seq(seq1, seq2)
                    full_list.append([i, j, tan_sim])
                except Exception:
                    print('ERROR', pdb2)
                    break

        df = pd.DataFrame(full_list, columns = ['pdb1', 'pdb2', 'similarity'])
        df.to_csv('/home/kkxw544/deepfrag/' + self.filename + '.csv', sep  = ',')


dekois2_new = ['3EML', '1R4L', '3NY9', '3KK6', '1CX2', '3SKC', '1M17', '1S6P', '1P44', '3K5E', '3LBK', '3KC3', '1HOV', '3DBS', '2IWI', '3RM2', '1AH5', '3MHW', '3HNG']

dek2_train = Tanimoto(mp.TRAIN, dekois2_new, 'dek2_train')
dek2_train.out()

dek2_val = Tanimoto(mp.VAL, dekois2_new, 'dek2_val')
dek2_val.out()

dek2_test = Tanimoto(mp.TEST, dekois2_new, 'dek2_test')
dek2_test.out()