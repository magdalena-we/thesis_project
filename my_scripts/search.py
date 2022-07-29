from config import moad_partitions

def search(pdb_list):
    train = []
    val = []
    test = []
    new = []
    for x in pdb_list:
        if x in moad_partitions.TRAIN:
            train.append(x)
        elif x in moad_partitions.VAL:
            val.append(x)
        elif x in moad_partitions.TEST:
            test.append(x)
        else:
            new.append(x)
    print(test)       
    with open('new_moad.txt', 'w') as f:
        f.write(str(train) + str(val) + str(test) + str(new))


pdb_id = ['3U81']
x = search(pdb_id)