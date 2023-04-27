'''
simple script to search for pdb-ids inside the originally specified subsets of the MOAD-Dataset.
Entire datasets can be compared and proteins will be classified 
according to the subset they have been found in or if they are unknown.
'''

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

####EXAMPLE-START#############################################################
pdb_id = ['3U81']
dekois2 = ['3TFQ', '3KLM', '3EML', '1UZE', '1R4L', '1EVE', '3EWJ', '3NY9', '3QKL', '1AH3', '1E3G', '3FDN', '2VGO', '2W3L', '3SKC', '3BC3', '1CKP', '3KK6', '1CX2', '3KX1', '1Z11', '1S3V', '1M17', '2VWZ', '3PP0', '3OLL', '1AGW', '2DG3', '1F0R', '2WEG', '1NHZ', '3I4B', '3MAX', '3SFF', '3NU3', '1S6P', '1HW8', '1UY6', '3NW7', '1P44', '3MJ1', '3LXL', '3ELJ', '3NPC', '2B1P', '3K5E', '3MPM', '3LBK', '3KC3', '1HOV', '1A4G', '1OUK', '3L3M', '3FRG', '1XP0', '2XCH', '3DBS', '3R04', '2IWI', '1B8O', '2P54', '1FM9', '2W8Y', '1XJD', '1XOI', '3DDS', '2AFX', '3V8S', '2P1T', '2Z94', '2SRC', '3RM2', '2OO8', '1W4R', '1UOU', '1A5H', '1I00', '3MHW', '3HNG', '3C7Q']
x = search(pdb_id)
y = search(dekois2)
####EXAMPLE-END###############################################################