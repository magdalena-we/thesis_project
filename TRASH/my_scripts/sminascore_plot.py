import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('../results/smina_results.csv')
df. rename(columns = {'2':'Protein', '3':'Ligand'}, inplace = True)
df['Affinity'] = df['0'].str.extract(r'(\d+.\d+)').astype('float')


aff_list1 = list(df['Affinity'])
aff_list2 = list(set(aff_list1))
aff_list2.sort()

total = len(aff_list1)
test = 0.2*total


'''count=[]
i = 0
m = 0
x = len(aff_list1)
while x > 0:    
    for n in aff_list1:
        if n <= m:
            i += 1
            aff_list1.remove(n)
    count.append(i)
    m += 50
    i = 0
    x = len(aff_list1)

print(count)'''



'''affinity=[]
count=[]
for m in aff_list2:
    i = 0
    for n in aff_list1:
        if m == n:
            i += 1
    affinity.append(m)
    count.append(i)

q = 0
cutoff = -1
for p in count:
    while q + p <= test:
        q = q + p
        cutoff += 1

    
cutoff_value = affinity[int(cutoff)]

print(count[0], test, cutoff, cutoff_value)'''


tuple_list = list(zip(df['Protein'], df['Affinity']))
list1 = []
list2 = []

tuple_list = list(dict(sorted(tuple_list, key=lambda x: int(x[1]))).items())

tv_list = []
test_list = []
for n in range(len(tuple_list)):
    if tuple_list[n][1] > 400:
        tv_list.append(tuple_list[n][0])
    else:
        test_list.append(tuple_list[n][0])
    
print(len(tv_list), len(test_list), tuple_list[:10], tuple_list[-10:])

val_list = tv_list[0:(len(tv_list)):5]
train_list = list(set(tv_list)-set(val_list))

print(len(train_list), len(val_list))

'''for b in affinity:
    for a in range(len(tuple_list)):
        if tuple_list[a][0] == b:
            if b == 0:
                list1.append((tuple_list[a][1], tuple_list[a][2]))
            else:
                list2.append((tuple_list[a][1], tuple_list[a][2]))

print(len(list1), len(list2)) 

ligands = []
a = 0
lenlist = len(list1)
while a < lenlist:
    for b in range(len(list2)):
        if list1[a][0] == list2[b][0]:
            x = list1.pop(a)
            ligands.append(x)
            lenlist = lenlist - 1
            break
    a += 1
                    

print(len(list1), len(list2))

only_p1 = []
for a in list1:
    only_p1.append(a[0])

zero_list = list(set(only_p1))

only_p2 = []
for a in list2:
    only_p2.append(a[0])

print(len(zero_list))'''


'''x = affinity[1:]
y = count[1:]
plt.plot(x, y, 'b+-')
plt.xlabel('Affinity')
plt.ylabel('Count')
#plt.axvspan(cutoff_value-1, cutoff_value+1, color='red', alpha=0.5)
plt.savefig('plot5.png')'''




#test_list = []

'''for a in tuple_list:
    if a[0] >= cutoff_value:
        trainval_list.append(a[1])
    elif a[0] < cutoff_value:
        test_list.append(a[1])
    else:
        print('Error')
test
val_list = []
i=0
while i < len(trainval_list):
    b = trainval_list.pop(i)
    val_list.append(b)
    i = i + 3

print(len(trainval_list), len(test_list), len(val_list))'''

with open('/home/kkxw544/deepfrag/results/moad_400.txt', 'w') as f:
    f.write(str('Train = ' + str(train_list) + 'VAL = ' + str(val_list) + 'TEST = ' + str(test_list)))