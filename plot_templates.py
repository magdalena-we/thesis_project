'''
script containing code snippets for creating plots for the test methods, sminatest and sminacor
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


'''count = [29582, 18348, 9213, 4506, 2943, 2189, 1823, 1683, 1535, 1674, 1729, 1826, 1633, 1327, 1001, 661, 426, 276, 176, 130, 79, 55, 52, 39, 39, 29, 21, 14, 9, 7, 4, 1, 1, 3, 1, 1, 0, 1, 1, 0, 2, 1, 0, 0, 0, 0, 0, 1]

xlist = [x for x in range(0, len(count)*50, 50)]

x = range(len(count)-1)
y = count[1:]

fig, ax = plt.subplots()

rects1 = ax.bar(x, y, width=1, edgecolor='white', linewidth=1)

#labels = ['Paper', 'Orig. Model', 'Retrain1', 'Retrain2']
x = np.arange(len(xlist))  # the label locations

width = 0.35  # the width of the bars


#Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Count')
ax.set_xlabel('Affinity')
ax.set_xticks(x[::4], xlist[::4], rotation=45)
#ax.set_xticks(np.arange(0, len(count)*50+50, 200)) --- doesn't work with bar plot

#ax.bar_label(rects1, padding=3)


fig.tight_layout()

plt.savefig('plot_sminaout4.png')'''


'''
labels = ['Paper', 'RM_01', 'RM_02', 'RM_03', 'RM_04', 'RM_05', 'RM_06', 'RM_07'] #, 'SM_01', 'SM_02', 'SM_03', 'SM_04', 'SM_05', 'SM_06', 'SM_07', 'SM_08'

#method_1:
#y1 = [0.5777, 0.3816, 0.3891, 0.4244, 0.4365, 0.4251, 0.4628, 0.0]
#y8 = [0.6616, 0.4755, 0.4612, 0.5068, 0.5187, 0.5039, 0.5357, 0.0]
#y64 = [0.7226, 0.5430, 0.5287, 0.5576, 0.5694, 0.5533, 0.5807, 0.0]

#method_2:
y1 = [0.5777, 0.5201, 0.5245, 0.5892, 0.5895, 0.5955, 0.6024, 0.6042] #, 0.5811, 0.5713, 0.6333, 0.6022, 0.6119, 0.6079, 0.6434, 0.6717
y8 = [0.6616, 0.6044, 0.6064, 0.6712, 0.6750, 0.6793, 0.6877, 0.6855] #, 0.6631, 0.6503, 0.7073, 0.6746, 0.6880, 0.6843, 0.7097, 0.7416
y64 = [0.7226, 0.6857, 0.6835, 0.7339, 0.7364, 0.7425, 0.7481, 0.7465] #, 0.7259, 0.7149, 0.7568, 0.7286, 0.7501, 0.7491, 0.7664, 0.7993

x = np.arange(len(labels))  # the label locations
width = 0.25

fig = plt.figure() #figsize=(16, 8)
ax = fig.add_subplot()
rects1 = ax.bar(x - width, y1, width, label='Top 1', color='#daa719', linewidth=1, edgecolor='white')
rects2 = ax.bar(x, y8, width, label='Top 8', color='#bd217a', linewidth=1, edgecolor='white')
rects3 = ax.bar(x + width, y64, width, label='Top 64', color='#55063a', linewidth=1, edgecolor='white')



#Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_ylim(top=1)
#ax.set_title('')
ax.set_xticks(x, labels)
ax.legend(loc='upper left', frameon=False)

ax.bar_label(rects1, labels=[f'{x:.1%}' for x in rects1.datavalues], padding=3, fontsize=10, rotation=90)
ax.bar_label(rects2, labels=[f'{x:.1%}' for x in rects2.datavalues], padding=3, fontsize=10, rotation=90)
ax.bar_label(rects3, labels=[f'{x:.1%}' for x in rects3.datavalues], padding=3, fontsize=10, rotation=90)

fig.tight_layout()

plt.savefig('/home/kkxw544/deepfrag/results/model_accuracy_m02_1.png')'''




'''x = [1, 8, 64]

y0 = [57.77, 66.16, 72.26]
y1 = [52.01, 52.01, 52.01]
y2 = [52.45, 52.45, 52.46]
y3 = [58.93, 58.98, 59.23]
y4 = [58.96, 59.01, 59.29]
y5 = [59.58, 59.61, 59.93]
y6 = [60.45, 60.52, 61.26]
#y7 = [0.0, 0.0, 0.0]

fig, ax = plt.subplots()

ax.plot(x, y0, marker='.', label='Paper')
ax.plot(x, y1, marker='.', label='RM_01')
ax.plot(x, y2, marker='.', label='RM_02')
ax.plot(x, y3, marker='.', label='RM_03')
ax.plot(x, y4, marker='.', label='RM_04')
ax.plot(x, y5, marker='.', label='RM_05')
ax.plot(x, y6, marker='.', label='RM_06')
#ax.plot(x, y7, 'o--', label='RM_07')

ax.set_ylabel('Accuracy in Percent')
ax.set_ylim(top=75)
ax.set_xlabel('top-k Fragments')
ax.legend(loc='upper left', fontsize='x-small', frameon=False)

plt.savefig('/home/kkxw544/deepfrag/results/model_accuracy_distribution.png')'''


'''x1 = [1, 6, 64, 640, 6400, 64000]
x2 = [1, 8, 64]

y0 = [57.77, 66.16, 72.26]
y1 = [52.01, 52.01, 52.01, 52.71, 61.93, 98.29]
y6 = [60.45, 60.52, 61.26, 64.26, 76.96, 99.16]

fig, ax = plt.subplots()

ax.plot(x2, y0, marker='.', label='Paper')
ax.plot(x1, y1, marker='.', label='RM_01')
ax.plot(x1, y6, marker='.', label='RM_06')

ax.set_ylabel('Accuracy in Percent')
ax.set_xlabel('top-k Fragments')
ax.legend(loc='upper left', fontsize='x-small', frameon=False)

plt.savefig('/home/kkxw544/deepfrag/results/model_accuracy_distribution_full.png')'''


'''labels = ['Retrain2', 'Subset_0', 'Subset_200', 'Set_200', 'Set_400']

y = [303.0, 303.4, 302.9, 303.8, 305.2]

x = np.arange(len(labels))  # the label locations
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x, y, width, color='#55063a', linewidth=1, edgecolor='white')



#Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Affinity')
#ax.set_title('')
ax.set_xticks(x, labels)

ax.bar_label(rects1, padding=3, fontsize=7)

fig.tight_layout()

plt.savefig('/home/kkxw544/deepfrag/results/smina_eval_affinity_01.png')'''




v_list = [325.9867850999012, 323.16156011434794, 332.9684385598848, 322.3425440437918, 338.1145208195566, 356.79870902662867, 329.1685860496484, 335.2433684513003, 360.18818821802597, 352.8440711667289]
v_mean = 329.87640877300595
y_list = []
for x in v_list:
    y = x - v_mean
    y_list.append(y)

print(y_list)


labels = ['RM_03', 'RM_06', 'SM_01', 'SM_02', 'SM_03', 'SM_04', 'SM_05', 'SM_06', 'SM_07', 'SM_08']

y = y_list

x = np.arange(len(labels))  # the label locations
width = 0.5

fig, ax = plt.subplots()
rects1 = ax.bar(x, y, width, color='#630c3c', linewidth=1, edgecolor='white')


ax.axhline(0, color='grey', linewidth=0.8)
#Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Deviation')
#ax.set_title('')
ax.set_xticks(x, labels)

ax.bar_label(rects1, labels=[f'{x:.2f}' for x in y_list], padding=3, fontsize=7)

fig.tight_layout()

plt.savefig('/home/kkxw544/deepfrag/results/smina_eval_deviation_03.png')

'''
labels = ['Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5']

m1 = [310.61, 306.81, 305.54, 302.30, 302.18]
m2 = [300.51, 299.42, 296.45, 296.60, 291.97]
m3 = [297.99, 295.91, 293.94, 293.40, 293.10]
m4 = [287.01, 285.57, 284.15, 284.18, 281.84]
m5 = [298.68, 300.06, 300.53, 298.08, 295.05]

x = np.arange(len(labels))  # the label locations
width = 0.95

fig, ax = plt.subplots()
#rects1 = ax.bar(x, m1, width, label='RM_03', color='#51c9d3', linewidth=0.1, edgecolor='white')
#rects2 = ax.bar(x, m2, width, label='RM_06', color='#afca4f', linewidth=0.1, edgecolor='white')
#rects3 = ax.bar(x, m3, width, label='SM_02', color='#daa719', linewidth=1, edgecolor='white')
#rects4 = ax.bar(x, m4, width, label='SM_04', color='#bd217a', linewidth=1, edgecolor='white')
rects5 = ax.bar(x, m5, width, label='SM_08', color='#55063a', linewidth=1, edgecolor='white')



#Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title('SM_08')
ax.set_ylabel('Binding Affinity')
ax.set_xticks(x, labels)
ax.set_ylim(bottom=200)

fig.tight_layout()

plt.savefig('/home/kkxw544/deepfrag/results/smina_corr_sm08.png')'''