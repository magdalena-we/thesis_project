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




'''labels = ['Paper', 'RM_01', 'RM_02', 'RM_03', 'RM_04', 'RM_05', 'RM_06', 'RM_07']

#method_1:
y1 = [0.5777, 0.3816, 0.3891, 0.4244, 0.4365, 0.4251, 0.4628, 0.0]
y8 = [0.6616, 0.4755, 0.4612, 0.5068, 0.5187, 0.5039, 0.5357, 0.0]
y64 = [0.7226, 0.5430, 0.5287, 0.5576, 0.5694, 0.5533, 0.5807, 0.0]

#method_2:
#y1 = [0.5777, 0.5201, 0.5245, 0.5893, 0.5896, 0.5958, 0.6045, 0.613]
#y8 = [0.6616, 0.5201, 0.5245, 0.5898, 0.5901, 0.5961, 0.6052, 0.614]
#y64 = [0.7226, 0.5201, 0.5246, 0.5923, 0.5929, 0.5993, 0.6126, 0.621]

x = np.arange(len(labels))  # the label locations
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, y1, width, label='Top 1', color='#daa719', linewidth=1, edgecolor='white')
rects2 = ax.bar(x, y8, width, label='Top 8', color='#bd217a', linewidth=1, edgecolor='white')
rects3 = ax.bar(x + width, y64, width, label='Top 64', color='#55063a', linewidth=1, edgecolor='white')



#Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_ylim(top=1)
#ax.set_title('')
ax.set_xticks(x, labels)
ax.legend(fontsize='x-small', frameon=False)

ax.bar_label(rects1, labels=[f'{x:.1%}' for x in rects1.datavalues], padding=3, fontsize=7, rotation=90)
ax.bar_label(rects2, labels=[f'{x:.1%}' for x in rects2.datavalues], padding=3, fontsize=7, rotation=90)
ax.bar_label(rects3, labels=[f'{x:.1%}' for x in rects3.datavalues], padding=3, fontsize=7, rotation=90)

fig.tight_layout()

plt.savefig('/home/kkxw544/deepfrag/results/model_accuracy_m01.png')'''




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




v_list = [322.21171556381165, 311.77631658947405, 343.7038718156256, 312.19003513173107, 357.73340505052164, 408.7145684089046, 330.2032303512831, 346.5765367761026, 365.73440291287017, 345.0073248401638]
v_mean = 298.6385761467889
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