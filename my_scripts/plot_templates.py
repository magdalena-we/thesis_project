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




'''labels = ['Paper', 'Retrain1', 'Retrain2', 'Retrain3', 'Retrain4', 'Retrain5']

y1 = [0.5777, 0.3816, 0.4244, 0.4347, 0.4225, 0.4623]
y8 = [0.6616, 0.4755, 0.5068, 0.5130, 0.5039, 0.5357]
y64 = [0.7226, 0.5430, 0.5576, 0.5596, 0.5533, 0.5807]

x = np.arange(len(labels))  # the label locations
width = 0.2

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, y1, width, label='Top 1', color='#daa719', linewidth=1, edgecolor='white')
rects2 = ax.bar(x, y8, width, label='Top 8', color='#bd217a', linewidth=1, edgecolor='white')
rects3 = ax.bar(x + width, y64, width, label='Top 64', color='#55063a', linewidth=1, edgecolor='white')



#Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
#ax.set_title('')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, labels=[f'{x:.1%}' for x in rects1.datavalues], padding=3, fontsize=7)
ax.bar_label(rects2, labels=[f'{x:.1%}' for x in rects2.datavalues], padding=3, fontsize=7)
ax.bar_label(rects3, labels=[f'{x:.1%}' for x in rects3.datavalues], padding=3, fontsize=7)

fig.tight_layout()

plt.savefig('/home/kkxw544/deepfrag/results/model_accuracy_01.png')




labels = ['Retrain2', 'Subset_0', 'Subset_200', 'Set_200', 'Set_400']

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




v_list = [302.98755552750646, 303.39191861257893, 302.91758345038147, 303.77412492667526, 305.2037124333636]
v_mean = 302.98755552750646
y_list = []
for x in v_list:
    y = x - v_mean
    y_list.append(y)

print(y_list)


labels = ['Retrain2', 'Subset_0', 'Subset_200', 'Set_200', 'Set_400']

y = y_list

x = np.arange(len(labels))  # the label locations
width = 0.2

fig, ax = plt.subplots()
rects1 = ax.bar(x, y, width, color='#55063a', linewidth=1, edgecolor='white')


ax.axhline(0, color='grey', linewidth=0.8)
#Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Deviation from Model with original Datasets')
#ax.set_title('')
ax.set_xticks(x, labels)

ax.bar_label(rects1, labels=[f'{x:.2}' for x in rects1.datavalues], padding=3, fontsize=7)

fig.tight_layout()

plt.savefig('/home/kkxw544/deepfrag/results/smina_eval_deviation_01.png')