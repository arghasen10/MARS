import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


# create data
fig = plt.figure(figsize=(11,6))
ax = plt.subplot(111)
x = ['1 User\nW-1', '1 User\nMa-1', '1 User\nMi-1', '2 User\nMa-1\nMi-1', '2 User\nW-2',
     '2 User\nW-1\nMa-1', '2 User\nMi-2', '3 User\nMa-1\nMi-2', '3 User\nMa-2\nMi-1',
     '3 User\nMa-3', '3 User\nMi-3', '3 User\nW-1\nMa-2', '3 User\nW-1\nMi-2']
y1 = np.array([4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6])           #Clustering+Denoising
y2 = np.array([1, 1, 1, 1.2, 1.2, 1.2, 1.2, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3])           #RF Classifier
y3 = np.array([2, 0, 0, 0, 3.5, 2, 0, 0, 0, 0, 0, 4, 3])            #Servo Tracking
y4 = np.array([0, 1.3, 1.3, 1.3, 0, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3])            #Configuration Switch 
y5 = np.array([0, 2, 0, 2, 0, 2, 0, 0, 2, 2, 0, 2, 0])             #Macro
y6 = np.array([0, 0, 0, 1.3, 0, 0, 0, 0, 1.3, 0, 0, 0, 0])             #Configuration Switch 
y7 = np.array([0, 0, 2, 2, 0, 0, 2, 2, 2, 0, 2, 0, 2])             #Micro 
y8 = np.array([0, 0, 0, 0, 0, 0,0, 1.3, 0,0, 0, 0,0])             #Configuration Switch 
y9 = np.array([0, 0 ,0, 0, 0, 0 ,0, 2, 0, 0 ,0, 0, 0])             #Micro 
# plot bars in stack manner
ax.bar(x, y1, color='r', hatch='//', label='Clustering+Denoising')
ax.bar(x, y2, bottom=y1, color='b', hatch='*', label='RF Classifier')
ax.bar(x, y3, bottom=y1+y2, color='g', hatch='.', label='Servo Tracker')
ax.bar(x, y4, bottom=y1+y2+y3, color='y', hatch='.', label='Configuration Switch')
ax.bar(x, y5, bottom=y1+y2+y3+y4, color='tab:orange', hatch='.', label='Macro Activity Classifier')
ax.bar(x, y6, bottom=y1+y2+y3+y4+y5, color='y', hatch='.')
ax.bar(x, y7, bottom=y1+y2+y3+y4+y5+y6, color='tab:brown', hatch='o', label='Micro Activity Classifier')
ax.bar(x, y8, bottom=y1+y2+y3+y4+y5+y6+y7, color='y', hatch='.')
ax.bar(x, y9, bottom=y1+y2+y3+y4+y5+y6+y7+y8, color='tab:orange', hatch='.')
ax.set_xlabel('Scenario')
ax.set_ylabel('Response Time (sec)')
ax.legend(ncol=3,  bbox_to_anchor=(1, 1.15),)
plt.grid()
plt.tight_layout()
plt.show()
