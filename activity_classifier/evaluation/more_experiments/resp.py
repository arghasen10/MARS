import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 24})
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
models = ['MARS', 'RadHAR', 'Vid2Doppler', 'VGG-16', 'Pantomime']
accuracy = [0.97, 0.85, 0.77, 0.7, 0.81]  
response_time = [1.87, 3.5, 3, 4.2, 4.1]  

fig, ax1 = plt.subplots()

ax1.bar(np.arange(len(models))-0.15, accuracy, width=0.3, color='tab:blue', align='center')
ax1.set_ylabel('Accuracy', color='tab:blue')
ax1.set_xticks(np.arange(len(models)))
ax1.set_xticklabels(models,rotation=20)
ax1.set_ylim(0, 1)

ax2 = ax1.twinx()
ax2.bar(np.arange(len(models))+0.15, response_time, width=0.3, color='tab:orange', align='center')
ax2.set_ylabel('Average Response Time (s)', color='tab:orange')

ax1.yaxis.grid()
plt.tight_layout()
plt.savefig('activity_switching_response_hits2.eps')
plt.show()


accuracy = [0.95, 0.82, 0.71, 0.68, 0.78]  
response_time = [1.8, 4, 3, 4.2, 4.1]  

fig, ax1 = plt.subplots()

ax1.bar(np.arange(len(models))-0.15, accuracy, width=0.3, color='tab:blue', align='center')
ax1.set_ylabel('Accuracy', color='tab:blue')
ax1.set_xticks(np.arange(len(models)))
ax1.set_xticklabels(models,rotation=20)
ax1.set_ylim(0, 1)

ax2 = ax1.twinx()
ax2.bar(np.arange(len(models))+0.15, response_time, width=0.3, color='tab:orange', align='center')
ax2.set_ylabel('Average Response Time (s)', color='tab:orange')

ax1.yaxis.grid()
plt.tight_layout()
plt.savefig('number_of_user_response_hits2.eps')
plt.show()

accuracy = [0.94, 0.8, 0.73, 0.71, 0.79]  
response_time = [1.8, 4, 3, 4.2, 4.1]  

fig, ax1 = plt.subplots()

ax1.bar(np.arange(len(models))-0.15, accuracy, width=0.3, color='tab:blue', align='center')
ax1.set_ylabel('Accuracy', color='tab:blue')
ax1.set_xticks(np.arange(len(models)))
ax1.set_xticklabels(models,rotation=20)
ax1.set_ylim(0, 1)

ax2 = ax1.twinx()
ax2.bar(np.arange(len(models))+0.15, response_time, width=0.3, color='tab:orange', align='center')
ax2.set_ylabel('Average Response Time (s)', color='tab:orange')

ax1.yaxis.grid()
plt.tight_layout()
plt.savefig('activity_switching_number_of_user_response_hits2.eps')
plt.show()
