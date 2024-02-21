import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


activities = ['jumping', 'clapping', 'lunges', 'squats', 'waving', 'vacuum\ncleaning', 'folding\nclothes', 'changing\nclothes']
groups = ['Front', 'Left', 'Back', 'Right']
accuracy = np.array([[0.99, 0.99, 1, 0.99], [1, 0.99, 0.98, 0.96], [0.99, 0.99, 0.99, 0.98], [0.99, 1,1,0.98], [1, 0.99, 1, 1], [1,1,0.99,1], [1, 0.99, 0.98,1], [1, 1, 0.98,1]]) 
fig, ax = plt.subplots()
bar_width = 0.2
opacity = 0.8
index = np.arange(len(activities))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:grey']

for i in range(len(groups)):
    plt.bar(index + i*bar_width, accuracy[:, i], bar_width, alpha=opacity, label=groups[i],ec='k',color=colors[i])

plt.ylabel('Accuracy')
plt.xticks(index + bar_width*1.5, activities, rotation=45)
plt.legend()
plt.ylim([0.85,1.03])
plt.yticks([0.85, 0.9, 0.95, 1])  
plt.legend(ncol=4)
plt.tight_layout()
ax.yaxis.grid()
plt.savefig('diff_orient_macro_2.eps')
plt.show()



activities = ['laptop\ntyping', 'phone\ntalking', 'phone\ntyping', 'sitting', 'playing\nguitar', 'eating\nfood', 'combing\nhair', 'brushing\nteeth', 'drinking\nwater']
groups = ['Front', 'Left', 'Back', 'Right']
accuracy = np.array([[0.87, 0.93, 0.91, 0.87], [0.91, 0.87, 0.93, 0.87], [0.87, 0.85, 0.91, 0.87], [0.6, 0.8,0.2,0.18], [1, 0.87, 0.85, 1], [0.88,0.88,0.88,0.84], [0.85, 0.84, 0.81,0.85], [1, 0.91, 0.87,0.89], [0.91, 0.87, 0.85,0.89]]) 
fig, ax = plt.subplots()
bar_width = 0.2
opacity = 0.8
index = np.arange(len(activities))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:grey']

for i in range(len(groups)):
    plt.bar(index + i*bar_width, accuracy[:, i], bar_width, alpha=opacity, label=groups[i],ec='k',color=colors[i])

plt.ylabel('Accuracy')
plt.xticks(index + bar_width*1.5, activities, rotation=45)
plt.legend()
plt.ylim([0.,1.15])
plt.yticks([0., 0.25, 0.5, 0.75, 1])  
plt.legend(ncol=4)
plt.tight_layout()
ax.yaxis.grid()
plt.savefig('diff_orient_micro_2.eps')
plt.show()


activities = ['jumping', 'clapping', 'lunges', 'squats', 'waving', 'vacuum\ncleaning', 'folding\nclothes', 'changing\nclothes']
groups = ['2m', '3m', '5m']
accuracy = np.array([[0.98, 1, 1], [0.96, 0.97, 1], [0.96, 1, 1], [0.95, 1,0.99], [0.99, 1, 1], [0.99, 0.99, 1], [0.985, 0.99, 1], [0.975, 0.99, 1]]) 
fig, ax = plt.subplots()
bar_width = 0.2
opacity = 0.8
index = np.arange(len(activities))
colors = ['tab:blue', 'tab:orange', 'tab:green']

for i in range(len(groups)):
    plt.bar(index + i*bar_width, accuracy[:, i], bar_width, alpha=opacity, label=groups[i],ec='k',color=colors[i])

plt.ylabel('Accuracy')
plt.xticks(index + bar_width*1.5, activities, rotation=45)
plt.legend()
plt.ylim([0.85,1.03])
plt.yticks([0.85, 0.9, 0.95, 1])  
plt.legend(ncol=4)
plt.tight_layout()
ax.yaxis.grid()
plt.savefig('diff_dist_macro_2.eps')
plt.show()

activities = ['laptop\ntyping', 'phone\ntalking', 'phone\ntyping', 'sitting', 'playing\nguitar', 'eating\nfood', 'combing\nhair', 'brushing\nteeth', 'drinking\nwater']
groups = ['2m', '3m', '5m']
accuracy = np.array([[1, 0.99, 0.98], [0.99, 0.98, 0.96], [0.99, 0.97, 0.96], [0.985, 0.65, 0.45], [0.99, 0.98, 0.96], [0.99, 0.98, 0.96], [0.985, 0.97, 0.96], [1, 0.99, 0.98], [0.99, 0.99, 1]]) 
fig, ax = plt.subplots()
bar_width = 0.2
opacity = 0.8
index = np.arange(len(activities))
colors = ['tab:blue', 'tab:orange', 'tab:green']

for i in range(len(groups)):
    plt.bar(index + i*bar_width, accuracy[:, i], bar_width, alpha=opacity, label=groups[i],ec='k',color=colors[i])

plt.ylabel('Accuracy')
plt.xticks(index + bar_width*1.5, activities, rotation=45)
plt.legend()
plt.ylim([0,1.12])
plt.yticks([0, 0.25, 0.5, 0.75, 1])  
plt.legend(ncol=4)
plt.tight_layout()
ax.yaxis.grid()
plt.savefig('diff_dist_micro_2.eps')
plt.show()