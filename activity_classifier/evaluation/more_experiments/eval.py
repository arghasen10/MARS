import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 28})
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# Sample data (replace with actual data)
data = {
    'macro': [0.99, 0.97, 0.95, 0.91, 0.89],
    'micro': [0.96, 0.94, 0.91, 0.87, 0.83]
}

subjects = [1, 2, 3, 4, 5]
bar_width = 0.35
index = np.arange(len(subjects))

fig, ax = plt.subplots()

for i, (label, values) in enumerate(data.items()):
    ax.bar(index + i * bar_width, values, bar_width, label=label)

ax.set_xlabel('Number of Subjects')
ax.set_ylabel('Accuracy')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(subjects)
ax.legend(ncol=2)
ax.yaxis.grid()
plt.tight_layout()
plt.savefig('grouped_bar_plot.png')
plt.show()

data = {
    'R1': {
        'Inter-room (M)':0.885,
        'Intra-room (M)': 0.92,
        'Intra-room ($\mu$)': 0.87,
        'Inter-room ($\mu$)': 0.84
    },
    'R2': {
        'Intra-room (M)': 0.98,
        'Intra-room ($\mu$)': 0.94
    },
    'R3': {
        'Inter-room (M)': 0.96,
        'Intra-room (M)': 0.95,
        'Intra-room ($\mu$)': 0.91,
        'Inter-room ($\mu$)': 0.89
    }
}

groups = ['R1', 'R2', 'R3']
bar_width = 0.2
index = np.arange(len(groups))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:gray']
fig, ax = plt.subplots()
visited_labels = []
count = 0
ext_count = 0
for i, (group, values) in enumerate(data.items()):
    for j, (label, heights) in enumerate(values.items()):
        print(index[i] + j * bar_width, label)
        if label in visited_labels:
            ax.bar((count*bar_width)+ext_count*bar_width, heights, bar_width,color=colors[visited_labels.index(label)])
        else:
            visited_labels.append(label)
            ax.bar((count*bar_width)+ext_count*bar_width, heights, bar_width, label=label, color=colors[visited_labels.index(label)])
        count+=1
    ext_count+=1

ax.set_ylabel('Accuracy')
ax.set_xticks([0.3,1.1,1.9])
ax.set_xticklabels(groups)
ax.legend(ncol=2, bbox_to_anchor=(1.1, 1.25),fontsize=22)
ax.yaxis.grid()
plt.tight_layout()
plt.savefig('grouped_bar_plot_3_groups.png')
plt.show()


data = {
    'Wooden board': {
        'macro': 0.9,
        'micro': 0.81
    },
    'Fibre board': {
        'macro': 0.92,
        'micro': 0.84
    },
    'Glass board': {
        'macro': 0.95,
        'micro': 0.83
    }
}

groups = ['Wooden\nboard', 'Fibre\nboard', 'Glass\nboard']
bar_width = 0.35
index = np.arange(len(groups))

fig, ax = plt.subplots()
visited_labels = []
for i, (group, values) in enumerate(data.items()):
    for j, (label, heights) in enumerate(values.items()):
        if label is 'macro':
            if label in visited_labels:
                ax.bar(index[i] + j * bar_width, heights, bar_width,color = 'tab:blue')
            else:
                ax.bar(index[i] + j * bar_width, heights, bar_width,color = 'tab:blue', label='macro')
                visited_labels.append(label)
        else:
            if label in visited_labels:
                            ax.bar(index[i] + j * bar_width, heights, bar_width, color='tab:orange')
            else:
                ax.bar(index[i] + j * bar_width, heights, bar_width, color='tab:orange', label='micro')
                visited_labels.append(label)

ax.set_ylabel('Accuracy')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(groups)
ax.legend(ncol=2)
ax.yaxis.grid()
plt.tight_layout()
plt.savefig('grouped_bar_plot_3_groups_2_bars.png')
plt.show()
