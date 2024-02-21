import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 24})
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
models = ['MARS', 'RadHAR', 'Vid2Doppler', 'VGG-16', 'Pantomime']
types = ['macro', 'micro']
accuracy = {
    'MARS': [0.97, 0.94],
    'RadHAR': [0.83, 0.69],
    'Vid2Doppler': [0.75, 0.71],
    'VGG-16': [0.69, 0.68],
    'Pantomime': [0.83, 0.73]
}

fig, ax = plt.subplots()

bar_width = 1
index = np.arange(len(types))

for i, model in enumerate(models):
    print(index + i * bar_width)
    print(accuracy[model])
    ax.bar(np.array([0,6])+i, accuracy[model], bar_width, label=model)

ax.set_ylabel('Accuracy')
ax.set_xticks([2,8])
ax.set_xticklabels(types)
ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=18)
ax.yaxis.grid()
plt.tight_layout()
plt.savefig('baseline_comp_2.eps')
plt.show()
