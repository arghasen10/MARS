import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 20})
plt.rcParams["figure.figsize"] = (16, 7)  # Adjust figure size for more types
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

models = ['MARS', 'RadHAR', 'Vid2Doppler', 'VGG-16', 'Pantomime']
types = [
    'walking', 'running', 'jumping', 'clapping', 'lunges', 'squats', 'waving',
    'vacuum\ncleaning', 'folding\nclothes', 'changing\nclothes', 'laptop\ntyping',
    'phone\ntalking', 'phone\ntyping', 'sitting', 'playing\nguitar', 'eating\nfood',
    'combing\nhair', 'brushing\nteeth', 'drinking\nwater'
]

accuracy = {}
for model in models:
    accuracy[model] = np.loadtxt(f'{model}_accuracy.txt')

fig, ax = plt.subplots()

bar_width = 0.15
index = np.arange(len(types))

for i, model in enumerate(models):
    ax.bar(index + i * bar_width, accuracy[model], bar_width, label=model)

ax.set_ylabel('Accuracy')
ax.set_xticks(index + bar_width * 2)
ax.set_xticklabels(types, rotation=45, ha='right')
ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.15),fontsize=18)
ax.yaxis.grid()
plt.tight_layout()
plt.savefig('individual_f1_2.eps')
plt.show()
