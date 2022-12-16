import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
import pickle
from imblearn.over_sampling import SMOTE

plt.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

with open("rf_save.pkl", "rb") as f:
    rf = pickle.load(f)

with open("rf_data.pkl", "rb") as f:
    rf_classifier_dataset = pickle.load(f)

data, label = rf_classifier_dataset

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=101)
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

pred = rf.predict(X_test)
conf_matrix = confusion_matrix(y_test, pred)
class_report = classification_report(y_test, pred)
f1 = f1_score(y_test, pred, average="weighted")
result = "confusion matrix\n" + repr(
    conf_matrix) + "\n" + "report\n" + class_report + "\nf1_score(weighted)\n" + repr(f1)
print(result)
total = conf_matrix / conf_matrix.sum(axis=1).reshape(-1, 1)
total = np.round(total,2)
labels = ['macro', 'micro', 'walking\nor running']
df_cm = pd.DataFrame(total, index=[i for i in labels], columns=[i for i in labels])
sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues")
plt.savefig('rf_classification.pdf')
plt.show()
