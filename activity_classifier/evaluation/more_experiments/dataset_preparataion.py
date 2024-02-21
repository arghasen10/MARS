#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import csv
from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
from collections import Counter
import pickle
from imblearn.over_sampling import SMOTE


# In[2]:


def convert_to_neumeric(label):
    lbl_map = \
        {'macro':0,
         'micro':1,
         'track':2
         }
    return np.array(list(map(lambda e: lbl_map[e], label)))


# In[3]:


def map_user(u):
    u_map = \
    {'anirban':1, 
     'sugandh':2, 
     'salma':3, 
     'aritra':4,
     'avijit':5,
     'debasree':6,
     'foss':7
#      'utkalika':6,
#      'prasenjit':7, 
#      'debasree':8, 
#      'sandipan':9,
     }
    return 'User-'+str(u_map[u])


# In[4]:


def map_activity(a):
    a_map = \
    {
        'Clapping':'macro',
        'jumping':'macro',
        'laptop-typing':'micro',
        'lunges':'macro',
        'phone-talking':'micro',
        'phone-typing':'micro',
        'running':'track',
        'sitting':'micro',
        'squats':'macro',
        'walking':'track',
        'waving':'macro',
        'playing-guitar':'micro',
        'eating-food':'micro'
    }
    return a_map[a]


# In[5]:


def process_mmwave(f):
    u = map_user(f.split('/')[1].split('_')[0])
    data = [json.loads(val) for val in open(f, "r")]
    datetime_str = datetime.strftime(datetime.strptime(data[0]['datenow'], "%d/%m/%Y") + relativedelta(months=1), "%Y-%m-%d")+' '
    mmwave_df = pd.DataFrame()
    for d in data:
        mmwave_df = mmwave_df.append(d, ignore_index=True)
    mmwave_df['datetime'] = mmwave_df['timenow'].apply(lambda e: datetime_str + ':'.join(e.split('_')))
    mmwave_df['User'] = u
    if 'doppz' in mmwave_df.columns:
        print('True')
        mmwave_df['doppz'] = list(np.array(mmwave_df['doppz'].values.tolist()))
        mmwave_df = mmwave_df[['datetime', 'rangeIdx', 'dopplerIdx', 'numDetectedObj', 'range', 'peakVal', 
                               'x_coord', 'y_coord', 'doppz']]
    else:
        return
    return mmwave_df


# In[6]:


def read_mmwave():
    mmwave_files = glob.glob('mmWave_data/*.txt')
    return pd.concat([process_mmwave(f) for f in mmwave_files], ignore_index=True)


# In[7]:


def process_activity():
    activity_files = glob.glob('*.csv')
    activity_df = pd.concat([pd.read_csv(f) for f in activity_files], ignore_index=True)
    activity_df['Datetime'] = activity_df['Datetime'].map(lambda e: e.replace('+AC0', ''))
    activity_df['User'] = activity_df['User'].map(lambda e: e.replace('+AC0', ''))
    activity_df['Activity'] = activity_df['Activity'].map(lambda e: e.replace('+AF8', ''))
    activity_df['datetime'] = activity_df['Datetime']
    activity_df = activity_df[['datetime', 'Activity', 'User', 'Position', 'Orientation']]
    return activity_df


# In[8]:


mmwave_df = read_mmwave()


# In[9]:


activity_df = process_activity()


# In[10]:


merged_df = mmwave_df.merge(activity_df, left_on= 'datetime', right_on= 'datetime', how='inner')


# In[11]:


merged_df['activity_class'] = merged_df['Activity'].map(lambda x: map_activity(x))


# In[12]:


merged_df = merged_df[['datetime', 'rangeIdx', 'dopplerIdx', 'numDetectedObj', 'range', 'peakVal', 
                        'x_coord', 'y_coord', 'doppz', 'Activity', 'User', 'Position', 'Orientation', 
                        'activity_class']]


# In[13]:


df = merged_df.copy()


# In[19]:


df[df.activity_class == 'micro'].to_pickle('micro_df2.pkl')


# ## Random Forest Classifier

# In[17]:


dfs = {x:df[df.datetime == x ] for x in df.datetime.unique()}


# In[18]:


data = []
timestamp = []
act = []
my_df = pd.DataFrame()
for e, val in dfs.items():
    range_elem = np.concatenate(np.array(val['rangeIdx'].values))
    dop_elem = np.concatenate(np.array(val['dopplerIdx'].values))
    x_elem = np.concatenate(np.array(val['x_coord'].values))
    y_elem = np.concatenate(np.array(val['y_coord'].values))
    range_mean, range_std, range_kur, range_sq = range_elem.mean(), range_elem.std(),kurtosis(range_elem),skew(range_elem)
    dop_mean, dop_std, dop_kur, dop_sq = dop_elem.mean(), dop_elem.std(),kurtosis(dop_elem),skew(dop_elem)
    x_mean, x_std, x_kur, x_sq = x_elem.mean(), x_elem.std(),kurtosis(x_elem),skew(x_elem)
    y_mean, y_std, y_kur, y_sq = y_elem.mean(), y_elem.std(),kurtosis(y_elem),skew(y_elem)
    t, f, a = e, [range_mean, range_std, range_kur, range_sq, dop_mean, dop_std, dop_kur, dop_sq, x_mean, x_std, x_kur, x_sq, y_mean, y_std, y_kur, y_sq], val['activity_class'].values[0]
    timestamp.append(t)
    data.append(f)
    act.append(a)


# In[19]:


my_df['datetime'] = timestamp
my_df['features'] = data
my_df['act'] = act


# In[20]:


my_df.groupby('act').size()


# In[21]:


data = np.array(my_df['features'].values.tolist())
label = my_df['act'].values
label = convert_to_neumeric(label)


# In[22]:


data.shape


# In[23]:


def stack_frame(data, label, frame_stack=4):
    max_index = data.shape[0] - frame_stack
    stacked_data = np.array([data[i:i + frame_stack] for i in range(max_index)])
    new_labels = np.array([label[i + frame_stack - 1] for i in range(max_index)])
    p = stacked_data.mean(axis=1)
    print(p.shape)
    print(new_labels.shape)
    return p, new_labels


# In[24]:


data, label = stack_frame(data, label)


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=101)
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)


# In[26]:


label.shape


# In[27]:


rf = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=10)
rf.fit(X_train,y_train)


# In[28]:


rf.score(X_train, y_train)


# In[29]:


rf.score(X_test, y_test)


# In[30]:


n_estimators = [int(x) for x in np.linspace(start = 20, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[31]:


rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)


# In[32]:


rf_random.best_params_


# In[33]:


best_random = rf_random.best_estimator_


# In[34]:


pred = best_random.predict(X_test)
conf_matrix = confusion_matrix(y_test, pred)
class_report = classification_report(y_test, pred)
f1 = f1_score(y_test, pred, average="weighted")
result = "confusion matrix\n" + repr(
    conf_matrix) + "\n" + "report\n" + class_report + "\nf1_score(weighted)\n" + repr(f1)
print(result)
total = conf_matrix / conf_matrix.sum(axis=1).reshape(-1, 1)
total = np.round(total,2)
labels = ['macro', 'micro', 'track']
df_cm = pd.DataFrame(total, index=[i for i in labels], columns=[i for i in labels])
sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues")
plt.show()


# In[37]:


# with open("rf_save.pkl", "wb") as f:
#     pickle.dump(best_random, f)


# In[38]:


with open("rf_save.pkl", "rb") as f:
    x = pickle.load(f)



# In[42]:


rf_classifier_dataset = (data, label)


# In[46]:


with open("evaluation/rf_data.pkl", "rb") as f:
    rf_classifier_dataset2 = pickle.load(f)


# In[50]:


data, label = rf_classifier_dataset2


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=101)
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)


# In[52]:


pred = x.predict(X_test)
conf_matrix = confusion_matrix(y_test, pred)
class_report = classification_report(y_test, pred)
f1 = f1_score(y_test, pred, average="weighted")
result = "confusion matrix\n" + repr(
    conf_matrix) + "\n" + "report\n" + class_report + "\nf1_score(weighted)\n" + repr(f1)
print(result)
total = conf_matrix / conf_matrix.sum(axis=1).reshape(-1, 1)
total = np.round(total,2)
labels = ['macro', 'micro', 'track']
df_cm = pd.DataFrame(total, index=[i for i in labels], columns=[i for i in labels])
sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues")
plt.show()

